"""
Train a voice style JSON from a WAV file for Supertonic TTS.

Approach: gradient-based inverse optimization.
  1. Convert ONNX TTS models to PyTorch (enables backprop)
  2. Initialize style_ttl from the closest reference voice
  3. Optimize style_ttl to maximize speaker similarity (SpeechBrain ECAPA)
  4. style_dp is frozen (copied from reference)

Usage:
    python train_style.py --wav voices/my_voice.wav --name my_voice --gender F
    python train_style.py --wav voices/my_voice.wav --name my_voice --reference_style auto
    python train_style.py --config configs/custom.py
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import torch

from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
ONNX_DIR = str(PIPELINE_DIR / "onnx")
VOICE_STYLES_DIR = str(PIPELINE_DIR.parent / "reference_styles")
LOGS_DIR = str(PIPELINE_DIR / "logs")

sys.path.insert(0, str(PIPELINE_DIR))
from helper import load_text_to_speech, load_voice_style
from utils import save_style, SupertonicModel, get_train_dataloader
from configs import texts, TrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_single_voice_style(path):
    style = load_voice_style([path])
    return (
        torch.tensor(style.ttl, dtype=torch.float32).to(DEVICE),
        torch.tensor(style.dp, dtype=torch.float32).to(DEVICE),
    )


def find_closest_style(wavlm_loss, target_wav_44k, dp_model, te_model, ve_model, voc_model,
                       text_ids, text_mask, noisy_latent, latent_mask, vocoder_steps, speed):
    all_styles = sorted(glob.glob(os.path.join(VOICE_STYLES_DIR, "[FM]*.json")))
    if not all_styles:
        return None, None
    best_dist = float("inf")
    best_path = None
    for sp in all_styles:
        s_ttl, s_dp = load_single_voice_style(sp)
        with torch.no_grad():
            text_emb = te_model(text_ids, s_ttl, text_mask)
            xt = noisy_latent * latent_mask
            total_step_t = torch.tensor([vocoder_steps], dtype=torch.float32).to(DEVICE)
            for step in range(vocoder_steps):
                current_step_t = torch.tensor([step], dtype=torch.float32).to(DEVICE)
                xt = ve_model(xt, text_emb, s_ttl, latent_mask, text_mask, current_step_t, total_step_t)
            wav = voc_model(xt)
            dist = wavlm_loss(wav).item()
        print(f"    {os.path.basename(sp)}: {dist:.4f}")
        if dist < best_dist:
            best_dist = dist
            best_path = sp
    print(f"  >> Best: {os.path.basename(best_path)} (dist={best_dist:.4f})")
    return load_single_voice_style(best_path)


def train(args, on_step=None):
    config = TrainConfig()
    name = args.name or config.NAME
    gender = args.gender or config.GENDER
    wav_path = args.wav or config.TARGET_WAV_PATH
    ref_style = args.reference_style or config.REFERENCE_STYLE
    seed = args.seed or config.SEED
    speed = args.speed or config.SPEED
    vocoder_steps = args.vocoder_steps or config.VOCODER_STEPS
    num_steps = args.num_steps or config.NUM_STEPS
    lr = args.lr or config.LEARNING_RATE
    save_steps = args.save_steps or config.SAVE_STEPS
    threshold = args.threshold or config.EARLY_STOP_LOSS_THRESHOLD

    print(f"Name: {name}  Gender: {gender}  Device: {DEVICE}")
    print(f"Target WAV: {wav_path}")

    log_dir = os.path.join(LOGS_DIR, name)
    os.makedirs(log_dir, exist_ok=True)

    model = SupertonicModel(ONNX_DIR, wav_path)
    tts = load_text_to_speech(ONNX_DIR)

    dataloader = get_train_dataloader(tts, texts)
    data_iter = iter(dataloader)

    torch.manual_seed(seed)
    np.random.seed(seed)

    tmp_ids, tmp_mask = next(data_iter)
    tmp_voice = "F4.json" if gender == "F" else "M1.json"
    tmp_ttl, tmp_dp = load_single_voice_style(os.path.join(VOICE_STYLES_DIR, tmp_voice))

    with torch.no_grad():
        init_dur = model.dp_model(tmp_ids, tmp_dp, tmp_mask) / speed
        init_dur = init_dur.detach().cpu().numpy()
    noisy_fixed, lmask = tts.sample_noisy_latent(duration=init_dur)
    noisy_fixed = torch.tensor(noisy_fixed, dtype=torch.float32).to(DEVICE)
    lmask = torch.tensor(lmask, dtype=torch.float32).to(DEVICE)

    del tmp_ttl, tmp_dp, tts

    if ref_style == "auto":
        print("\nFinding closest reference style (WavLM Layer 3)...")
        style_ttl, style_dp = find_closest_style(
            model.voice_encoder, wav_path,
            model.dp_model, model.te_model, model.ve_model, model.voc_model,
            tmp_ids, tmp_mask, noisy_fixed, lmask, vocoder_steps, speed
        )
        if style_ttl is None:
            print("  No reference styles found, using random init")
            _, style_dp = load_single_voice_style(os.path.join(VOICE_STYLES_DIR, tmp_voice))
            style_ttl = torch.randn(1, 50, 256, device=DEVICE) * 0.1
    elif ref_style and os.path.exists(ref_style):
        print(f"\nLoading reference: {ref_style}")
        style_ttl, style_dp = load_single_voice_style(ref_style)
    else:
        print("\nRandom init (not recommended)")
        _, style_dp = load_single_voice_style(os.path.join(VOICE_STYLES_DIR, tmp_voice))
        style_ttl = torch.randn(1, 50, 256, device=DEVICE) * 0.1

    style_ttl = style_ttl.clone().requires_grad_(True)
    style_dp = style_dp.detach().clone()
    print(f"  style_ttl: {style_ttl.shape}  style_dp: {style_dp.shape} (dp frozen)")

    optimizer = torch.optim.Adam([style_ttl], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=lr * 0.01)
    optimizer.zero_grad()

    best_loss = float("inf")
    best_ttl = None
    best_dp = style_dp.detach().clone()
    best_dp = style_dp.detach().clone()
    start_time = time.time()

    print(f"\nOptimizing ({num_steps} steps, early stop at {threshold})...")
    for step in range(num_steps):
        try:
            text_ids, text_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            text_ids, text_mask = next(data_iter)

        text_ids = text_ids.to(DEVICE)
        text_mask = text_mask.to(DEVICE)

        _, loss = model(text_ids, text_mask, style_ttl, vocoder_steps, noisy_fixed, lmask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([style_ttl], max_norm=1.0)
        optimizer.step()
        scheduler.step(loss)
        optimizer.zero_grad()

        step_loss = loss.detach().item()
        if step_loss < best_loss:
            best_loss = step_loss
            best_ttl = style_ttl.detach().clone()

        if (step + 1) % 8 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"  Step {step+1}/{num_steps} | Loss: {step_loss:.4f} | LR: {cur_lr:.6f} | Best: {best_loss:.4f}")
            if on_step:
                on_step(step + 1, num_steps, step_loss, best_loss, cur_lr)

        if (step + 1) % save_steps == 0:
            ckpt = os.path.join(log_dir, f"{name}_{step+1:04d}.json")
            save_style(ckpt, best_ttl, best_dp, wav_path)
            print(f"  >> Checkpoint: {ckpt}")

        if best_loss <= threshold:
            print(f"  Early stop at step {step+1}: {best_loss:.4f} <= {threshold}")
            break

    final_path = os.path.join(log_dir, f"{name}.json")
    save_style(final_path, best_ttl, best_dp, wav_path)
    elapsed = time.time() - start_time
    print(f"\nDone! Best loss: {best_loss:.4f} | Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"Saved: {final_path}")
    return final_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wav", help="Target WAV file to clone")
    p.add_argument("--name", default="custom_voice", help="Output voice name")
    p.add_argument("--gender", choices=["F", "M"], default="F")
    p.add_argument("--reference_style", default="auto", help="auto | path/to/style.json | none")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--vocoder_steps", type=int, default=6)
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.15)
    train(p.parse_args())
