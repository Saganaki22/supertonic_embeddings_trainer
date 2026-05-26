import os
import sys
import glob
import time
import argparse
import numpy as np
import torch

from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
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


def find_closest_style(styles_dir, model, target_wav_path,
                       text_ids, text_mask, noisy_latent, latent_mask,
                       vocoder_steps, speed):
    from pipeline.utils.loss import load_wavlm, extract_wavlm_targets, wavlm_loss

    wavlm = load_wavlm(DEVICE)
    import soundfile as sf
    target_np, _ = sf.read(target_wav_path, dtype="float32")
    target_t = torch.tensor(target_np, dtype=torch.float32).to(DEVICE)
    target_feats = extract_wavlm_targets(wavlm, target_t, DEVICE)

    all_styles = sorted(glob.glob(os.path.join(styles_dir, "[FM]*.json")))
    if not all_styles:
        return None, None
    best_dist = float("inf")
    best_path = None
    for sp in all_styles:
        s_ttl, _ = load_single_voice_style(sp)
        with torch.no_grad():
            wav = model(text_ids, text_mask, s_ttl, vocoder_steps, noisy_latent, latent_mask)
            dist = wavlm_loss(wavlm, wav.squeeze(), target_feats, DEVICE).item()
        print(f"    {os.path.basename(sp)}: {dist:.4f}")
        if dist < best_dist:
            best_dist = dist
            best_path = sp
    print(f"  >> Best: {os.path.basename(best_path)} (dist={best_dist:.4f})")
    return load_single_voice_style(best_path)


def train(args, on_step=None):
    config = TrainConfig()
    name = args.name if args.name is not None else config.NAME
    gender = args.gender if args.gender is not None else config.GENDER
    wav_path = args.wav if args.wav is not None else config.TARGET_WAV_PATH
    ref_style = args.reference_style if args.reference_style is not None else config.REFERENCE_STYLE
    seed = args.seed if args.seed is not None else config.SEED
    speed = args.speed if args.speed is not None else config.SPEED
    vocoder_steps = args.vocoder_steps if args.vocoder_steps is not None else config.VOCODER_STEPS
    num_steps = args.num_steps if args.num_steps is not None else config.NUM_STEPS
    lr = args.lr if args.lr is not None else config.LEARNING_RATE
    save_steps = args.save_steps if args.save_steps is not None else config.SAVE_STEPS
    threshold = args.threshold if args.threshold is not None else config.EARLY_STOP_LOSS_THRESHOLD

    onnx_dir = args.onnx_dir if hasattr(args, "onnx_dir") and args.onnx_dir else str(PIPELINE_DIR / "onnx_v2")
    styles_dir = args.styles_dir if hasattr(args, "styles_dir") and args.styles_dir else str(PIPELINE_DIR.parent / "reference_styles_v2")

    print(f"Name: {name}  Gender: {gender}  Device: {DEVICE}")
    print(f"Target WAV: {wav_path}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = os.path.join(LOGS_DIR, name)
    os.makedirs(log_dir, exist_ok=True)

    tts = load_text_to_speech(onnx_dir)
    dataloader = get_train_dataloader(tts, texts)
    del tts
    data_iter = iter(dataloader)

    tmp_ids, tmp_mask = next(data_iter)

    model = SupertonicModel(onnx_dir)

    if ref_style == "auto":
        print("\nFinding closest reference style (WavLM Layer 3)...")
        dummy_dp = load_single_voice_style(os.path.join(styles_dir, "M1.json"))[1]
        with torch.no_grad():
            init_dur_tmp = model.dp_model(tmp_ids, dummy_dp, tmp_mask) / speed
            init_dur_tmp = init_dur_tmp.detach().cpu().numpy()
        tts_tmp = load_text_to_speech(onnx_dir)
        noisy_tmp, lmask_tmp = tts_tmp.sample_noisy_latent(duration=init_dur_tmp)
        noisy_tmp = torch.tensor(noisy_tmp, dtype=torch.float32).to(DEVICE)
        lmask_tmp = torch.tensor(lmask_tmp, dtype=torch.float32).to(DEVICE)
        del tts_tmp

        style_ttl, style_dp = find_closest_style(
            styles_dir, model, wav_path,
            tmp_ids, tmp_mask, noisy_tmp, lmask_tmp, vocoder_steps, speed
        )
        del noisy_tmp, lmask_tmp
        if style_ttl is None:
            print("  No reference styles found, using random init")
            _, style_dp = load_single_voice_style(os.path.join(styles_dir, "M1.json"))
            style_ttl = torch.randn(1, 50, 256, device=DEVICE) * 0.1
    elif ref_style and os.path.exists(ref_style):
        print(f"\nLoading reference: {ref_style}")
        style_ttl, style_dp = load_single_voice_style(ref_style)
    else:
        print("\nRandom init (not recommended)")
        _, style_dp = load_single_voice_style(os.path.join(styles_dir, "M1.json"))
        style_ttl = torch.randn(1, 50, 256, device=DEVICE) * 0.1

    tts = load_text_to_speech(onnx_dir)
    with torch.no_grad():
        init_dur = model.dp_model(tmp_ids, style_dp, tmp_mask) / speed
        init_dur = init_dur.detach().cpu().numpy()
    noisy_fixed, lmask = tts.sample_noisy_latent(duration=init_dur)
    noisy_fixed = torch.tensor(noisy_fixed, dtype=torch.float32).to(DEVICE)
    lmask = torch.tensor(lmask, dtype=torch.float32).to(DEVICE)
    del tts

    style_ttl = style_ttl.clone().requires_grad_(True)
    style_dp = style_dp.detach().clone()
    print(f"  style_ttl: {style_ttl.shape}  style_dp: {style_dp.shape} (dp frozen)")

    optimizer = torch.optim.Adam([style_ttl], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=lr * 0.01)

    best_loss = float("inf")
    best_ttl = None
    best_dp = style_dp.detach().clone()
    start_time = time.time()

    from pipeline.utils.loss import load_wavlm, extract_wavlm_targets, wavlm_loss
    wavlm = load_wavlm(DEVICE)
    import soundfile as sf
    target_np, _ = sf.read(wav_path, dtype="float32")
    target_t = torch.tensor(target_np, dtype=torch.float32).to(DEVICE)
    target_feats = extract_wavlm_targets(wavlm, target_t, DEVICE)

    print(f"\nOptimizing ({num_steps} steps, early stop at {threshold})...")
    for step in range(num_steps):
        try:
            text_ids, text_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            text_ids, text_mask = next(data_iter)

        text_ids = text_ids.to(DEVICE)
        text_mask = text_mask.to(DEVICE)

        optimizer.zero_grad()
        wav_out = model(text_ids, text_mask, style_ttl, vocoder_steps, noisy_fixed, lmask)
        loss = wavlm_loss(wavlm, wav_out.squeeze(), target_feats, DEVICE)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([style_ttl], max_norm=1.0)
        optimizer.step()

        step_loss = loss.detach().item()
        if step_loss < best_loss:
            best_loss = step_loss
            best_ttl = style_ttl.detach().clone()

        scheduler.step(best_loss)

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
    p.add_argument("--vocoder_steps", type=int, default=5)
    p.add_argument("--num_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--threshold", type=float, default=0.24)
    p.add_argument("--onnx_dir", default=None, help="ONNX model directory")
    p.add_argument("--styles_dir", default=None, help="Reference styles directory")
    train(p.parse_args())
