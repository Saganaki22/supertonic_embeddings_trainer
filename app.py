"""
Supertonic Embeddings Trainer
Upload WAV → Train Style JSON → Synthesize Speech
Supports: stop mid-training, resume from checkpoint
"""

import gradio as gr
import os
import sys
import shutil
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))

from pipeline.helper import load_text_to_speech, load_voice_style, timer
from pipeline.configs.utterances import texts

ONNX_DIR = PIPELINE_DIR / "pipeline" / "onnx"
VOICES_DIR = PIPELINE_DIR / "pipeline" / "voices"
SAMPLES_DIR = PIPELINE_DIR / "pipeline" / "samples"
LOGS_DIR = PIPELINE_DIR / "pipeline" / "logs"
REF_DIR = PIPELINE_DIR / "reference_styles"

for d in [VOICES_DIR, SAMPLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_stop_flag = {"stop": False}


def check_onnx_models():
    if not ONNX_DIR.exists():
        return False
    required = ["tts.json", "duration_predictor.onnx", "text_encoder.onnx",
                "vector_estimator.onnx", "vocoder.onnx", "unicode_indexer.json"]
    return all((ONNX_DIR / f).exists() for f in required)


def download_models(progress=gr.Progress()):
    if check_onnx_models():
        return "Models already downloaded."

    progress(0, desc="Downloading Supertonic-2 ONNX models...")
    try:
        from huggingface_hub import snapshot_download
        target = str(PIPELINE_DIR / "pipeline" / "supertonic2")
        snapshot_download("Supertone/supertonic-2", local_dir=target)
        onnx_src = os.path.join(target, "onnx")
        styles_src = os.path.join(target, "voice_styles")

        if os.path.exists(onnx_src) and not ONNX_DIR.exists():
            shutil.copytree(onnx_src, str(ONNX_DIR))
        elif not ONNX_DIR.exists():
            os.makedirs(str(ONNX_DIR), exist_ok=True)
            for f in ["tts.json", "duration_predictor.onnx", "text_encoder.onnx",
                      "vector_estimator.onnx", "vocoder.onnx", "unicode_indexer.json"]:
                src = os.path.join(target, "onnx", f)
                if os.path.exists(src):
                    shutil.copy2(src, str(ONNX_DIR / f))

        if os.path.exists(styles_src):
            for f in os.listdir(styles_src):
                if f.endswith(".json"):
                    shutil.copy2(os.path.join(styles_src, f), str(REF_DIR / f))

        progress(1.0, desc="Done!")
        return "Models downloaded successfully."
    except Exception as e:
        return f"Download failed: {e}\nTry manually: hf download Supertone/supertonic-2 --local-dir pipeline/supertonic2"


def stop_training():
    _stop_flag["stop"] = True
    return "Stopping... (will save checkpoint)"


def check_resume(name):
    log_dir = LOGS_DIR / name
    state_path = log_dir / "training_state.pt"
    if state_path.exists():
        import torch
        state = torch.load(str(state_path), weights_only=False, map_location="cpu")
        step = state.get("step", 0)
        best = state.get("best_loss", float("inf"))
        return f"Found checkpoint at step {step} (best loss: {best:.4f}). Training will resume from here."
    return "No checkpoint found. Starting fresh."


def train_voice(wav_path, name, gender, ref_mode, num_steps, save_every, lr, threshold, progress=gr.Progress()):
    if not wav_path:
        yield "Upload a WAV file first.", None
        return

    if not check_onnx_models():
        yield "Models not downloaded. Click 'Download Models' first.", None
        return

    name = name.strip().replace(" ", "_") or "custom_voice"
    dst = str(VOICES_DIR / f"{name}.wav")

    import torch
    import numpy as np
    import librosa
    import argparse

    _stop_flag["stop"] = False

    if not os.path.exists(dst):
        y, sr = librosa.load(wav_path, sr=44100, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak * 0.95
        import soundfile as sf
        sf.write(dst, y, 44100)

    progress(0.0, desc="Loading models...")

    from pipeline.train_style import train as do_train

    save_every_int = int(save_every)
    total = int(num_steps)
    status_lines = []

    def on_step(step, total_steps, loss, best, lr_val):
        pct = step / total_steps
        elapsed_s = time.time() - on_step.t0 if hasattr(on_step, "t0") else 0
        remaining_s = (elapsed_s / step) * (total_steps - step) if step > 0 else 0
        status_lines.clear()
        status_lines.append(f"Step {step}/{total_steps} ({pct*100:.1f}%)")
        status_lines.append(f"Loss: {loss:.4f} | Best: {best:.4f} | LR: {lr_val:.6f}")
        if elapsed_s > 0:
            status_lines.append(f"Elapsed: {elapsed_s/60:.1f}min | ETA: {remaining_s/60:.1f}min")
        if best <= float(threshold):
            status_lines.append(f"Target reached: {best:.4f} <= {threshold}")
        progress(pct, desc=f"Training: step {step}/{total_steps} | loss={loss:.4f} | best={best:.4f}")

    args = argparse.Namespace(
        wav=dst,
        name=name,
        gender=gender,
        reference_style=ref_mode if ref_mode != "none" else None,
        seed=42,
        speed=1.05,
        vocoder_steps=5,
        num_steps=total,
        lr=float(lr),
        save_steps=save_every_int,
        threshold=float(threshold),
    )

    try:
        on_step.t0 = time.time()

        import pipeline.train_style as ts
        config = ts.TrainConfig()
        gender_val = args.gender if args.gender is not None else config.GENDER
        wav_val = args.wav if args.wav is not None else config.TARGET_WAV_PATH
        ref_val = args.reference_style if args.reference_style is not None else config.REFERENCE_STYLE
        seed_val = args.seed if args.seed is not None else config.SEED
        speed_val = args.speed if args.speed is not None else config.SPEED
        vocoder_steps_val = args.vocoder_steps if args.vocoder_steps is not None else config.VOCODER_STEPS
        num_steps_val = args.num_steps if args.num_steps is not None else config.NUM_STEPS
        lr_val = args.lr if args.lr is not None else config.LEARNING_RATE
        save_steps_val = args.save_steps if args.save_steps is not None else config.SAVE_STEPS
        threshold_val = args.threshold if args.threshold is not None else config.EARLY_STOP_LOSS_THRESHOLD

        log_dir = str(LOGS_DIR / name)
        os.makedirs(log_dir, exist_ok=True)
        state_path = os.path.join(log_dir, "training_state.pt")

        DEVICE = ts.DEVICE

        resumed = False
        if os.path.exists(state_path):
            print(f"Resuming from checkpoint: {state_path}")
            state = torch.load(state_path, weights_only=False, map_location=DEVICE)
            start_step = state["step"]
            best_loss = state["best_loss"]
            best_ttl = state["best_ttl"].to(DEVICE)
            best_dp = state["best_dp"].to(DEVICE)
            style_ttl = state["style_ttl"].to(DEVICE).requires_grad_(True)
            style_dp = state["style_dp"].to(DEVICE)
            noisy_fixed = state["noisy_fixed"].to(DEVICE)
            lmask = state["lmask"].to(DEVICE)
            resumed = True

            torch.manual_seed(seed_val)
            np.random.seed(seed_val)

            model = ts.SupertonicModel(str(ONNX_DIR), wav_val)

            optimizer = torch.optim.Adam([style_ttl], lr=lr_val)
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=lr_val * 0.01)
            scheduler.load_state_dict(state["scheduler_state"])

            tts = ts.load_text_to_speech(str(ONNX_DIR))
            dataloader = ts.get_train_dataloader(tts, ts.texts)
            del tts
            data_iter = iter(dataloader)
            for _ in range(start_step % len(ts.texts)):
                try:
                    next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    next(data_iter)

            on_step.t0 = time.time()
            status_lines.append(f"Resumed from step {start_step} (best: {best_loss:.4f})")
            print(f"  Resumed: step {start_step}, best_loss={best_loss:.4f}")
        else:
            torch.manual_seed(seed_val)
            np.random.seed(seed_val)

            print(f"Name: {name}  Gender: {gender_val}  Device: {DEVICE}")
            print(f"Target WAV: {wav_val}")

            tts = ts.load_text_to_speech(str(ONNX_DIR))
            dataloader = ts.get_train_dataloader(tts, ts.texts)
            del tts
            data_iter = iter(dataloader)

            tmp_ids, tmp_mask = next(data_iter)

            model = ts.SupertonicModel(str(ONNX_DIR), wav_val)

            if ref_val == "auto":
                print("\nFinding closest reference style (WavLM Layer 3)...")
                dummy_dp = ts.load_single_voice_style(os.path.join(str(REF_DIR), "M1.json"))[1]
                with torch.no_grad():
                    init_dur = model.dp_model(tmp_ids, dummy_dp, tmp_mask) / speed_val
                    init_dur = init_dur.detach().cpu().numpy()
                tts_tmp = ts.load_text_to_speech(str(ONNX_DIR))
                noisy_tmp, lmask_tmp = tts_tmp.sample_noisy_latent(duration=init_dur)
                noisy_tmp = torch.tensor(noisy_tmp, dtype=torch.float32).to(DEVICE)
                lmask_tmp = torch.tensor(lmask_tmp, dtype=torch.float32).to(DEVICE)
                del tts_tmp

                style_ttl, style_dp = ts.find_closest_style(
                    model.voice_encoder, wav_val,
                    model.dp_model, model.te_model, model.ve_model, model.voc_model,
                    tmp_ids, tmp_mask, noisy_tmp, lmask_tmp, vocoder_steps_val, speed_val
                )
                del noisy_tmp, lmask_tmp
                if style_ttl is None:
                    _, style_dp = ts.load_single_voice_style(os.path.join(str(REF_DIR), "M1.json"))
                    style_ttl = torch.randn(1, 50, 256, device=DEVICE) * 0.1
            elif ref_val and os.path.exists(ref_val):
                style_ttl, style_dp = ts.load_single_voice_style(ref_val)
            else:
                _, style_dp = ts.load_single_voice_style(os.path.join(str(REF_DIR), "M1.json"))
                style_ttl = torch.randn(1, 50, 256, device=DEVICE) * 0.1

            tts = ts.load_text_to_speech(str(ONNX_DIR))
            with torch.no_grad():
                init_dur = model.dp_model(tmp_ids, style_dp, tmp_mask) / speed_val
                init_dur = init_dur.detach().cpu().numpy()
            noisy_fixed, lmask = tts.sample_noisy_latent(duration=init_dur)
            noisy_fixed = torch.tensor(noisy_fixed, dtype=torch.float32).to(DEVICE)
            lmask = torch.tensor(lmask, dtype=torch.float32).to(DEVICE)
            del tts

            style_ttl = style_ttl.clone().requires_grad_(True)
            style_dp = style_dp.detach().clone()

            optimizer = torch.optim.Adam([style_ttl], lr=lr_val)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5, min_lr=lr_val * 0.01)

            best_loss = float("inf")
            best_ttl = None
            best_dp = style_dp.detach().clone()
            start_step = 0
            on_step.t0 = time.time()

        print(f"\nOptimizing (step {start_step} -> {num_steps_val}, early stop at {threshold_val})...")
        for step in range(start_step, num_steps_val):
            if _stop_flag["stop"]:
                print(f"  Stopped at step {step}. Saving state...")
                torch.save({
                    "step": step,
                    "best_loss": best_loss,
                    "best_ttl": best_ttl.cpu(),
                    "best_dp": best_dp.cpu(),
                    "style_ttl": style_ttl.detach().cpu(),
                    "style_dp": style_dp.cpu(),
                    "noisy_fixed": noisy_fixed.cpu(),
                    "lmask": lmask.cpu(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }, state_path)
                ts.save_style(os.path.join(log_dir, f"{name}.json"), best_ttl, best_dp, wav_val)
                stop_msg = f"Training stopped at step {step}.\nBest loss: {best_loss:.4f}\nState saved. Re-train with same name to resume."
                print(stop_msg)
                yield stop_msg, None
                return

            try:
                text_ids, text_mask = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                text_ids, text_mask = next(data_iter)

            text_ids = text_ids.to(DEVICE)
            text_mask = text_mask.to(DEVICE)

            optimizer.zero_grad()
            _, loss = model(text_ids, text_mask, style_ttl, vocoder_steps_val, noisy_fixed, lmask)
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
                print(f"  Step {step+1}/{num_steps_val} | Loss: {step_loss:.4f} | LR: {cur_lr:.6f} | Best: {best_loss:.4f}")
                on_step(step + 1, num_steps_val, step_loss, best_loss, cur_lr)
                yield "\n".join(status_lines), None

            if (step + 1) % save_steps_val == 0:
                ckpt = os.path.join(log_dir, f"{name}_{step+1:04d}.json")
                ts.save_style(ckpt, best_ttl, best_dp, wav_val)
                torch.save({
                    "step": step + 1,
                    "best_loss": best_loss,
                    "best_ttl": best_ttl.cpu(),
                    "best_dp": best_dp.cpu(),
                    "style_ttl": style_ttl.detach().cpu(),
                    "style_dp": style_dp.cpu(),
                    "noisy_fixed": noisy_fixed.cpu(),
                    "lmask": lmask.cpu(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                }, state_path)
                print(f"  >> Checkpoint: {ckpt}")
                status_lines.append(f"Saved checkpoint: {ckpt}")
                yield "\n".join(status_lines), None

            if best_loss <= threshold_val:
                print(f"  Early stop at step {step+1}: {best_loss:.4f} <= {threshold_val}")
                break

        final_path = os.path.join(log_dir, f"{name}.json")
        ts.save_style(final_path, best_ttl, best_dp, wav_val)
        if os.path.exists(state_path):
            os.remove(state_path)
        elapsed = time.time() - on_step.t0
        final_msg = f"Training complete!\nBest loss: {best_loss:.4f}\nTime: {elapsed/60:.1f}min\nStyle JSON: {final_path}"
        print(final_msg)
        progress(1.0, desc="Done!")
        yield final_msg, final_path

    except Exception as e:
        import traceback
        yield f"Training failed:\n{traceback.format_exc()}", None


def synthesize_speech(text, style_path, lang, speed, steps, progress=gr.Progress()):
    if not text:
        return None, "Enter text to synthesize."
    if not style_path or not os.path.exists(style_path):
        return None, "Train a voice first or provide a style JSON path."
    if not check_onnx_models():
        return None, "Models not downloaded yet."

    progress(0, desc="Loading TTS...")
    try:
        from pipeline.generate import generate
        progress(0.3, desc="Synthesizing...")
        out_path = generate(text, style_path, lang, int(steps), float(speed), str(ONNX_DIR))
        progress(1.0, desc="Done!")
        return out_path, f"Generated: {out_path}"
    except Exception as e:
        import traceback
        return None, f"Synthesis failed:\n{traceback.format_exc()}"


def list_trained_styles():
    styles = list(LOGS_DIR.rglob("*.json"))
    styles = [str(s) for s in styles if not any(x in s.name for x in ["_0005", "_0010", "_0015", "_0020"])]
    if not styles:
        return "No trained styles yet. Train a voice first."
    return "\n".join(styles)


def build_app():
    with gr.Blocks(title="Supertonic Embeddings Trainer") as app:
        gr.Markdown("# [Supertonic Embeddings Trainer](https://github.com/Saganaki22/supertonic_embeddings_trainer)")
        gr.Markdown("Upload a WAV → train a voice style → synthesize speech in that voice")

        with gr.Tab("Setup"):
            gr.Markdown("### Step 0: Download Models (~400MB)")
            dl_btn = gr.Button("Download Models", variant="primary")
            dl_status = gr.Textbox(label="Status", interactive=False)
            dl_btn.click(download_models, outputs=[dl_status])

        with gr.Tab("Clone Voice"):
            gr.Markdown("### Step 1: Upload voice sample and train")
            gr.Markdown("Provide 3-30 seconds of clean speech. More = better quality.\n\n**Resume**: Re-train with the same voice name to continue from where you left off.")
            with gr.Row():
                wav_input = gr.Audio(label="Voice Sample (WAV)", type="filepath")
                with gr.Column():
                    voice_name = gr.Textbox(label="Voice Name", value="my_voice")
                    gender = gr.Radio(["F", "M"], label="Gender", value="F")
                    ref_mode = gr.Radio(
                        ["auto", "none"],
                        label="Reference Style Init",
                        value="auto",
                        info="auto=find closest built-in voice, none=random"
                    )
                    resume_status = gr.Textbox(label="Resume Status", interactive=False, value="Enter a name and check")
                    check_btn = gr.Button("Check for Checkpoint", size="sm")
                    check_btn.click(check_resume, inputs=[voice_name], outputs=[resume_status])
            with gr.Row():
                num_steps = gr.Slider(500, 10000, value=3000, step=500, label="Training Steps")
                save_every = gr.Slider(50, 2000, value=250, step=50, label="Save Every N Steps")
                lr = gr.Slider(0.00005, 0.001, value=0.0002, step=0.00005, label="Learning Rate")
                threshold = gr.Slider(0.20, 0.40, value=0.24, step=0.01, label="Early Stop Threshold")
            with gr.Row():
                train_btn = gr.Button("Train Voice Style", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop")
            train_status = gr.Textbox(label="Status", lines=8, interactive=False)
            trained_style_path = gr.Textbox(label="Trained Style JSON Path", visible=True)

            train_btn.click(
                train_voice,
                inputs=[wav_input, voice_name, gender, ref_mode, num_steps, save_every, lr, threshold],
                outputs=[train_status, trained_style_path],
            )
            stop_btn.click(stop_training, outputs=[train_status])

        with gr.Tab("Synthesize"):
            gr.Markdown("### Step 2: Generate speech with the cloned voice")
            synth_style = gr.Textbox(label="Style JSON Path", placeholder="e.g. pipeline/logs/my_voice/my_voice.json")
            with gr.Row():
                synth_text = gr.Textbox(label="Text", lines=3, placeholder="Enter text to speak...")
                synth_lang = gr.Dropdown(
                    ["en", "ko", "ja", "de", "fr", "es", "it", "pt", "ru", "zh"],
                    value="en", label="Language"
                )
            with gr.Row():
                synth_speed = gr.Slider(0.8, 1.5, value=1.05, step=0.05, label="Speed")
                synth_steps = gr.Slider(4, 16, value=5, step=2, label="Vocoder Steps (more=better)")
            synth_btn = gr.Button("Synthesize", variant="primary")
            synth_audio = gr.Audio(label="Generated Audio", type="filepath")
            synth_status = gr.Textbox(label="Status", interactive=False)

            synth_btn.click(
                synthesize_speech,
                inputs=[synth_text, synth_style, synth_lang, synth_speed, synth_steps],
                outputs=[synth_audio, synth_status],
            )

        with gr.Tab("Browse Styles"):
            gr.Markdown("### Trained voice styles")
            browse_btn = gr.Button("Refresh")
            styles_list = gr.Textbox(label="Available Styles", lines=10, interactive=False)
            browse_btn.click(list_trained_styles, outputs=[styles_list])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
