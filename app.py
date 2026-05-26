"""
Supertonic Embeddings Trainer — Dual Mode (v2 / v3)
Upload WAV -> Train Style JSON -> Synthesize Speech
Supports: v2/v3 models, WavLM/ECAPA loss, stop/resume
"""

import gradio as gr
import os
import sys
import shutil
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
sys.path.insert(0, str(PIPELINE_DIR))

from config import MODEL_CONFIGS
from pipeline.helper import load_text_to_speech, load_voice_style, timer
from pipeline.configs.utterances import texts

VOICES_DIR = PIPELINE_DIR / "pipeline" / "voices"
SAMPLES_DIR = PIPELINE_DIR / "pipeline" / "samples"
LOGS_DIR = PIPELINE_DIR / "pipeline" / "logs"

for d in [VOICES_DIR, SAMPLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

_stop_flag = {"stop": False}


def _parse_version(version_str):
    return "v2" if "v2" in version_str else "v3"


def _parse_loss_mode(loss_str):
    return "ecapa" if "ECAPA" in loss_str else "wavlm"


def check_onnx_models(version="v2"):
    onnx_dir = MODEL_CONFIGS[version]["onnx_dir"]
    if not onnx_dir.exists():
        return False
    required = ["tts.json", "duration_predictor.onnx", "text_encoder.onnx",
                "vector_estimator.onnx", "vocoder.onnx", "unicode_indexer.json"]
    return all((onnx_dir / f).exists() for f in required)


def download_models(version_str, progress=gr.Progress()):
    version = _parse_version(version_str)
    cfg = MODEL_CONFIGS[version]
    onnx_dir = cfg["onnx_dir"]
    styles_dir = cfg["styles_dir"]

    if check_onnx_models(version):
        n = len(list(styles_dir.glob("*.json"))) if styles_dir.exists() else 0
        if n >= 10:
            return f"Already downloaded: {onnx_dir.name} ({n} voice styles)"

    progress(0, desc=f"Downloading {cfg['hf_repo']}...")
    try:
        from huggingface_hub import snapshot_download
        local = str(PIPELINE_DIR / "pipeline" / "hf_cache" / cfg["hf_repo"].replace("/", "_"))
        snapshot_download(cfg["hf_repo"], local_dir=local)

        onnx_dir.mkdir(parents=True, exist_ok=True)
        styles_dir.mkdir(parents=True, exist_ok=True)

        required = ["tts.json", "duration_predictor.onnx", "text_encoder.onnx",
                    "vector_estimator.onnx", "vocoder.onnx", "unicode_indexer.json"]
        for f in required:
            for search_path in [
                Path(local) / "onnx" / f,
                Path(local) / f,
            ]:
                if search_path.exists():
                    shutil.copy2(str(search_path), str(onnx_dir / f))
                    break

        src_styles = Path(local) / "voice_styles"
        if src_styles.exists():
            for sf in src_styles.glob("*.json"):
                shutil.copy2(str(sf), str(styles_dir / sf.name))

        n = len(list(styles_dir.glob("*.json")))
        progress(1.0, desc="Done!")
        return f"Downloaded {cfg['hf_repo']} -> {onnx_dir.name} ({n} voice styles)"
    except Exception as e:
        import traceback
        return f"Download failed:\n{traceback.format_exc()}"


def stop_training():
    _stop_flag["stop"] = True
    return "Stopping... (will save checkpoint)"


def check_resume(name, version_str):
    version = _parse_version(version_str)
    name_versioned = f"{name.strip().replace(' ', '_')}_{version}"
    log_dir = LOGS_DIR / name_versioned
    state_path = log_dir / "training_state.pt"
    if state_path.exists():
        import torch
        state = torch.load(str(state_path), weights_only=False, map_location="cpu")
        step = state.get("step", 0)
        best = state.get("best_loss", float("inf"))
        saved_version = state.get("version", version)
        saved_loss_mode = state.get("loss_mode", "wavlm")
        ecapa = state.get("best_ecapa", None)
        extra = f" | ECAPA sim: {1-ecapa:.4f}" if ecapa is not None else ""
        return (f"Found checkpoint at step {step} (best WavLM loss: {best:.4f}{extra}).\n"
                f"Saved version: {saved_version} | Loss mode: {saved_loss_mode}\n"
                f"Training will resume from here with saved settings.")
    return "No checkpoint found. Starting fresh."


def _save_state(state_path, step, best_loss, best_ecapa_val, best_ttl, best_dp,
                style_ttl, style_dp, noisy_fixed, lmask, optimizer, scheduler,
                version, loss_mode):
    import torch
    torch.save({
        "step": step, "best_loss": best_loss, "best_ecapa": best_ecapa_val,
        "best_ttl": best_ttl.cpu(), "best_dp": best_dp.cpu(),
        "style_ttl": style_ttl.detach().cpu(), "style_dp": style_dp.cpu(),
        "noisy_fixed": noisy_fixed.cpu(), "lmask": lmask.cpu(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "version": version, "loss_mode": loss_mode,
    }, state_path)


def train_voice(
    wav_path, name, gender,
    model_version_str, loss_mode_str, ref_mode,
    num_steps, save_every, lr, threshold,
    progress=gr.Progress(),
):
    if not wav_path:
        yield "Upload a WAV file first.", None
        return

    version = _parse_version(model_version_str)
    loss_mode = _parse_loss_mode(loss_mode_str)

    if not check_onnx_models(version):
        yield f"Models not downloaded for {version}. Go to Setup tab and download them first.", None
        return

    cfg = MODEL_CONFIGS[version]
    onnx_dir = str(cfg["onnx_dir"])
    styles_dir = str(cfg["styles_dir"])
    vocoder_steps = cfg["default_vocoder_steps"]

    name_versioned = f"{name.strip().replace(' ', '_') or 'custom_voice'}_{version}"
    dst = str(VOICES_DIR / f"{name_versioned}.wav")

    import torch
    import numpy as np
    import librosa

    _stop_flag["stop"] = False

    # Copy WAV immediately to avoid Gradio temp file PermissionError
    if not os.path.exists(dst):
        tmp_copy = dst + ".tmp"
        shutil.copy2(wav_path, tmp_copy)
        y, sr = librosa.load(tmp_copy, sr=44100, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        peak = np.abs(y).max()
        if peak > 0:
            y = y / peak * 0.95
        import soundfile as sf
        sf.write(dst, y, 44100)
        if os.path.exists(tmp_copy):
            os.remove(tmp_copy)

    progress(0.0, desc="Loading models...")

    import pipeline.train_style as ts
    from pipeline.utils.loss import (
        load_wavlm, extract_wavlm_targets, wavlm_loss,
        load_ecapa, extract_ecapa_targets, ecapa_loss,
    )

    DEVICE = ts.DEVICE

    tts = ts.load_text_to_speech(onnx_dir)
    dataloader = ts.get_train_dataloader(tts, ts.texts)
    del tts
    data_iter = iter(dataloader)

    log_dir = str(LOGS_DIR / name_versioned)
    os.makedirs(log_dir, exist_ok=True)
    state_path = os.path.join(log_dir, "training_state.pt")

    save_every_int = int(save_every)
    total = int(num_steps)
    status_lines = []

    resumed = False
    if os.path.exists(state_path):
        print(f"Resuming from checkpoint: {state_path}")
        state = torch.load(state_path, weights_only=False, map_location=DEVICE)

        # Auto-restore version and loss mode from checkpoint
        saved_version = state.get("version", version)
        saved_loss_mode = state.get("loss_mode", loss_mode)
        if saved_version != version:
            print(f"  WARNING: checkpoint was trained with {saved_version}, you selected {version}. Using saved: {saved_version}")
            version = saved_version
            cfg = MODEL_CONFIGS[version]
            onnx_dir = str(cfg["onnx_dir"])
            styles_dir = str(cfg["styles_dir"])
            vocoder_steps = cfg["default_vocoder_steps"]
        if saved_loss_mode != loss_mode:
            print(f"  WARNING: checkpoint used {saved_loss_mode}, you selected {loss_mode}. Using saved: {saved_loss_mode}")
            loss_mode = saved_loss_mode

        start_step = state["step"]
        best_loss = state["best_loss"]
        best_ttl = state["best_ttl"].to(DEVICE)
        best_dp = state["best_dp"].to(DEVICE)
        style_ttl = state["style_ttl"].to(DEVICE).requires_grad_(True)
        style_dp = state["style_dp"].to(DEVICE)
        noisy_fixed = state["noisy_fixed"].to(DEVICE)
        lmask = state["lmask"].to(DEVICE)
        best_ecapa_val = state.get("best_ecapa", None)
        resumed = True

        torch.manual_seed(42)
        np.random.seed(42)

        model = ts.SupertonicModel(onnx_dir)

        optimizer = torch.optim.Adam([style_ttl], lr=float(lr))
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=200, factor=0.5, min_lr=float(lr) * 0.01
        )
        scheduler.load_state_dict(state["scheduler_state"])

        for _ in range(start_step % len(ts.texts)):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                next(data_iter)

        status_lines.append(f"Resumed from step {start_step} (best: {best_loss:.4f}) [version={version}, loss={loss_mode}]")
        print(f"  Resumed: step {start_step}, best_loss={best_loss:.4f}, version={version}, loss_mode={loss_mode}")
    else:
        torch.manual_seed(42)
        np.random.seed(42)

        best_ecapa_val = None

        print(f"Name: {name_versioned}  Gender: {gender}  Device: {DEVICE}")
        print(f"Target WAV: {dst}")

        tmp_ids, tmp_mask = next(data_iter)

        model = ts.SupertonicModel(onnx_dir)

        if ref_mode == "auto":
            print("\nFinding closest reference style (WavLM Layer 3)...")
            dummy_dp = ts.load_single_voice_style(os.path.join(styles_dir, "M1.json"))[1]
            with torch.no_grad():
                init_dur_tmp = model.dp_model(tmp_ids, dummy_dp, tmp_mask) / 1.05
                init_dur_tmp = init_dur_tmp.detach().cpu().numpy()
            tts_tmp = ts.load_text_to_speech(onnx_dir)
            noisy_tmp, lmask_tmp = tts_tmp.sample_noisy_latent(duration=init_dur_tmp)
            noisy_tmp = torch.tensor(noisy_tmp, dtype=torch.float32).to(DEVICE)
            lmask_tmp = torch.tensor(lmask_tmp, dtype=torch.float32).to(DEVICE)
            del tts_tmp

            style_ttl, style_dp = ts.find_closest_style(
                styles_dir, model, dst,
                tmp_ids, tmp_mask, noisy_tmp, lmask_tmp, vocoder_steps, 1.05
            )
            del noisy_tmp, lmask_tmp
            if style_ttl is None:
                _, style_dp = ts.load_single_voice_style(os.path.join(styles_dir, "M1.json"))
                style_ttl = torch.randn(1, 50, 256, device=DEVICE) * 0.1
        elif ref_mode and os.path.exists(ref_mode):
            style_ttl, style_dp = ts.load_single_voice_style(ref_mode)
        else:
            _, style_dp = ts.load_single_voice_style(os.path.join(styles_dir, "M1.json"))
            style_ttl = torch.randn(1, 50, 256, device=DEVICE) * 0.1

        tts = ts.load_text_to_speech(onnx_dir)
        with torch.no_grad():
            init_dur = model.dp_model(tmp_ids, style_dp, tmp_mask) / 1.05
            init_dur = init_dur.detach().cpu().numpy()
        noisy_fixed, lmask = tts.sample_noisy_latent(duration=init_dur)
        noisy_fixed = torch.tensor(noisy_fixed, dtype=torch.float32).to(DEVICE)
        lmask = torch.tensor(lmask, dtype=torch.float32).to(DEVICE)
        del tts

        style_ttl = style_ttl.clone().requires_grad_(True)
        style_dp = style_dp.detach().clone()

        optimizer = torch.optim.Adam([style_ttl], lr=float(lr))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=200, factor=0.5, min_lr=float(lr) * 0.01
        )

        best_loss = float("inf")
        best_ttl = None
        best_dp = style_dp.detach().clone()
        start_step = 0

    # ── loss function setup ────────────────────────────────────────────────
    progress(0.05, desc="Loading WavLM-Large...")
    wavlm_model = load_wavlm(DEVICE)
    import soundfile as sf
    target_np, _ = sf.read(dst, dtype="float32")
    target_wav_t = torch.tensor(target_np, dtype=torch.float32).to(DEVICE)
    target_feats = extract_wavlm_targets(wavlm_model, target_wav_t, DEVICE)

    ecapa_model = None
    target_emb = None
    if loss_mode == "ecapa":
        progress(0.08, desc="Loading ECAPA-TDNN...")
        ecapa_model = load_ecapa(DEVICE)
        target_emb = extract_ecapa_targets(ecapa_model, target_wav_t, DEVICE)

    def compute_loss(gen_wav):
        return wavlm_loss(wavlm_model, gen_wav, target_feats, DEVICE)

    # ── training loop ──────────────────────────────────────────────────────
    t0 = time.time()
    thr = float(threshold)
    wavlm_thr = 0.24

    print(f"\nTraining [{version}] loss=[{loss_mode}] steps={start_step}->{total} threshold={thr}")

    for step in range(start_step, total):
        if _stop_flag["stop"]:
            print(f"  Stopped at step {step}. Saving state...")
            _save_state(state_path, step, best_loss, best_ecapa_val, best_ttl, best_dp,
                        style_ttl, style_dp, noisy_fixed, lmask, optimizer, scheduler,
                        version, loss_mode)
            stop_style = os.path.join(log_dir, f"{name_versioned}.json")
            ts.save_style(stop_style, best_ttl, best_dp, dst)
            stop_msg = f"Training stopped at step {step}.\nBest loss: {best_loss:.4f}\nStyle JSON: {stop_style}\nState: {state_path}\nRe-train with same name to resume."
            print(stop_msg)
            yield stop_msg, None
            return

        optimizer.zero_grad()

        try:
            text_ids, text_mask = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            text_ids, text_mask = next(data_iter)

        text_ids = text_ids.to(DEVICE)
        text_mask = text_mask.to(DEVICE)

        wav_out = model(text_ids, text_mask, style_ttl, vocoder_steps, noisy_fixed, lmask)
        gen_wav = wav_out.squeeze()

        loss = compute_loss(gen_wav)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([style_ttl], max_norm=1.0)
        optimizer.step()

        step_loss = loss.detach().item()
        if step_loss < best_loss:
            best_loss = step_loss
            best_ttl = style_ttl.detach().clone()

        scheduler.step(best_loss)

        # ECAPA monitor
        ecapa_val = None
        if ecapa_model is not None:
            with torch.no_grad():
                ecapa_val = ecapa_loss(ecapa_model, gen_wav.detach(), target_emb, DEVICE).item()
            if best_ecapa_val is None or ecapa_val < best_ecapa_val:
                best_ecapa_val = ecapa_val

        if (step + 1) % 8 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            eta = (elapsed / (step + 1 - start_step)) * (total - step - 1) if (step + 1 - start_step) > 0 else 0
            status_lines.clear()
            status_lines.append(f"Step {step+1}/{total} ({(step+1)/total*100:.1f}%)")
            status_lines.append(f"WavLM loss: {step_loss:.4f} | Best: {best_loss:.4f} | LR: {cur_lr:.6f}")
            if ecapa_val is not None:
                status_lines.append(f"ECAPA speaker sim: {1-ecapa_val:.4f} (loss: {ecapa_val:.4f}) | Best sim: {1-best_ecapa_val:.4f}" if best_ecapa_val is not None else "")
            status_lines.append(f"Elapsed: {elapsed/60:.1f}min | ETA: {eta/60:.1f}min")
            print(f"  Step {step+1}/{total} | Loss: {step_loss:.4f} | Best: {best_loss:.4f} | LR: {cur_lr:.6f} | ETA: {eta/60:.1f}min")
            progress((step + 1) / total,
                     desc=f"[{version}/{loss_mode}] step {step+1}/{total} loss={step_loss:.4f}")
            yield "\n".join(status_lines), None

        if (step + 1) % save_every_int == 0:
            ckpt = os.path.join(log_dir, f"{name_versioned}_{step+1:04d}.json")
            ts.save_style(ckpt, best_ttl, best_dp, dst)
            _save_state(state_path, step + 1, best_loss, best_ecapa_val, best_ttl, best_dp,
                        style_ttl, style_dp, noisy_fixed, lmask, optimizer, scheduler,
                        version, loss_mode)
            print(f"  >> Checkpoint: {ckpt} | State: {state_path}")
            status_lines.append(f"Saved checkpoint: {ckpt}")
            yield "\n".join(status_lines), None

        # Early stop:
        # - WavLM mode: stop when WavLM loss <= threshold
        # - ECAPA mode: stop when ECAPA loss <= threshold OR WavLM loss <= 0.24 (fallback)
        if loss_mode == "ecapa" and ecapa_val is not None:
            if ecapa_val <= thr:
                print(f"  Early stop at step {step+1}: ECAPA {ecapa_val:.4f} <= {thr}")
                print(f"  Style JSON: {os.path.join(log_dir, f'{name_versioned}.json')}")
                break
            if best_loss <= wavlm_thr:
                print(f"  Early stop at step {step+1}: WavLM {best_loss:.4f} <= {wavlm_thr} (fallback)")
                print(f"  Style JSON: {os.path.join(log_dir, f'{name_versioned}.json')}")
                break
        else:
            if best_loss <= thr:
                print(f"  Early stop at step {step+1}: WavLM {best_loss:.4f} <= {thr}")
                print(f"  Style JSON: {os.path.join(log_dir, f'{name_versioned}.json')}")
                break

    final_path = os.path.join(log_dir, f"{name_versioned}.json")
    ts.save_style(final_path, best_ttl, best_dp, dst)
    _save_state(state_path, step + 1 if 'step' in dir() else total,
                best_loss, best_ecapa_val, best_ttl, best_dp,
                style_ttl, style_dp, noisy_fixed, lmask, optimizer, scheduler,
                version, loss_mode)
    elapsed = time.time() - t0
    ecapa_info = f"\nBest ECAPA sim: {1-best_ecapa_val:.4f}" if best_ecapa_val is not None else ""
    final_msg = f"Training complete!\nBest WavLM loss: {best_loss:.4f}{ecapa_info}\nTime: {elapsed/60:.1f}min\nStyle JSON: {final_path}\nState: {state_path}\nVersion: {version} | Loss mode: {loss_mode}\n\nRe-train with same name to continue optimizing."
    print(f"\nDone! Best loss: {best_loss:.4f} | Time: {elapsed/60:.1f}min")
    print(f"  Style JSON: {final_path}")
    print(f"  State PT:   {state_path}")
    print(f"  Version: {version} | Loss mode: {loss_mode}")
    progress(1.0, desc="Done!")
    yield final_msg, final_path


def synthesize_speech(text, style_path, lang, speed, steps, version_str, progress=gr.Progress()):
    if not text:
        return None, "Enter text to synthesize."
    if not style_path or not os.path.exists(style_path):
        return None, "Train a voice first or provide a style JSON path."

    version = _parse_version(version_str)
    if not check_onnx_models(version):
        return None, f"Models not downloaded for {version}."

    progress(0, desc="Loading TTS...")
    try:
        from pipeline.generate import generate
        onnx_dir = str(MODEL_CONFIGS[version]["onnx_dir"])
        progress(0.3, desc="Synthesizing...")
        out_path = generate(text, style_path, lang, int(steps), float(speed), onnx_dir)
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
    return "\n".join(sorted(styles))


def update_threshold_default(loss_str):
    if "ECAPA" in loss_str:
        return gr.Slider(value=0.15,
                         info="ECAPA mode: 0.15 = speaker cosine similarity >= 0.85")
    return gr.Slider(value=0.24,
                     info="WavLM mode: 0.24 = same-speaker floor (calibrated for v2)")


def update_lang_choices(version_str):
    version = _parse_version(version_str)
    langs = MODEL_CONFIGS[version]["languages"]
    return gr.Dropdown(choices=langs, value="en")


def build_app():
    with gr.Blocks(title="Supertonic Embeddings Trainer") as app:
        gr.Markdown("# [Supertonic Embeddings Trainer](https://github.com/Saganaki22/supertonic_embeddings_trainer)")
        gr.Markdown("Upload a WAV -> train a voice style -> synthesize speech in that voice. Supports **v2** (5 langs) and **v3** (31 langs).")

        # ── Setup Tab ──────────────────────────────────────────────────────
        with gr.Tab("Setup"):
            gr.Markdown("### Download Model Files")
            gr.Markdown(
                "**v2 and v3 use DIFFERENT style spaces. JSONs are NOT cross-compatible.** "
                "Train and synthesize with the same version."
            )

            model_version_dl = gr.Radio(
                choices=[
                    "v2 -- Supertonic 2 (66M params, 5 languages: en/ko/es/pt/fr)",
                    "v3 -- Supertonic 3 (99M params, 31 languages)",
                ],
                value="v2 -- Supertonic 2 (66M params, 5 languages: en/ko/es/pt/fr)",
                label="Model Version to Download",
            )

            dl_btn = gr.Button("Download Selected Model Files", variant="primary")
            dl_status = gr.Textbox(label="Download Status", interactive=False)
            dl_btn.click(download_models, inputs=[model_version_dl], outputs=[dl_status])

        # ── Clone Voice Tab ────────────────────────────────────────────────
        with gr.Tab("Clone Voice"):
            gr.Markdown("### Step 1: Upload voice sample and train")
            gr.Markdown(
                "Provide 3-30 seconds of clean speech. More = better quality.\n\n"
                "**Resume**: Re-train with the same voice name and version to continue from where you left off.\n"
                "**Auto-restore**: If a checkpoint exists, the app will use the saved version and loss mode automatically."
            )

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

            with gr.Row():
                model_version_train = gr.Radio(
                    choices=[
                        "v2 -- Supertonic 2 (5 languages)",
                        "v3 -- Supertonic 3 (31 languages)",
                    ],
                    value="v2 -- Supertonic 2 (5 languages)",
                    label="Model Version",
                    info="Must match the model files you downloaded. JSONs are version-specific.",
                )

                loss_mode = gr.Radio(
                    choices=[
                        "WavLM (standard -- WavLM-Large Layer 3 MSE)",
                        "ECAPA-guided stop (WavLM backprop + ECAPA speaker similarity early stop)",
                    ],
                    value="WavLM (standard -- WavLM-Large Layer 3 MSE)",
                    label="Training Method",
                    info=(
                        "WavLM: original method, good for both v2/v3. "
                        "ECAPA-guided: uses ECAPA cosine similarity as the early-stop signal. "
                        "Better speaker identity stopping, especially for v3."
                    ),
                )

            with gr.Row():
                num_steps = gr.Slider(500, 10000, value=3000, step=500, label="Training Steps",
                                      info="Default 3000. Early stop fires before this.")
                save_every = gr.Slider(50, 2000, value=250, step=50, label="Save Every N Steps")

            with gr.Row():
                lr = gr.Slider(0.00005, 0.001, value=0.0002, step=0.00005, label="Learning Rate")
                threshold = gr.Slider(0.05, 0.40, value=0.24, step=0.01,
                                      label="Early Stop Threshold",
                                      info="WavLM: 0.24 (v2 calibrated). ECAPA: 0.15 (cosine sim >= 0.85).")

            loss_mode.change(update_threshold_default, inputs=[loss_mode], outputs=[threshold])

            with gr.Row():
                resume_status = gr.Textbox(label="Resume Status", interactive=False, value="Enter a name and check")
                check_btn = gr.Button("Check for Checkpoint", size="sm")

            check_btn.click(
                check_resume,
                inputs=[voice_name, model_version_train],
                outputs=[resume_status],
            )

            with gr.Row():
                train_btn = gr.Button("Train Voice Style", variant="primary")
                stop_btn = gr.Button("Stop Training", variant="stop")
            train_status = gr.Textbox(label="Status", lines=10, interactive=False)
            trained_style_path = gr.Textbox(label="Trained Style JSON Path", visible=True)

            train_btn.click(
                train_voice,
                inputs=[
                    wav_input, voice_name, gender,
                    model_version_train, loss_mode, ref_mode,
                    num_steps, save_every, lr, threshold,
                ],
                outputs=[train_status, trained_style_path],
            )
            stop_btn.click(stop_training, outputs=[train_status])

        # ── Synthesize Tab ─────────────────────────────────────────────────
        with gr.Tab("Synthesize"):
            gr.Markdown("### Step 2: Generate speech with the cloned voice")

            synth_version = gr.Radio(
                choices=["v2 -- Supertonic 2", "v3 -- Supertonic 3"],
                value="v2 -- Supertonic 2",
                label="Model Version for Synthesis",
                info="Must match the version used to train the style JSON.",
            )

            synth_style = gr.Textbox(label="Style JSON Path", placeholder="e.g. pipeline/logs/my_voice_v2/my_voice_v2.json")
            with gr.Row():
                synth_text = gr.Textbox(label="Text", lines=3, placeholder="Enter text to speak...")
                synth_lang = gr.Dropdown(
                    ["en", "ko", "es", "pt", "fr"],
                    value="en", label="Language"
                )
            with gr.Row():
                synth_speed = gr.Slider(0.8, 1.5, value=1.05, step=0.05, label="Speed")
                synth_steps = gr.Slider(4, 16, value=5, step=2, label="Vocoder Steps (more=better)")
            synth_btn = gr.Button("Synthesize", variant="primary")
            synth_audio = gr.Audio(label="Generated Audio", type="filepath")
            synth_status = gr.Textbox(label="Status", interactive=False)

            synth_version.change(
                update_lang_choices,
                inputs=[synth_version],
                outputs=[synth_lang],
            )

            synth_btn.click(
                synthesize_speech,
                inputs=[synth_text, synth_style, synth_lang, synth_speed, synth_steps, synth_version],
                outputs=[synth_audio, synth_status],
            )

        # ── Browse Styles Tab ──────────────────────────────────────────────
        with gr.Tab("Browse Styles"):
            gr.Markdown("### Trained voice styles")
            browse_btn = gr.Button("Refresh")
            styles_list = gr.Textbox(label="Available Styles", lines=10, interactive=False)
            browse_btn.click(list_trained_styles, outputs=[styles_list])

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
