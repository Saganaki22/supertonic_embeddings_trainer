# Supertonic Embeddings Trainer

Local voice cloning pipeline for [SupertonicTTS](https://github.com/kdrkdrkdr/supertonic.embed). Upload a WAV sample -> train a style embedding -> synthesize speech in that voice. No cloud API needed.

Supports both **v2** (Supertonic 2, 66M params, 5 languages) and **v3** (Supertonic 3, 99M params, 31 languages).

Based on gradient-based inverse optimization of frozen TTS models, matching WavLM-Large Layer 3 features (Chiu et al. 2025).

## Features

- **Dual model support** -- Train with Supertonic v2 or v3 (separate style spaces, separate model files)
- **Two loss modes** -- WavLM Layer 3 MSE (original) or ECAPA-guided early stop (speaker cosine similarity)
- **Stop & Resume** -- Stop training mid-run and continue later from exactly where you left off
- **Auto voice matching** -- Finds the closest reference preset via WavLM Layer 3 distance
- **Audio preprocessing** -- VAD silence trimming + peak normalization before training
- **Live progress** -- Loss, best loss, learning rate, ECAPA similarity, ETA updated in real time
- **Checkpoint saving** -- Style JSONs saved every N steps so you can pick the best one

## Requirements

- Python 3.10+
- CUDA GPU (12.8+)
- ~4GB VRAM

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Use the **Setup** tab in the Gradio UI to download models (Supertone/supertonic-2 and/or Supertone/supertonic-3). A `training_state.pt` checkpoint is saved so you can stop and resume training at any time.

## Usage

```bash
# Launch Gradio UI
start.bat
# or: python app.py
```

### Gradio Tabs

| Tab | Description |
|-----|-------------|
| **Setup** | Download v2 and/or v3 ONNX models. Separate downloads for each version. |
| **Clone Voice** | Upload WAV -> pick v2/v3, loss mode, steps/threshold -> train. Stop anytime, resume by re-training with the same name |
| **Synthesize** | Pick v2/v3, enter text + style JSON -> generate speech. Language list updates based on version. |
| **Browse Styles** | Preview trained voice styles |

### v2 vs v3

**v2 and v3 JSONs are NOT cross-compatible.** They use identical JSON schema but completely different style spaces. A v2-trained style JSON fed into v3 will produce unintelligible output, and vice versa. Always train and synthesize with the same version.

| | v2 | v3 |
|---|---|---|
| Parameters | 66M | 99M |
| Languages | 5 (en, ko, es, pt, fr) | 31 |
| Vocoder steps default | 5 | 6 |
| Recommended loss | WavLM (recommended) | WavLM (recommended) |
| Recommended threshold | 0.24 | 0.24 |

### Training Methods

| Method | Loss function | Early stop criterion | Status |
|---|---|---|---|
| **WavLM** | WavLM-Large Layer 3 mean+std MSE | WavLM loss <= threshold | **Recommended** for both v2 and v3 |
| **ECAPA-guided** | WavLM backprop + ECAPA cosine similarity monitor | ECAPA 1-cosine <= threshold (WavLM 0.24 fallback) | **Experimental** -- needs more steps to reach desired speaker cosine similarity |

WavLM always drives gradient backprop. ECAPA-guided mode is experimental: it monitors ECAPA-TDNN speaker similarity as an additional stopping criterion, but typically needs significantly more training steps to reach the target cosine similarity threshold.

### Stop & Resume

- Training state is **always saved** -- on stop, on checkpoint, and on early stop/completion
- Click **Stop Training** at any point -- state is preserved and you can resume later
- Re-click **Train Voice Style** with the same voice name and version -- training resumes from the saved step
- Even after training completes (early stop or max steps), re-training with the same name continues optimizing
- Check **Check for Checkpoint** to see if a resume point exists for a name

### CLI

```bash
python pipeline/train_style.py --wav voice.wav --name myvoice --vocoder_steps 5 --num_steps 3000 --onnx_dir pipeline/onnx_v2 --styles_dir reference_styles_v2
```

## How It Works

1. **Auto-select** closest reference style via WavLM Layer 3 distance across all 10 presets (version-matched)
2. **Initialize** `style_ttl` from that reference (trainable), freeze `style_dp`
3. **Optimize** `style_ttl` via Adam, minimizing WavLM Layer 3 feature statistics (mean + std MSE) between generated and target audio
4. **Early stop** when loss drops below threshold (WavLM: 0.24, ECAPA: 0.15)
5. In **ECAPA-guided** mode, backprop still uses WavLM but early stop uses ECAPA speaker similarity

## Defaults

| Parameter | v2 Default | v3 Default | Notes |
|---|---|---|---|
| Training Steps | 3000 | 3000 | Early stop usually fires ~500 steps |
| Early Stop Threshold | 0.24 (WavLM) / 0.15 (ECAPA) | 0.24 (WavLM) / 0.15 (ECAPA) | Auto-switches with loss mode |
| Vocoder Steps | 5 | 6 | Denoising iterations |
| Learning Rate | 2e-4 | 2e-4 | Adam with ReduceLROnPlateau (patience=200) |
| Speed | 1.05 | 1.05 | Duration scaling |

## Project Structure

```
app.py                  # Gradio UI (dual-mode, stop/resume, audio preprocessing)
config.py               # MODEL_CONFIGS for v2/v3
start.bat               # Launch script
requirements.txt
reference_styles_v2/    # F1-F5.json, M1-M5.json (v2 presets)
reference_styles_v3/    # F1-F5.json, M1-M5.json (v3 presets)
pipeline/
  onnx_v2/              # Supertonic-2 ONNX models (downloaded via UI)
  onnx_v3/              # Supertonic-3 ONNX models (downloaded via UI)
  train_style.py        # Standalone training script (accepts onnx_dir/styles_dir)
  generate.py           # Synthesis from text + style
  helper.py             # TTS engine, text processing (31 languages)
  utils/
    loss.py             # WavLM-Large Layer 3 loss + ECAPA-TDNN speaker loss
    model.py            # ONNX->PyTorch conversion (onnxslim + opset17 + _fix_clip)
    style.py            # Style JSON I/O
    dataloader.py       # Training dataloader
  configs/
    default.py          # Training defaults
    utterances.py       # Multi-text rotation for training
```

## Citation

If you use this work, please cite:

```bibtex
@misc{kim2026supertonicembed,
  author       = {Gyeongmin Kim},
  title        = {Extracting Voice Styles from Frozen TTS Models via Gradient-Based Inverse Optimization},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19646514},
  url          = {https://doi.org/10.5281/zenodo.19646514}
}
```

Preprint available on Zenodo: https://doi.org/10.5281/zenodo.19646514
