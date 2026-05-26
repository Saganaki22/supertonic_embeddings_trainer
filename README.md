# Supertonic Embeddings Trainer

Local voice cloning pipeline for [SupertonicTTS](https://github.com/kdrkdrkdr/supertonic.embed). Upload a WAV sample → train a style embedding → synthesize speech in that voice. No cloud API needed.

Based on gradient-based inverse optimization of frozen TTS models, matching WavLM-Large Layer 3 features (Chiu et al. 2025).

## Features

- **Stop & Resume** — Stop training mid-run and continue later from exactly where you left off (optimizer state, scheduler, latent geometry all preserved)
- **Auto voice matching** — Finds the closest reference preset via WavLM Layer 3 distance
- **Audio preprocessing** — VAD silence trimming + peak normalization before training
- **Live progress** — Loss, best loss, learning rate, ETA updated in real time
- **Checkpoint saving** — Style JSONs saved every N steps so you can pick the best one

## Requirements

- Python 3.10+
- CUDA GPU (12.8+)
- ~8GB VRAM

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Or use the Setup tab in the Gradio UI to download models automatically (Supertone/supertonic-2).

## Usage

```bash
# Launch Gradio UI
start.bat
# or: python app.py
```

### Gradio Tabs

| Tab | Description |
|-----|-------------|
| **Setup** | Download Supertonic-2 ONNX models (~400MB) |
| **Clone Voice** | Upload WAV → set name/steps/threshold → train. Stop anytime, resume by re-training with the same name |
| **Synthesize** | Enter text + select trained style → generate speech |
| **Browse Styles** | Preview reference and trained voice styles |

### Stop & Resume

1. Click **Stop Training** at any point — a `training_state.pt` checkpoint is saved
2. Re-click **Train Voice Style** with the same voice name — training resumes from the saved step
3. Check **Check for Checkpoint** to see if a resume point exists for a name

### CLI

```bash
python pipeline/train_style.py --wav voice.wav --name myvoice --steps 3000
```

## How It Works

1. **Auto-select** closest reference style via WavLM Layer 3 distance across all 10 presets
2. **Initialize** `style_ttl` from that reference (trainable), freeze `style_dp`
3. **Optimize** `style_ttl` via Adam, minimizing WavLM Layer 3 feature statistics (mean + std MSE) between generated and target audio
4. **Early stop** when loss drops below threshold (default 0.24)

## Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| Training Steps | 3000 | Early stop usually fires ~500 steps |
| Early Stop Threshold | 0.24 | Same-speaker baseline from Chiu et al. 2025 |
| Vocoder Steps | 5 | Denoising iterations |
| Learning Rate | 2e-4 | Adam with ReduceLROnPlateau (patience=200) |
| Speed | 1.05 | Duration scaling |

## Project Structure

```
app.py                  # Gradio UI (stop/resume, audio preprocessing)
start.bat               # Launch script
requirements.txt
reference_styles/       # F1-F5.json, M1-M5.json
pipeline/
  onnx/                 # Supertonic-2 ONNX models (downloaded via UI)
  train_style.py        # Standalone training script
  generate.py           # Synthesis from text + style
  helper.py             # TTS engine, text processing
  utils/
    loss.py             # WavLM-Large Layer 3 loss (WavLMModel)
    model.py            # ONNX→PyTorch conversion (onnxslim + opset17 + _fix_clip)
    style.py            # Style JSON I/O
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
