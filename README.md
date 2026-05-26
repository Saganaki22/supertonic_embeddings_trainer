# Supertonic Embeddings Trainer

Local voice cloning pipeline for [SupertonicTTS](https://github.com/kdrkdrkdr/supertonic.embed). Upload a WAV sample → train a style embedding → synthesize speech in that voice. No cloud API needed.

Based on gradient-based inverse optimization of frozen TTS models, matching WavLM-Large Layer 3 features (Chiu et al. 2025).

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

Download the 10 reference style JSONs from [HuggingFace](https://huggingface.co/kdrkdrkdr/supertonic.embed/tree/main/voice_styles) into `reference_styles/` (F1-F5, M1-M5).

Download the 4 Supertonic ONNX models and `tts.json` / `unicode_indexer.json` into `pipeline/onnx/`:
- `duration_predictor.onnx`
- `text_encoder.onnx`
- `vector_estimator.onnx`
- `vocoder.onnx`

## Usage

```bash
# Launch Gradio UI
start.bat
# or: python app.py
```

### Gradio Tabs

| Tab | Description |
|-----|-------------|
| **Clone Voice** | Upload WAV → set name/steps/threshold → train style embedding |
| **Synthesize** | Enter text + select trained style → generate speech |
| **Browse Styles** | Preview reference and trained voice styles |

### CLI

```bash
python pipeline/train_style.py --wav voice.wav --name myvoice --steps 1000
```

## How It Works

1. **Auto-select** closest reference style via WavLM Layer 3 distance across all 10 presets
2. **Initialize** `style_ttl` from that reference (trainable), freeze `style_dp`
3. **Optimize** `style_ttl` via Adam, minimizing WavLM Layer 3 feature statistics (mean + std MSE) between generated and target audio
4. **Early stop** when loss drops below threshold (default 0.15)

## Project Structure

```
app.py                  # Gradio UI
start.bat               # Launch script
requirements.txt
reference_styles/       # F1-F5.json, M1-M5.json
pipeline/
  onnx/                 # Supertonic ONNX models (not included)
  train_style.py        # Standalone training script
  generate.py           # Synthesis from text + style
  helper.py             # TTS engine, text processing
  utils/
    loss.py             # WavLM-Large Layer 3 loss
    model.py            # ONNX→PyTorch model wrapper
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
