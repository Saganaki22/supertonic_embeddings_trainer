from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
REF_STYLES_DIR = BASE_DIR / "reference_styles"

HF_BASE = "https://huggingface.co/Supertone/supertonic-3/resolve/main"

ONNX_FILES = {
    "config": f"{HF_BASE}/tts.json",
    "duration_predictor": f"{HF_BASE}/duration_predictor.onnx",
    "text_encoder": f"{HF_BASE}/text_encoder.onnx",
    "vector_estimator": f"{HF_BASE}/vector_estimator.onnx",
    "vocoder": f"{HF_BASE}/vocoder.onnx",
    "unicode_indexer": f"{HF_BASE}/unicode_indexer.json",
    "style_encoder": f"{HF_BASE}/style_encoder.onnx",
}

SAMPLE_RATE = 44100
MIN_AUDIO_SEC = 3
MAX_AUDIO_SEC = 30
REFERENCE_WAV_SR = 44100
