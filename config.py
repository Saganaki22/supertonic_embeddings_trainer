from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_CONFIGS = {
    "v2": {
        "hf_repo": "Supertone/supertonic-2",
        "onnx_dir": BASE_DIR / "pipeline" / "onnx_v2",
        "styles_dir": BASE_DIR / "reference_styles_v2",
        "default_steps": 3000,
        "default_vocoder_steps": 5,
        "default_lr": 2e-4,
        "threshold_wavlm": 0.24,
        "threshold_ecapa": 0.15,
        "languages": ["en", "ko", "es", "pt", "fr"],
    },
    "v3": {
        "hf_repo": "Supertone/supertonic-3",
        "onnx_dir": BASE_DIR / "pipeline" / "onnx_v3",
        "styles_dir": BASE_DIR / "reference_styles_v3",
        "default_steps": 3000,
        "default_vocoder_steps": 6,
        "default_lr": 2e-4,
        "threshold_wavlm": 0.24,
        "threshold_ecapa": 0.15,
        "languages": [
            "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et",
            "fi", "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl",
            "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
        ],
    },
}

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
SAMPLE_RATE = 44100
MIN_AUDIO_SEC = 3
MAX_AUDIO_SEC = 30
REFERENCE_WAV_SR = 44100
