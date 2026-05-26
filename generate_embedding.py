import argparse
import json
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

from config import MODELS_DIR, SAMPLE_RATE, MIN_AUDIO_SEC, REFERENCE_WAV_SR


class StyleEncoder:
    def __init__(self, model_path: str | None = None):
        if ort is None:
            raise RuntimeError("onnxruntime required: .venv pip install onnxruntime")
        default = MODELS_DIR / "style_encoder.onnx"
        path = model_path or str(default)
        self.session = ort.InferenceSession(path)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        print(f"Style encoder loaded: {path}")
        print(f"  inputs:  {self.input_names}")
        print(f"  outputs: {self.output_names}")

    def encode(self, wav_path: str) -> dict:
        if librosa is None:
            raise RuntimeError("librosa/soundfile required: .venv pip install librosa soundfile")

        audio, sr = librosa.load(wav_path, sr=REFERENCE_WAV_SR, mono=True)
        dur = len(audio) / REFERENCE_WAV_SR
        if dur < MIN_AUDIO_SEC:
            raise ValueError(f"Audio too short: {dur:.1f}s (min {MIN_AUDIO_SEC}s)")

        audio = audio.astype(np.float32)
        ref_tensor = np.array([audio], dtype=np.float32)

        feeds = {self.input_names[0]: ref_tensor}
        outputs = self.session.run(self.output_names, feeds)

        result = {}
        for name, arr in zip(self.output_names, outputs):
            result[name] = {
                "dims": list(arr.shape),
                "data": arr.tolist(),
            }
        return result


def extract_embedding_librosa(wav_path: str, target_sr: int = REFERENCE_WAV_SR) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa required")
    audio, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=target_sr, n_mfcc=80)
    return np.mean(mfcc, axis=1, keepdims=True).T


def build_style_from_embedding(
    style_ttl_data: np.ndarray,
    style_dp_data: np.ndarray,
    ttl_dims: list[int] | None = None,
    dp_dims: list[int] | None = None,
) -> dict:
    if ttl_dims is None:
        ttl_dims = [1] + list(style_ttl_data.flatten().reshape(1, -1).shape)
    if dp_dims is None:
        dp_dims = [1] + list(style_dp_data.flatten().reshape(1, -1).shape)

    return {
        "style_ttl": {
            "dims": ttl_dims,
            "data": style_ttl_data.flatten().tolist(),
        },
        "style_dp": {
            "dims": dp_dims,
            "data": style_dp_data.flatten().tolist(),
        },
    }


def load_reference_style(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def probe_dimensions(ref_style_path: str):
    style = load_reference_style(ref_style_path)
    print(f"Reference style: {ref_style_path}")
    for key in ("style_ttl", "style_dp"):
        if key in style:
            dims = style[key]["dims"]
            numel = np.array(style[key]["data"]).size
            print(f"  {key}: dims={dims}, numel={numel}")
    return style


def main():
    parser = argparse.ArgumentParser(description="Generate speaker embeddings from audio")
    parser.add_argument("audio", help="Path to input WAV file")
    parser.add_argument("--output", "-o", default=None, help="Output style JSON path")
    parser.add_argument("--reference", "-r", default=None, help="Reference style JSON to match dimensions")
    parser.add_argument("--model", "-m", default=None, help="Custom style_encoder.onnx path")
    parser.add_argument("--probe-only", action="store_true", help="Only probe reference dimensions")
    args = parser.parse_args()

    if args.probe_only and args.reference:
        probe_dimensions(args.reference)
        return

    if args.model:
        encoder = StyleEncoder(args.model)
        style = encoder.encode(args.audio)
    else:
        ref_style = None
        if args.reference:
            ref_style = load_reference_style(args.reference)
            print(f"Loaded reference: {args.reference}")
            for k in ("style_ttl", "style_dp"):
                if k in ref_style:
                    print(f"  {k} target dims: {ref_style[k]['dims']}")

        ttl_dim = 256
        dp_dim = 256
        if ref_style:
            if "style_ttl" in ref_style:
                ttl_dim = int(np.prod(ref_style["style_ttl"]["dims"]))
            if "style_dp" in ref_style:
                dp_dim = int(np.prod(ref_style["style_dp"]["dims"]))

        print(f"Generating dummy embedding with ttl_dim={ttl_dim}, dp_dim={dp_dim}")
        print("(Replace this with a real speaker encoder model for actual voice cloning)")

        style = build_style_from_embedding(
            np.random.randn(ttl_dim).astype(np.float32) * 0.01,
            np.random.randn(dp_dim).astype(np.float32) * 0.01,
        )

        if ref_style:
            if "style_ttl" in ref_style:
                style["style_ttl"]["dims"] = ref_style["style_ttl"]["dims"]
            if "style_dp" in ref_style:
                style["style_dp"]["dims"] = ref_style["style_dp"]["dims"]

    out_path = args.output or str(Path(args.audio).with_suffix(".style.json"))
    with open(out_path, "w") as f:
        json.dump(style, f)
    print(f"Saved style JSON: {out_path}")


if __name__ == "__main__":
    main()
