import argparse
import os
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from helper import load_text_to_speech, load_voice_style, timer

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)


def generate(text, style_path, lang="en", total_step=6, speed=1.05, onnx_dir=None):
    if onnx_dir is None:
        onnx_dir = os.path.join(os.path.dirname(__file__), "onnx_v2")
    tts = load_text_to_speech(onnx_dir)
    voice = load_voice_style([style_path], verbose=True)
    name = os.path.basename(style_path).replace(".json", "")
    print(f"\nSynthesizing: '{text[:60]}...' with voice '{name}'")
    with timer("Generation"):
        wav, duration = tts(text, lang, voice, total_step, speed)
    w = wav[0, : int(tts.sample_rate * duration[0].item())]
    ts = int(time.time())
    fname = f"{name}_{ts}.wav"
    path = os.path.join(SAMPLES_DIR, fname)
    sf.write(path, w, tts.sample_rate)
    print(f"Saved: {path}")
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--text", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--lang", default="en")
    p.add_argument("--total_step", type=int, default=6)
    p.add_argument("--speed", type=float, default=1.05)
    p.add_argument("--onnx_dir", default=None)
    args = p.parse_args()
    generate(args.text, args.style, args.lang, args.total_step, args.speed, args.onnx_dir)
