import json
import os
import re
import time
import numpy as np
import onnxruntime as ort
from contextlib import contextmanager
from typing import Optional
from unicodedata import normalize

AVAILABLE_LANGS = [
    "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
    "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
    "ru", "sk", "sl", "sv", "tr", "uk", "vi",
]


class UnicodeProcessor:
    def __init__(self, unicode_indexer_path: str):
        with open(unicode_indexer_path) as f:
            self.indexer = json.load(f)

    def _preprocess_text(self, text: str, lang: str) -> str:
        text = normalize("NFKD", text)
        text = re.compile(
            "[\U0001f600-\U0001f64f\U0001f300-\U0001f5ff\U0001f680-\U0001f6ff"
            "\U0001f700-\U0001f77f\U0001f780-\U0001f7ff\U0001f800-\U0001f8ff"
            "\U0001f900-\U0001f9ff\U0001fa00-\U0001fa6f\U0001fa70-\U0001faff"
            "\u2600-\u26ff\u2700-\u27bf\U0001f1e6-\U0001f1ff]+",
            flags=re.UNICODE,
        ).sub("", text)
        for k, v in {
            "\u2013": "-", "\u2010": "-", "\u2014": "-", "_": " ",
            "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
            "\u00b4": "'", "`": "'", "[": " ", "]": " ", "|": " ",
            "/": " ", "#": " ", "\u2192": " ", "\u2190": " ",
        }.items():
            text = text.replace(k, v)
        text = re.sub(r"[\u2665\u2606\u2661\u00a9\\]", "", text)
        for k, v in {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}.items():
            text = text.replace(k, v)
        text = re.sub(r" ,", ",", text)
        text = re.sub(r" \.", ".", text)
        text = re.sub(r" !", "!", text)
        text = re.sub(r" \?", "?", text)
        text = re.sub(r" ;", ";", text)
        text = re.sub(r" :", ":", text)
        text = re.sub(r" '", "'", text)
        while '""' in text:
            text = text.replace('""', '"')
        while "''" in text:
            text = text.replace("''", "'")
        while "``" in text:
            text = text.replace("``", "`")
        text = re.sub(r"\s+", " ", text).strip()
        if not re.search(r"[.!?;:,'\"\')\]}\u2026\u3002\u300d\u3011\u300f\u300b\u203a\u00bb]$", text):
            text += "."
        if lang not in AVAILABLE_LANGS:
            raise ValueError(f"Invalid language: {lang}")
        return f"<{lang}>{text}</{lang}>"

    def __call__(self, text_list, lang_list):
        processed = [self._preprocess_text(t, l) for t, l in zip(text_list, lang_list)]
        lengths = np.array([len(t) for t in processed], dtype=np.int64)
        max_len = lengths.max()
        text_ids = np.zeros((len(processed), max_len), dtype=np.int64)
        for i, text in enumerate(processed):
            for j, ch in enumerate(text):
                cp = ord(ch)
                text_ids[i, j] = self.indexer[cp] if cp < len(self.indexer) else -1
        text_mask = length_to_mask(lengths)
        return text_ids, text_mask


def length_to_mask(lengths, max_len=None):
    max_len = max_len or lengths.max()
    ids = np.arange(max_len)
    return (ids < lengths[:, None]).astype(np.float32).reshape(-1, 1, max_len)


def get_latent_mask(wav_lengths, base_chunk_size, chunk_compress_factor):
    chunk = base_chunk_size * chunk_compress_factor
    latent_lengths = (wav_lengths + chunk - 1) // chunk
    return length_to_mask(latent_lengths)


class Style:
    def __init__(self, ttl, dp):
        self.ttl = ttl
        self.dp = dp


class TextToSpeech:
    def __init__(self, cfgs, text_processor, dp_ort, text_enc_ort, vector_est_ort, vocoder_ort):
        self.cfgs = cfgs
        self.text_processor = text_processor
        self.dp_ort = dp_ort
        self.text_enc_ort = text_enc_ort
        self.vector_est_ort = vector_est_ort
        self.vocoder_ort = vocoder_ort
        self.sample_rate = cfgs["ae"]["sample_rate"]
        self.base_chunk_size = cfgs["ae"]["base_chunk_size"]
        self.chunk_compress_factor = cfgs["ttl"]["chunk_compress_factor"]
        self.ldim = cfgs["ttl"]["latent_dim"]

    def sample_noisy_latent(self, duration):
        bsz = len(duration)
        wav_len_max = duration.max() * self.sample_rate
        wav_lengths = (duration * self.sample_rate).astype(np.int64)
        chunk_size = self.base_chunk_size * self.chunk_compress_factor
        latent_len = int((wav_len_max + chunk_size - 1) / chunk_size)
        latent_dim = self.ldim * self.chunk_compress_factor
        noisy = np.random.randn(bsz, latent_dim, latent_len).astype(np.float32)
        lmask = get_latent_mask(wav_lengths, self.base_chunk_size, self.chunk_compress_factor)
        return noisy * lmask, lmask

    def _infer(self, text_list, lang_list, style, total_step, speed=1.05):
        bsz = len(text_list)
        text_ids, text_mask = self.text_processor(text_list, lang_list)
        dur, *_ = self.dp_ort.run(None, {
            "text_ids": text_ids, "style_dp": style.dp, "text_mask": text_mask,
        })
        dur = dur / speed
        text_emb, *_ = self.text_enc_ort.run(None, {
            "text_ids": text_ids, "style_ttl": style.ttl, "text_mask": text_mask,
        })
        xt, latent_mask = self.sample_noisy_latent(dur)
        total_step_np = np.array([total_step] * bsz, dtype=np.float32)
        for step in range(total_step):
            current_step = np.array([step] * bsz, dtype=np.float32)
            xt, *_ = self.vector_est_ort.run(None, {
                "noisy_latent": xt, "text_emb": text_emb, "style_ttl": style.ttl,
                "text_mask": text_mask, "latent_mask": latent_mask,
                "current_step": current_step, "total_step": total_step_np,
            })
        wav, *_ = self.vocoder_ort.run(None, {"latent": xt})
        return wav, dur

    def __call__(self, text, lang, style, total_step, speed=1.05, silence_duration=0.3):
        max_len = 120 if lang in ("ko", "ja") else 300
        chunks = _chunk_text(text, max_len)
        wav_cat, dur_cat = None, None
        for chunk in chunks:
            wav, dur = self._infer([chunk], [lang], style, total_step, speed)
            if wav_cat is None:
                wav_cat = wav
                dur_cat = dur
            else:
                sil = np.zeros((1, int(silence_duration * self.sample_rate)), dtype=np.float32)
                wav_cat = np.concatenate([wav_cat, sil, wav], axis=1)
                dur_cat = dur_cat + dur + silence_duration
        return wav_cat, dur_cat


def _chunk_text(text, max_len=300):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    chunks = []
    for para in paragraphs:
        sentences = re.split(
            r"(?<!Mr\.)(?<!Mrs\.)(?<!Ms\.)(?<!Dr\.)(?<!Prof\.)(?<!Sr\.)(?<!Jr\.)"
            r"(?<!Ph\.D\.)(?<!etc\.)(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)(?<!Inc\.)"
            r"(?<!Ltd\.)(?<!Co\.)(?<!Corp\.)(?<!St\.)(?<!Ave\.)(?<!Blvd\.)"
            r"(?<!\b[A-Z]\.)(?<=[.!?])\s+",
            para,
        )
        current = ""
        for s in sentences:
            if len(current) + len(s) + 1 <= max_len:
                current += (" " if current else "") + s
            else:
                if current:
                    chunks.append(current.strip())
                current = s
        if current:
            chunks.append(current.strip())
    return chunks


def load_text_to_speech(onnx_dir):
    with open(os.path.join(onnx_dir, "tts.json")) as f:
        cfgs = json.load(f)
    opts = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    models = {}
    for name in ["duration_predictor", "text_encoder", "vector_estimator", "vocoder"]:
        models[name] = ort.InferenceSession(
            os.path.join(onnx_dir, f"{name}.onnx"), opts, providers
        )
    with open(os.path.join(onnx_dir, "unicode_indexer.json")) as f:
        indexer = json.load(f)
    return TextToSpeech(
        cfgs,
        UnicodeProcessor(os.path.join(onnx_dir, "unicode_indexer.json")),
        models["duration_predictor"],
        models["text_encoder"],
        models["vector_estimator"],
        models["vocoder"],
    )


def load_voice_style(paths, verbose=False):
    bsz = len(paths)
    with open(paths[0]) as f:
        first = json.load(f)
    ttl_dims = first["style_ttl"]["dims"]
    dp_dims = first["style_dp"]["dims"]
    ttl = np.zeros([bsz, ttl_dims[1], ttl_dims[2]], dtype=np.float32)
    dp = np.zeros([bsz, dp_dims[1], dp_dims[2]], dtype=np.float32)
    for i, p in enumerate(paths):
        with open(p) as f:
            s = json.load(f)
        ttl[i] = np.array(s["style_ttl"]["data"], dtype=np.float32).reshape(ttl_dims[1], ttl_dims[2])
        dp[i] = np.array(s["style_dp"]["data"], dtype=np.float32).reshape(dp_dims[1], dp_dims[2])
    if verbose:
        print(f"Loaded {bsz} voice styles")
    return Style(ttl, dp)


@contextmanager
def timer(name):
    t0 = time.time()
    print(f"{name}...")
    yield
    print(f"  -> {name} in {time.time() - t0:.2f}s")
