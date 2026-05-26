---
license: openrail
language:
    - en
    - ko
    - ja
    - ar
    - bg
    - cs
    - da
    - de
    - el
    - es
    - et
    - fi
    - fr
    - hi
    - hr
    - hu
    - id
    - it
    - lt
    - lv
    - nl
    - pl
    - pt
    - ro
    - ru
    - sk
    - sl
    - sv
    - tr
    - uk
    - vi
pipeline_tag: text-to-speech
tags:
    - text-to-speech
    - speech-synthesis
    - tts
    - onnx
    - multilingual
    - on-device
library_name: supertonic
---

# Supertonic 3 | Lightning Fast, On-Device, Accurate TTS

![Supertonic 3 Preview](img/Supertonic3_HeroImage.png)

<p align="center">
  <a href="https://huggingface.co/spaces/Supertone/supertonic-3"><img src="https://img.shields.io/badge/Demo-Hugging_Face-yellow?style=for-the-badge" alt="Demo"></a>
  <a href="https://github.com/supertone-inc/supertonic"><img src="https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github" alt="Code"></a>
  <a href="https://pypi.org/project/supertonic/"><img src="https://img.shields.io/badge/Python-SDK-blue?style=for-the-badge&logo=python" alt="Python SDK"></a>
</p>

**Supertonic** is a lightweight text-to-speech system for local inference. It runs with ONNX Runtime entirely on your device, with no cloud call required for synthesis.

**Supertonic 3** expands the open-weight release from 5 to **31 languages**, improves reading stability, and reduces repeat/skip failures.

## Quick Start

Install the Python SDK and generate speech immediately. On first run, the SDK downloads the model assets from Hugging Face.

```bash
pip install supertonic
```

```python
from supertonic import TTS

tts = TTS(auto_download=True)
style = tts.get_voice_style(voice_name="M1")

text = "A gentle breeze moved through the open window while everyone listened to the story."
wav, duration = tts.synthesize(text, voice_style=style, lang="en")

tts.save_audio(wav, "output.wav")
print(f"Generated {duration:.2f}s of audio")
```

## What's New in Supertonic 3

- **31 languages**: expanded from the 5-language Supertonic 2 release.
- **More stable reading**: fewer repeat and skip failures, especially on short and long utterances.
- **Higher speaker similarity**: improved similarity across the shared-language set compared with Supertonic 2.
- **Expression tags**: supports simple tags such as `<laugh>`, `<breath>`, and `<sigh>`.

## Custom Voices and Audio Samples

The open-weight package includes fixed preset voice styles for immediate local inference. If you want to hear how Supertonic 3 performs with zero-shot custom voice styles, visit the [Audio Sample Demo](https://supertonic3.github.io/) to compare reference audio and generated speech across several use cases. To create your own Supertonic 3 voice-style JSON from reference audio, use [Supertonic Voice Builder](https://supertonic.supertone.ai/voice-builder); purchased Voice Builder styles include downloadable embeddings for both Supertonic 2 and Supertonic 3.

Here are a few reference/generated pairs from the audio sample demo:

**Call center, English**  
Text: Good morning, thank you for calling. How can I help you today?

| Reference voice | Supertonic 3 output |
|---|---|
| <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/nora_reference.wav"></audio> | <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/nora_supertonic3.wav"></audio> |

**Character voice, Japanese**  
Text: ふふっ、退屈してたところなの。ちょうどいい遊び相手、見つけたかも♪

| Reference voice | Supertonic 3 output |
|---|---|
| <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/moka_reference.wav"></audio> | <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/moka_supertonic3.wav"></audio> |

**Elder character voice, Korean**  
Text: 혼자 떠나기엔 길이 험하구나. 이 낡은 검을 가져가거라. 언젠가 어둠이 네 이름을 부르더라도, 부디 빛을 잊지 말거라.

| Reference voice | Supertonic 3 output |
|---|---|
| <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/alphonse_reference.wav"></audio> | <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/alphonse_supertonic3.wav"></audio> |

**Audiobook, English**  
Text: I was not afraid of silence. I had lived with it long enough to know that, sometimes, it speaks more honestly than people do.

| Reference voice | Supertonic 3 output |
|---|---|
| <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/luna_reference.wav"></audio> | <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/luna_supertonic3.wav"></audio> |

**Audiobook, Japanese**  
Text: その朝、ロンドンの霧はいつになく低く垂れこめていた。私はただの訪問者だと思っていたが、ホームズの目はすでに別の結論にたどり着いていた。

| Reference voice | Supertonic 3 output |
|---|---|
| <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/watson_reference.wav"></audio> | <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/watson_supertonic3.wav"></audio> |

**News, English**  
Text: Here’s a story worth paying attention to. Supertone has released Supertonic 3, its on-device TTS model. This version expands support to thirty-one languages and improves reading stability.

| Reference voice | Supertonic 3 output |
|---|---|
| <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/keld_reference.wav"></audio> | <audio controls preload="metadata" src="https://huggingface.co/Supertone/supertonic-3/resolve/main/audio_samples/keld_supertonic3.wav"></audio> |

## Performance Highlights

Supertonic 3 is designed for practical on-device inference: compact enough to run locally, while staying competitive with much larger open TTS systems.

### Reading Accuracy

<p align="center">
  <img src="img/metrics/s3_vs_measured_wer_range_voxcpm2.png" alt="Supertonic 3 reading accuracy compared with measured model ranges and VoxCPM2">
</p>

Across measured languages, Supertonic 3 stays within a competitive WER/CER range against much larger open TTS models such as VoxCPM2, while preserving a lightweight on-device deployment path. Asterisked languages use CER; the others use WER.

### Supertonic 2 to Supertonic 3

<p align="center">
  <img src="img/metrics/supertonic2_vs_3_comparison.png" alt="Supertonic 2 and Supertonic 3 comparison">
</p>

Compared with Supertonic 2, Supertonic 3 reduces repeat and skip failures, improves speaker similarity across the shared-language set, and expands language coverage from 5 to 31 languages.

### Runtime Footprint

<p align="center">
  <img src="img/metrics/runtime_cpu_gpu_latency_memory.png" alt="Supertonic CPU runtime compared with GPU baselines">
</p>

Supertonic 3 runs fast on CPU, even compared with larger baselines measured on A100 GPU, and uses substantially less memory. It does not require a GPU, which makes local, browser, and edge deployment much easier.

### Model Size

<p align="center">
  <img src="img/metrics/model_size_comparison.png" alt="Model size comparison">
</p>

At about 99M parameters across the public ONNX assets, Supertonic 3 is much smaller than 0.7B to 2B class open TTS systems. The smaller model size is a practical advantage for download size, startup time, and on-device inference.

## Supported Languages

| Code | Language | Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|------|----------|
| `en` | English | `ko` | Korean | `ja` | Japanese | `ar` | Arabic |
| `bg` | Bulgarian | `cs` | Czech | `da` | Danish | `de` | German |
| `el` | Greek | `es` | Spanish | `et` | Estonian | `fi` | Finnish |
| `fr` | French | `hi` | Hindi | `hr` | Croatian | `hu` | Hungarian |
| `id` | Indonesian | `it` | Italian | `lt` | Lithuanian | `lv` | Latvian |
| `nl` | Dutch | `pl` | Polish | `pt` | Portuguese | `ro` | Romanian |
| `ru` | Russian | `sk` | Slovak | `sl` | Slovenian | `sv` | Swedish |
| `tr` | Turkish | `uk` | Ukrainian | `vi` | Vietnamese | | |

## License

This project's sample code is released under the MIT License. See the [GitHub repository](https://github.com/supertone-inc/supertonic) for details.

The accompanying model is released under the OpenRAIL-M License. See the [LICENSE](https://huggingface.co/Supertone/supertonic-3/blob/main/LICENSE) file in this repository for details.

This model was trained using PyTorch, which is licensed under the BSD 3-Clause License but is not redistributed with this project. See the [PyTorch license](https://docs.pytorch.org/FBGEMM/general/License.html) for details.

Copyright (c) 2026 Supertone Inc.
