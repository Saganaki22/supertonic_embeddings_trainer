import os
import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio
from transformers import WavLMModel

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_audio_16khz_mono(file_path):
    wav, sr = sf.read(file_path, dtype="float32")
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = torch.from_numpy(wav).unsqueeze(0).float()
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


class WavLMLoss(nn.Module):
    def __init__(self, target_wav_path: str):
        super().__init__()
        print("  Loading WavLM-Large for perceptual loss...")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.to(DEVICE)

        target_wav = load_audio_16khz_mono(target_wav_path).to(DEVICE)
        with torch.no_grad():
            out = self.model(target_wav, output_hidden_states=True)
            layer3 = out.hidden_states[3]
        self.target_mean = layer3.mean(dim=1).detach()
        self.target_std = layer3.std(dim=1).detach()
        print(f"  Target audio: {target_wav_path} | {target_wav.shape[-1]/16000:.2f}s")

    def forward(self, generated_wav):
        gen_16k = torchaudio.functional.resample(generated_wav, 44100, 16000)
        out = self.model(gen_16k, output_hidden_states=True)
        layer3 = out.hidden_states[3]
        gen_mean = layer3.mean(dim=1)
        gen_std = layer3.std(dim=1)
        loss = (
            nn.functional.mse_loss(gen_mean, self.target_mean)
            + nn.functional.mse_loss(gen_std, self.target_std)
        )
        return loss
