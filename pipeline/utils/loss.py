import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

WAVLM_LAYER = 3


def load_wavlm(device):
    from transformers import WavLMModel
    model = WavLMModel.from_pretrained("microsoft/wavlm-large")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_wavlm_targets(wavlm, wav_tensor, device, sr=44100):
    wav_16k = torchaudio.functional.resample(
        wav_tensor.unsqueeze(0).to(device), sr, 16000
    )
    with torch.no_grad():
        out = wavlm(wav_16k, output_hidden_states=True)
    feat = out.hidden_states[WAVLM_LAYER]
    return feat.mean(dim=1).detach(), feat.std(dim=1).detach()


def wavlm_loss(wavlm, gen_wav, target_features, device, sr=44100):
    wav_16k = torchaudio.functional.resample(gen_wav.unsqueeze(0), sr, 16000)
    out = wavlm(wav_16k, output_hidden_states=True)
    feat = out.hidden_states[WAVLM_LAYER]
    tgt_mean, tgt_std = target_features
    return (
        F.mse_loss(feat.mean(dim=1), tgt_mean)
        + F.mse_loss(feat.std(dim=1), tgt_std)
    )


def load_ecapa(device):
    from speechbrain.inference.speaker import EncoderClassifier
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pipeline/pretrained/ecapa",
        run_opts={"device": str(device)},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_ecapa_targets(ecapa, wav_tensor, device, sr=44100):
    wav_16k = torchaudio.functional.resample(wav_tensor, sr, 16000)
    with torch.no_grad():
        emb = ecapa.encode_batch(wav_16k.unsqueeze(0).cpu())
    return emb.squeeze().to(device).detach()


def ecapa_loss(ecapa, gen_wav, target_emb, device, sr=44100):
    wav_16k = torchaudio.functional.resample(gen_wav, sr, 16000)
    with torch.no_grad():
        gen_emb = ecapa.encode_batch(wav_16k.unsqueeze(0).detach().cpu())
    gen_emb = gen_emb.squeeze().to(device)
    sim = F.cosine_similarity(gen_emb.unsqueeze(0), target_emb.unsqueeze(0))
    return 1.0 - sim
