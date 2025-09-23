import torch
import torchaudio
import torchaudio.transforms as T

TARGET_SR = 16000
N_MELS = 80
N_FFT = 1024
HOP = 160  # 10 ms at 16 kHz
WIN_LENGTH = 400  # 25 ms at 16 kHz
EPS = 1e-6

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=N_FFT,
    win_length=WIN_LENGTH,
    hop_length=HOP,
    n_mels=N_MELS,
    center=True,
    power=2.0,
    norm="slaney",
    mel_scale="htk",
)

amplitude_to_db = T.AmplitudeToDB(stype="power")

def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    # waveform shape (C, T) or (T,)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform

def resample_if_needed(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr == TARGET_SR:
        return waveform
    return torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

def extract_logmel(waveform: torch.Tensor) -> torch.Tensor:
    # waveform expected (1, T), float32, 16 kHz
    mel = mel_transform(waveform)  # (n_mels, time)
    mel_db = amplitude_to_db(mel + EPS)  # dB scale
    # normalize per-utterance
    mean = mel_db.mean()
    std = mel_db.std().clamp_min(1e-5)
    mel_db = (mel_db - mean) / std
    # add channel dim for CNN: (1, n_mels, time)
    return mel_db.unsqueeze(0)

def extract_features(waveform: torch.Tensor, sr: int, feature_type: str = "logmel") -> torch.Tensor:
    waveform = ensure_mono(waveform)
    waveform = resample_if_needed(waveform, sr)
    if feature_type.lower() == "logmel":
        return extract_logmel(waveform)
    raise ValueError(f"Unsupported feature_type: {feature_type}")
