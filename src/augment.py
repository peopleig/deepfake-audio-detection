import os
import random
import tempfile
import torch
import torchaudio
import torchaudio.transforms as T
try:
    # torchaudio.functional provides biquad filters across versions
    import torchaudio.functional as F
except Exception:
    F = None

TARGET_SR = 16000

def add_noise(waveform: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    # Additive white noise based on target SNR
    sig_power = waveform.pow(2).mean().clamp_min(1e-8)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return waveform + noise

def time_shift(waveform: torch.Tensor, max_frac: float = 0.2) -> torch.Tensor:
    shift = int(random.uniform(-max_frac, max_frac) * waveform.size(-1))
    return torch.roll(waveform, shifts=shift, dims=-1)

def bandpass_filter(waveform: torch.Tensor, low=300, high=3400) -> torch.Tensor:
    # Prefer functional bandpass if available; fall back to highpass+lowpass; otherwise no-op
    if F is not None:
        try:
            if hasattr(F, "bandpass_biquad"):
                return F.bandpass_biquad(waveform, TARGET_SR, low, high)
        except Exception:
            pass
        try:
            if hasattr(F, "highpass_biquad") and hasattr(F, "lowpass_biquad"):
                wf = F.highpass_biquad(waveform, TARGET_SR, low)
                wf = F.lowpass_biquad(wf, TARGET_SR, high)
                return wf
        except Exception:
            pass
    # As a safe fallback, return the original waveform unmodified
    return waveform

def gain_db(waveform: torch.Tensor, gain: float) -> torch.Tensor:
    return waveform * (10 ** (gain / 20))

def mp3_compress_roundtrip(waveform: torch.Tensor, sr: int = TARGET_SR) -> torch.Tensor:
    # Requires ffmpeg in PATH
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    tmp_mp3 = tmp_wav.replace(".wav", ".mp3")
    torchaudio.save(tmp_wav, waveform, sr)
    os.system(f"ffmpeg -y -loglevel quiet -i {tmp_wav} -b:a 48k {tmp_mp3}")
    wav, _ = torchaudio.load(tmp_mp3)
    try:
        os.remove(tmp_wav)
        os.remove(tmp_mp3)
    except Exception:
        pass
    # ensure mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav

def augment_audio(waveform: torch.Tensor, enable_mp3: bool = False) -> torch.Tensor:
    if random.random() < 0.5:
        waveform = add_noise(waveform, snr_db=random.uniform(10, 30))
    if random.random() < 0.5:
        waveform = time_shift(waveform, max_frac=0.2)
    if random.random() < 0.3:
        waveform = bandpass_filter(waveform)
    if random.random() < 0.3:
        waveform = gain_db(waveform, gain=random.uniform(-6, 6))
    if enable_mp3 and random.random() < 0.2:
        waveform = mp3_compress_roundtrip(waveform, TARGET_SR)
    return waveform
