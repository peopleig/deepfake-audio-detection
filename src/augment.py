import torch
import torchaudio
import torchaudio.transforms as T
import random

TARGET_SR = 16000

def add_noise(waveform, noise_level=0.005):
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise

def time_shift(waveform, shift_max=0.2):
    shift_amt = int(random.uniform(-shift_max, shift_max) * waveform.size(-1))
    return torch.roll(waveform, shifts=shift_amt)

def time_stretch(waveform, rate=1.0):
    if rate == 1.0:
        return waveform
    transform = T.TimeStretch(n_freq=201)
    spec = torch.stft(waveform, n_fft=400, return_complex=True)
    stretched = transform(spec, rate)
    return torch.istft(stretched, n_fft=400)

def pitch_shift(waveform, sr=TARGET_SR, n_steps=2):
    return T.PitchShift(sr, n_steps=n_steps)(waveform)

def volume_change(waveform, gain_db=5.0):
    return waveform * (10 ** (gain_db / 20))

def apply_reverb(waveform):
    reverb = T.Reverberate(sample_rate=TARGET_SR)
    return reverb(waveform)

def bandpass_filter(waveform, low=300, high=3400):
    bandpass = T.BandpassBiquad(TARGET_SR, low, high)
    return bandpass(waveform)

def mp3_compression(waveform, sr=TARGET_SR):
    import tempfile, os
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_wav = f.name
    sf.write(tmp_wav, waveform.squeeze().numpy(), sr)

    tmp_mp3 = tmp_wav.replace(".wav", ".mp3")
    os.system(f"ffmpeg -y -loglevel quiet -i {tmp_wav} -b:a 32k {tmp_mp3}")

    wav, _ = torchaudio.load(tmp_mp3)
    os.remove(tmp_wav)
    os.remove(tmp_mp3)
    return wav.mean(dim=0, keepdim=True)

def augment_audio(waveform, sr=TARGET_SR):
    # Choose random augmentations
    if random.random() < 0.5:
        waveform = add_noise(waveform)
    if random.random() < 0.5:
        waveform = time_shift(waveform)
    if random.random() < 0.3:
        waveform = pitch_shift(waveform, sr, n_steps=random.choice([-2, -1, 1, 2]))
    if random.random() < 0.3:
        waveform = volume_change(waveform, gain_db=random.uniform(-6, 6))
    if random.random() < 0.3:
        waveform = bandpass_filter(waveform)
    if random.random() < 0.2:
        waveform = apply_reverb(waveform)
    if random.random() < 0.2:
        waveform = mp3_compression(waveform, sr)

    return waveform

if _name_ == "_main_":
    wav, sr = torchaudio.load("example.wav")
    wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

    aug_wav = augment_audio(wav)

    torchaudio.save("augment ed_example.wav", aug_wav, TARGET_SR)
    print("Saved augmented audio → augmented_example.wav")