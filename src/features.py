import torch
import torchaudio
import torchaudio.transforms as T

TARGET_SR = 16000
N_MELS = 128
N_MFCC = 40

mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR,
    n_fft=1024,
    hop_length=256,
    n_mels=N_MELS
)

def extract_logmel(waveform):
    mel = mel_transform(waveform)
    mel_db = torch.log(mel + 1e-6)  
    return mel_db    

mfcc_transform = T.MFCC(
    sample_rate=TARGET_SR,
    n_mfcc=N_MFCC,
    melkwargs={"n_fft": 1024, "n_mels": N_MELS, "hop_length": 256}
)

def extract_mfcc(waveform):
    mfcc = mfcc_transform(waveform)
    return mfcc    


spec_transform = T.Spectrogram(
    n_fft=1024,
    hop_length=256,
    power=2
)

def extract_spectrogram(waveform):
    spec = spec_transform(waveform)
    spec_db = torch.log(spec + 1e-6)
    return spec_db   


def extract_ssl_embeddings(waveform, model=None, processor=None):
    from transformers import Wav2Vec2Processor, Wav2Vec2Model

    if model is None or processor is None:
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values)
    return outputs.last_hidden_state.squeeze(0) 

if _name_ == "_main_":
    wav, sr = torchaudio.load("example.wav")

    if sr != TARGET_SR:
        wav = T.Resample(sr, TARGET_SR)(wav)

    wav = wav.mean(dim=0, keepdim=True)

    logmel = extract_logmel(wav)
    mfcc = extract_mfcc(wav)
    spec = extract_spectrogram(wav)
    ssl_emb = extract_ssl_embeddings(wav)

    print("Log-Mel:", logmel.shape)
    print("MFCC:", mfcc.shape)
    print("Spectrogram:", spec.shape)
    print("SSL Embeddings:", ssl_emb.shape)