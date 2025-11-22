import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import torch
import soundfile as sf
import numpy as np
import torchaudio  # Need this for resampling
from src.model import ResNetAudioClassifier
from src.features import extract_features, TARGET_SR


@torch.no_grad()
def predict(model, wav_path, device, feature_type="logmel", fixed_seconds=2.0, threshold=0.5):
    # Use soundfile to load audio (bypassing torchaudio/torchcodec issues)
    wav_np, sr = sf.read(str(wav_path), dtype='float32')
    wav = torch.from_numpy(wav_np)
    
    # Ensure correct shape
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.transpose(0, 1)
    
    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        wav = resampler(wav)
    
    # Pad or trim to fixed length if specified
    if fixed_seconds is not None:
        fixed_samples = int(fixed_seconds * TARGET_SR)
        current_samples = wav.size(-1)
        
        if current_samples > fixed_samples:
            # Trim
            wav = wav[..., :fixed_samples]
        elif current_samples < fixed_samples:
            # Pad
            pad_amount = fixed_samples - current_samples
            wav = torch.nn.functional.pad(wav, (0, pad_amount))
    
    # Extract features WITHOUT fixed_seconds parameter
    feats = extract_features(wav, TARGET_SR, feature_type=feature_type)
    x = feats.unsqueeze(0).to(device)

    logits = model(x)
    print(f"DEBUG: Raw logit = {logits.item()}", file=sys.stderr)
    prob = torch.sigmoid(logits).item()
    print(f"DEBUG: Probability after sigmoid = {prob}", file=sys.stderr)
    verdict = "deepfake" if prob >= threshold else "real"
    return {"verdict": verdict, "probability": round(prob, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to input .wav file")
    parser.add_argument("--ckpt", type=str, default="runs/for_resnet34.pt", help="Path to trained model checkpoint")
    parser.add_argument("--feature_type", type=str, default="logmel", help="Feature type (logmel or lfcc)")
    parser.add_argument("--fixed_seconds", type=float, default=2.0, help="Length to pad/trim audio to")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for deepfake classification")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetAudioClassifier(in_channels=1, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    result = predict(model, args.audio, device, args.feature_type, args.fixed_seconds, args.threshold)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
