import argparse
import json
import torch
import torchaudio
from pathlib import Path
from src.model import ResNetAudioClassifier
from src.features import extract_features

@torch.no_grad()
def predict(model, wav_path, device, feature_type="logmel", fixed_seconds=2.0, threshold=0.5):
    wav, sr = torchaudio.load(wav_path)

    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    feats = extract_features(wav, sr, feature_type=feature_type, fixed_seconds=fixed_seconds)
    x = feats.unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.sigmoid(logits).item()
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
