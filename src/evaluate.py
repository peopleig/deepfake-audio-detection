import argparse
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from .dataset import make_dataloader
from .model import ResNetAudioClassifier

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="FoR variant root to evaluate on (e.g., data/for-rerec)")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--fixed_seconds", type=float, default=None)
    ap.add_argument("--feature_type", type=str, default="logmel")
    ap.add_argument("--threshold", type=float, default=0.5)
    return ap.parse_args()

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    y_true = []
    y_score = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        y_true.extend(y.numpy().tolist())
        y_score.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    # Threshold for metrics that require hard labels
    import numpy as np
    y_pred = (np.array(y_score) > threshold).astype(int)
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(f"AUC: {auc:.4f}  F1: {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = make_dataloader(args.data_dir, feature_type=args.feature_type, split="test",
                             batch_size=args.batch_size, shuffle=False, augment=False,
                             fixed_seconds=args.fixed_seconds, enable_mp3_aug=False)
    model = ResNetAudioClassifier(in_channels=1, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    evaluate(model, loader, device, threshold=args.threshold)

if __name__ == "__main__":
    main()
