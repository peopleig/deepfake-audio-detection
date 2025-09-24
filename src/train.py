import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from .dataset import make_dataloader
from .model import ResNetAudioClassifier

import warnings
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Path to FoR variant root (e.g., data/for-2sec)")
    ap.add_argument("--feature_type", type=str, default="logmel")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fixed_seconds", type=float, default=None, help="If set, pad/trim to this length (e.g., 2.0)")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--mp3_aug", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--out", type=str, default="runs/for_resnet34.pt")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    return ap.parse_args()

def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))

@torch.no_grad()
def eval_loss(model, loader, device, criterion):
    model.eval()
    total = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total += loss.item()
    return total / max(1, len(loader))

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_dataloader(args.data_dir, feature_type=args.feature_type, split="train",
                                   batch_size=args.batch_size, shuffle=True, augment=args.augment,
                                   fixed_seconds=args.fixed_seconds, enable_mp3_aug=args.mp3_aug)
    val_loader = make_dataloader(args.data_dir, feature_type=args.feature_type, split="val",
                                 batch_size=args.batch_size, shuffle=False, augment=False,
                                 fixed_seconds=args.fixed_seconds, enable_mp3_aug=False)

    model = ResNetAudioClassifier(in_channels=1, pretrained=args.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = int(ckpt["epoch"]) + 1
        if "best_val" in ckpt:
            best_val = float(ckpt["best_val"])

    Path("runs").mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, device, criterion, optimizer)
        vl = eval_loss(model, val_loader, device, criterion)
        print(f"Epoch {epoch}: train_loss={tr:.4f} val_loss={vl:.4f}")
        if vl < best_val:
            best_val = vl
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            }, args.out)
            print(f"Saved best checkpoint → {args.out}")

if __name__ == "__main__":
    main()
