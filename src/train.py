import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from .dataset import make_dataloader
from .model import ResNetAudioClassifier
import logging
import sys
from datetime import datetime
import time
import json
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message="In 2.9, this function's implementation will be changed")


def setup_logging(log_file="log.txt"):
    """Setup logging to both console and file"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_epoch_counter(counter_file="epoch_counter.json"):
    """Load total epoch counter from file"""
    if Path(counter_file).exists():
        with open(counter_file, 'r') as f:
            data = json.load(f)
            return data.get('total_epochs', 0)
    return 0


def save_epoch_counter(total_epochs, counter_file="epoch_counter.json"):
    """Save total epoch counter to file"""
    with open(counter_file, 'w') as f:
        json.dump({'total_epochs': total_epochs}, f, indent=2)


def save_latest_metrics(epoch, total_epochs, cumulative_epoch, train_loss, val_loss,
                       train_acc, val_acc, epoch_time, samples, lr, best_val, checkpoint_saved,
                       metrics_file="latest_metrics.json"):
    """Save latest epoch metrics to JSON for frontend"""
    metrics = {
        "current_epoch": epoch,
        "total_epochs_planned": total_epochs,
        "cumulative_epochs": cumulative_epoch,
        "train_loss": round(train_loss, 4),
        "train_accuracy": round(train_acc, 2),
        "val_loss": round(val_loss, 4),
        "val_accuracy": round(val_acc, 2),
        "best_val_loss": round(best_val, 4),
        "epoch_time_seconds": int(epoch_time),
        "samples_trained": samples,
        "learning_rate": lr,
        "checkpoint_saved": checkpoint_saved,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def log_epoch_stats(epoch, total_epochs, cumulative_epoch, train_loss, val_loss, 
                   train_acc, val_acc, epoch_time, samples_trained, lr, checkpoint_saved=False):
    """Log comprehensive epoch statistics"""
    minutes = int(epoch_time // 60)
    seconds = int(epoch_time % 60)
    time_str = f"{minutes}m {seconds}s"
    
    log_msg = f"""
{'='*70}
Epoch [{epoch}/{total_epochs}] | Cumulative: {cumulative_epoch} | {datetime.now().strftime('%H:%M:%S')}
{'='*70}
  Time:           {time_str}
  Samples:        {samples_trained}
  Learning Rate:  {lr:.6f}
  
  Training:
    Loss:         {train_loss:.4f}
    Accuracy:     {train_acc:.2f}%
  
  Validation:
    Loss:         {val_loss:.4f}
    Accuracy:     {val_acc:.2f}%
  
  Checkpoint:     {'[SAVED]' if checkpoint_saved else '[NOT SAVED]'}
{'='*70}
"""
    logging.info(log_msg)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Path to FoR variant root")
    ap.add_argument("--feature_type", type=str, default="logmel")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fixed_seconds", type=float, default=None, help="Pad/trim to this length")
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--mp3_aug", action="store_true")
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--out", type=str, default="runs/for_resnet34.pt")
    ap.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--max_files", type=int, default=None, help="Limit dataset to N files")
    ap.add_argument("--file_selection", type=str, default="first", choices=["first", "last"], 
                    help="Select 'first' or 'last' N files when max_files is set")
    return ap.parse_args()


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Training", unit="batch", ncols=120, leave=True)
    
    for batch_idx, (x, y) in enumerate(progress_bar):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        current_loss = running_loss / total
        current_acc = (correct / total) * 100
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%',
            'samples': f'{total}/{len(loader.dataset)}'
        })
    
    avg_loss = running_loss / total
    accuracy = (correct / total) * 100
    return avg_loss, accuracy


@torch.no_grad()
def eval_loss(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(loader, desc="Validation", unit="batch", ncols=120, leave=True)
    
    for x, y in progress_bar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        logits = model(x).squeeze()
        loss = criterion(logits, y)
        
        running_loss += loss.item() * x.size(0)
        
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == y).sum().item()
        total += y.size(0)
        
        current_loss = running_loss / total
        current_acc = (correct / total) * 100
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.2f}%'
        })
    
    avg_loss = running_loss / total
    accuracy = (correct / total) * 100
    return avg_loss, accuracy


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger = setup_logging("log.txt")
    cumulative_epochs = load_epoch_counter()
    
    print("\n" + "="*70)
    print(f"🔍 CUDA CHECK:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("   ⚠️  WARNING: Running on CPU - training will be VERY slow!")
    print("="*70 + "\n")
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Training Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"{'='*70}")
    logging.info(f"Configuration:")
    logging.info(f"  Data Dir:       {args.data_dir}")
    logging.info(f"  Epochs:         {args.epochs}")
    logging.info(f"  Batch Size:     {args.batch_size}")
    logging.info(f"  Learning Rate:  {args.lr}")
    logging.info(f"  Feature Type:   {args.feature_type}")
    logging.info(f"  Fixed Seconds:  {args.fixed_seconds}")
    logging.info(f"  Augment:        {args.augment}")
    logging.info(f"  Device:         {device}")
    logging.info(f"  Resume From:    {args.resume if args.resume else 'None (fresh start)'}")
    logging.info(f"  Max Files:      {args.max_files if args.max_files else 'All'}")
    logging.info(f"  File Selection: {args.file_selection}")
    logging.info(f"  Cumulative Epochs (all time): {cumulative_epochs}")
    logging.info(f"{'='*70}\n")

    train_loader = make_dataloader(
        args.data_dir, feature_type=args.feature_type, split="train",
        batch_size=args.batch_size, shuffle=True, augment=args.augment,
        fixed_seconds=args.fixed_seconds, enable_mp3_aug=args.mp3_aug, 
        max_files=args.max_files, file_selection=args.file_selection
    )
    val_loader = make_dataloader(
        args.data_dir, feature_type=args.feature_type, split="val",
        batch_size=args.batch_size, shuffle=False, augment=False,
        fixed_seconds=args.fixed_seconds, enable_mp3_aug=False, 
        max_files=args.max_files, file_selection=args.file_selection
    )

    model = ResNetAudioClassifier(in_channels=1, pretrained=args.pretrained).to(device)
    print(f"✅ Model loaded on: {next(model.parameters()).device}\n")
    
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
        logging.info(f"[OK] Resumed from checkpoint: {args.resume}")
        logging.info(f"  Starting from epoch {start_epoch}, Best val loss: {best_val:.4f}\n")

    Path("runs").mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = eval_loss(model, val_loader, device, criterion)
        
        epoch_time = time.time() - epoch_start_time
        cumulative_epochs += 1
        
        checkpoint_saved = False
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
            }, args.out)
            checkpoint_saved = True
        
        log_epoch_stats(
            epoch=epoch, total_epochs=args.epochs, cumulative_epoch=cumulative_epochs,
            train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc,
            epoch_time=epoch_time, samples_trained=len(train_loader.dataset),
            lr=optimizer.param_groups[0]['lr'], checkpoint_saved=checkpoint_saved
        )
        
        save_epoch_counter(cumulative_epochs)
        save_latest_metrics(
            epoch=epoch, total_epochs=args.epochs, cumulative_epoch=cumulative_epochs,
            train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc,
            epoch_time=epoch_time, samples=len(train_loader.dataset),
            lr=optimizer.param_groups[0]['lr'], best_val=best_val, checkpoint_saved=checkpoint_saved
        )
    
    logging.info(f"\n{'='*70}")
    logging.info(f"Training Session Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Best Validation Loss: {best_val:.4f}")
    logging.info(f"Model saved to: {args.out}")
    logging.info(f"Total Cumulative Epochs (all time): {cumulative_epochs}")
    logging.info(f"{'='*70}\n")


if __name__ == "__main__":
    main()
