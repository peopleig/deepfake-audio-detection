import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, 
    accuracy_score, precision_score, recall_score,
    roc_curve, classification_report
)
from .dataset import make_dataloader
from .model import ResNetAudioClassifier
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="FoR variant root to evaluate on")
    ap.add_argument("--ckpt", type=str, required=True, help="Primary checkpoint to evaluate")
    ap.add_argument("--compare_ckpt", type=str, default=None, help="Optional: Second checkpoint for comparison")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--fixed_seconds", type=float, default=None)
    ap.add_argument("--feature_type", type=str, default="logmel")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save_plots", action="store_true", help="Save plots to disk")
    return ap.parse_args()


@torch.no_grad()
def get_predictions(model, loader, device):
    """Get predictions and ground truth from model"""
    model.eval()
    y_true = []
    y_score = []
    
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        y_true.extend(y.numpy().tolist())
        y_score.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    
    return np.array(y_true), np.array(y_score)


def calculate_metrics(y_true, y_score, threshold=0.5):
    """Calculate comprehensive evaluation metrics"""
    y_pred = (y_score >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)
    
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'specificity': specificity,
        'fpr': fpr,
        'fnr': fnr,
        'confusion_matrix': cm,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def print_metrics(metrics, model_name="Model"):
    """Print metrics in a beautiful formatted way"""
    print(f"\n{'='*70}")
    print(f"📊 {model_name} Evaluation Results")
    print(f"{'='*70}")
    
    print(f"\n🎯 Classification Metrics:")
    print(f"   Accuracy:     {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"   Precision:    {metrics['precision']:.4f}")
    print(f"   Recall:       {metrics['recall']:.4f}")
    print(f"   F1 Score:     {metrics['f1']:.4f}")
    print(f"   AUC-ROC:      {metrics['auc']:.4f}")
    print(f"   Specificity:  {metrics['specificity']:.4f}")
    
    print(f"\n📉 Error Rates:")
    print(f"   False Positive Rate: {metrics['fpr']:.4f}")
    print(f"   False Negative Rate: {metrics['fnr']:.4f}")
    
    print(f"\n🔢 Confusion Matrix Breakdown:")
    print(f"   True Positives:  {metrics['tp']}")
    print(f"   True Negatives:  {metrics['tn']}")
    print(f"   False Positives: {metrics['fp']}")
    print(f"   False Negatives: {metrics['fn']}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Real  Fake")
    print(f"   Actual Real  {metrics['tn']:4d}  {metrics['fp']:4d}")
    print(f"          Fake  {metrics['fn']:4d}  {metrics['tp']:4d}")
    print(f"{'='*70}\n")


def plot_confusion_matrix(cm, model_name="Model", save_path=None):
    """Plot beautiful confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Real', 'Deepfake'],
        yticklabels=['Real', 'Deepfake'],
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved confusion matrix to {save_path}")
    plt.show()


def plot_roc_curve(y_true, y_score, auc, model_name="Model", save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'{model_name} - ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved ROC curve to {save_path}")
    plt.show()


def compare_models(metrics1, metrics2, y_true1, y_score1, y_true2, y_score2, 
                   name1="Model 1", name2="Model 2", save_dir=None):
    """Create comparison visualizations between two models"""
    
    # 1. ROC Curve Comparison
    plt.figure(figsize=(10, 6))
    
    fpr1, tpr1, _ = roc_curve(y_true1, y_score1)
    fpr2, tpr2, _ = roc_curve(y_true2, y_score2)
    
    plt.plot(fpr1, tpr1, linewidth=2, label=f'{name1} (AUC = {metrics1["auc"]:.4f})')
    plt.plot(fpr2, tpr2, linewidth=2, label=f'{name2} (AUC = {metrics2["auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Model Comparison - ROC Curves', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/roc_comparison.png", dpi=300, bbox_inches='tight')
        print(f"💾 Saved ROC comparison to {save_dir}/roc_comparison.png")
    plt.show()
    
    # 2. Metrics Comparison Bar Chart
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'Specificity']
    model1_values = [metrics1['accuracy'], metrics1['precision'], metrics1['recall'], 
                     metrics1['f1'], metrics1['auc'], metrics1['specificity']]
    model2_values = [metrics2['accuracy'], metrics2['precision'], metrics2['recall'],
                     metrics2['f1'], metrics2['auc'], metrics2['specificity']]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, model1_values, width, label=name1, alpha=0.8)
    bars2 = ax.bar(x + width/2, model2_values, width, label=name2, alpha=0.8)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/metrics_comparison.png", dpi=300, bbox_inches='tight')
        print(f"💾 Saved metrics comparison to {save_dir}/metrics_comparison.png")
    plt.show()
    
    # 3. Prediction Distribution Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Model 1 distribution
    ax1.hist(y_score1[y_true1 == 0], bins=50, alpha=0.7, label='Real', color='blue')
    ax1.hist(y_score1[y_true1 == 1], bins=50, alpha=0.7, label='Deepfake', color='red')
    ax1.axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Prediction Score', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'{name1} - Score Distribution', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Model 2 distribution
    ax2.hist(y_score2[y_true2 == 0], bins=50, alpha=0.7, label='Real', color='blue')
    ax2.hist(y_score2[y_true2 == 1], bins=50, alpha=0.7, label='Deepfake', color='red')
    ax2.axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Threshold')
    ax2.set_xlabel('Prediction Score', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title(f'{name2} - Score Distribution', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/distribution_comparison.png", dpi=300, bbox_inches='tight')
        print(f"💾 Saved distribution comparison to {save_dir}/distribution_comparison.png")
    plt.show()


def evaluate_single_model(args, model_name="Model"):
    """Evaluate a single model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data
    loader = make_dataloader(
        args.data_dir, feature_type=args.feature_type, split="test",
        batch_size=args.batch_size, shuffle=False, augment=False,
        fixed_seconds=args.fixed_seconds, enable_mp3_aug=False
    )
    
    # Load model
    model = ResNetAudioClassifier(in_channels=1, pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    
    # Get predictions
    y_true, y_score = get_predictions(model, loader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_score, args.threshold)
    
    # Print results
    print_metrics(metrics, model_name)
    
    # Plot confusion matrix
    save_cm = f"confusion_matrix_{model_name.replace(' ', '_')}.png" if args.save_plots else None
    plot_confusion_matrix(metrics['confusion_matrix'], model_name, save_cm)
    
    # Plot ROC curve
    save_roc = f"roc_curve_{model_name.replace(' ', '_')}.png" if args.save_plots else None
    plot_roc_curve(y_true, y_score, metrics['auc'], model_name, save_roc)
    
    return metrics, y_true, y_score


def main():
    args = parse_args()
    
    # Evaluate primary model
    metrics1, y_true1, y_score1 = evaluate_single_model(args, model_name=Path(args.ckpt).stem)
    
    # If comparison model provided, evaluate and compare
    if args.compare_ckpt:
        print(f"\n{'#'*70}")
        print(f"# COMPARING WITH SECOND MODEL")
        print(f"{'#'*70}\n")
        
        # Temporarily swap checkpoint
        original_ckpt = args.ckpt
        args.ckpt = args.compare_ckpt
        
        metrics2, y_true2, y_score2 = evaluate_single_model(args, model_name=Path(args.compare_ckpt).stem)
        
        # Restore original
        args.ckpt = original_ckpt
        
        # Create comparison plots
        print(f"\n{'='*70}")
        print("📊 GENERATING COMPARISON VISUALIZATIONS")
        print(f"{'='*70}\n")
        
        save_dir = "comparison_plots" if args.save_plots else None
        if save_dir:
            Path(save_dir).mkdir(exist_ok=True)
        
        compare_models(
            metrics1, metrics2, y_true1, y_score1, y_true2, y_score2,
            name1=Path(original_ckpt).stem,
            name2=Path(args.compare_ckpt).stem,
            save_dir=save_dir
        )


if __name__ == "__main__":
    main()
