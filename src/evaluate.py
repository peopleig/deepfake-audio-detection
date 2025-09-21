import torch
from torch.utils.data import DataLoader
from dataset import ASVspoofDataset  # or DeepfakeDataset if using preprocessed .pt
from model import ResNetAudioClassifier  # or CNNDetector if you trained that

device = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, dataloader, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze(1)
            preds = (outputs > threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")

    # Optional: compute additional metrics
    try:
        from sklearn.metrics import f1_score, confusion_matrix
        f1 = f1_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(cm)
    except ImportError:
        print("Install scikit-learn to see F1 score and confusion matrix.")

    return accuracy


if __name__ == "__main__":
    # Load dataset
    test_dataset = ASVspoofDataset("/path/to/ASVspoof2021", subset="eval", feature_type="LFCC", augment=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    model = ResNetAudioClassifier(n_input_channels=1, n_classes=1).to(device)
    model.load_state_dict(torch.load("deepfake_detector.pth", map_location=device))

    evaluate(model, test_loader)
