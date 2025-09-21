import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm

class DeepfakeDataset(Dataset):
    def __init__(self, feature_dir):
        self.files = list(Path(feature_dir).glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        mel = data["mel"]  # (n_mels, time)
        label = data["label"]
        return mel, torch.tensor(label, dtype=torch.float32)

class CNNDetector(nn.Module):
    def __init__(self, n_mels=128):
        super(CNNDetector, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(X).squeeze(1)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} → Train Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                X, y = X.to(device), y.to(device)
                preds = model(X).squeeze(1)
                loss = criterion(preds, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} → Val Loss: {avg_val_loss:.4f}")

    print("✅ Training Finished")
    torch.save(model.state_dict(), "deepfake_detector.pth")
    print("Model saved as deepfake_detector.pth")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = DeepfakeDataset("processed/train")
    val_dataset = DeepfakeDataset("processed/val")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = CNNDetector(n_mels=128).to(device)

    train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3)
