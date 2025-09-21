import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from augment import augment_audio
from features import extract_features  

TARGET_SR = 16000  

class ASVspoofDataset(Dataset):
    def _init_(self, root_dir, subset="train", feature_type="LFCC", augment=False):
        """
        Args:
            root_dir (str): Root folder of ASVspoof dataset.
            subset (str): "train", "dev", or "eval".
            feature_type (str): "LFCC" or "spectrogram".
            augment (bool): Whether to apply augmentation.
        """
        self.root_dir = root_dir
        self.subset = subset
        self.feature_type = feature_type
        self.augment = augment

        self.file_paths = []
        self.labels = []

        self._load_file_paths_and_labels()

    def _load_file_paths_and_labels(self):
        """Load audio file paths and corresponding labels."""
        subset_dir = os.path.join(self.root_dir, self.subset)

        for root, _, files in os.walk(subset_dir):
            for file in files:
                if file.endswith(".wav"):
                    self.file_paths.append(os.path.join(root, file))
                    # Basic labeling: bonafide=0, spoof=1
                    if "bonafide" in file.lower():
                        self.labels.append(0)
                    else:
                        self.labels.append(1)

    def _len_(self):
        return len(self.file_paths)

    def _getitem_(self, idx):
        audio_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(audio_path)

        
        if sr != TARGET_SR:
            waveform = torchaudio.transforms.Resample(sr, TARGET_SR)(waveform)

        if self.augment:
            waveform = augment_audio(waveform, TARGET_SR)

        features = extract_features(waveform, TARGET_SR, self.feature_type)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return features, label


def get_dataloader(root_dir, subset="train", feature_type="LFCC", augment=False,
                   batch_size=16, shuffle=True, num_workers=2):
    dataset = ASVspoofDataset(root_dir=root_dir, subset=subset,
                              feature_type=feature_type, augment=augment)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers)
    return dataloader


if _name_ == "_main_":
    root = "/path/to/ASVspoof2021"  
    train_loader = get_dataloader(root, subset="train", feature_type="LFCC", augment=True)

    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"Batch {batch_idx} → Features: {features.shape}, Labels: {labels.shape}")
        break
