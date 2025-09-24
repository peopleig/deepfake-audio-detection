import os
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import platform
import torchaudio
from .features import extract_features, TARGET_SR
from .augment import augment_audio

class FoRDataset(Dataset):
    def __init__(self, root_dir: str, feature_type: str = "logmel", split: str = "train",
                 train_ratio: float = 0.8, val_ratio: float = 0.1, augment: bool = False,
                 fixed_seconds: float = None, enable_mp3_aug: bool = False):
        self.root_dir = Path(root_dir)
        self.feature_type = feature_type
        self.augment = augment
        self.fixed_samples = int(fixed_seconds * TARGET_SR) if fixed_seconds else None
        self.enable_mp3_aug = enable_mp3_aug

        wavs, labels = self._scan_files(self.root_dir)
        real_idx = [i for i, y in enumerate(labels) if y == 0]
        fake_idx = [i for i, y in enumerate(labels) if y == 1]
        def split_indices(idx_list):
            n = len(idx_list)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train_i = idx_list[:n_train]
            val_i = idx_list[n_train:n_train+n_val]
            test_i = idx_list[n_train+n_val:]
            return {"train": train_i, "val": val_i, "test": test_i}
        parts_real = split_indices(real_idx)
        parts_fake = split_indices(fake_idx)
        part = {"train": [], "val": [], "test": []}
        for k in part.keys():
            part[k] = parts_real[k] + parts_fake[k]
            part[k].sort()
        sel = part["train" if split == "train" else "val" if split == "val" else "test"]

        self.files = [wavs[i] for i in sel]
        self.labels = [labels[i] for i in sel]

    def _scan_files(self, root: Path) -> Tuple[List[Path], List[int]]:
        wavs, labels = [], []
        alias_to_label = {
            0: {"real", "bonafide", "genuine", "human", "authentic"},
            1: {"fake", "spoof", "ai", "synthetic", "tts", "vc", "replay"},
        }
        for p in root.rglob("*.wav"):
            parts = {q.name.lower() for q in p.parents}
            assigned = False
            if alias_to_label[0].intersection(parts):
                wavs.append(p)
                labels.append(0)
                assigned = True
            elif alias_to_label[1].intersection(parts):
                wavs.append(p)
                labels.append(1)
                assigned = True
            if not assigned:
                continue
        if not wavs:
            raise RuntimeError(f"No WAV files found under {root}. Ensure folders contain 'real' and 'fake'.")
        return wavs, labels

    def _pad_or_trim(self, wav: torch.Tensor) -> torch.Tensor:
        if self.fixed_samples is None:
            return wav
        T = wav.size(-1)
        if T == self.fixed_samples:
            return wav
        if T > self.fixed_samples:
            return wav[..., :self.fixed_samples]
        pad = self.fixed_samples - T
        return torch.nn.functional.pad(wav, (0, pad))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != TARGET_SR:
            wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
        wav = self._pad_or_trim(wav)
        if self.augment:
            wav = augment_audio(wav, enable_mp3=self.enable_mp3_aug)
        feat = extract_features(wav, TARGET_SR, self.feature_type) 
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feat, label

def pad_collate(batch):
    feats, labels = zip(*batch)
    M = feats[0].size(1)
    T_max = max(f.size(2) for f in feats)
    padded = []
    for f in feats:
        T = f.size(2)
        if T < T_max:
            f = torch.nn.functional.pad(f, (0, T_max - T))
        padded.append(f)
    x = torch.stack(padded, dim=0)
    y = torch.stack(labels, dim=0)
    return x, y

def make_dataloader(root_dir: str, feature_type="logmel", split="train", batch_size=32,
                    shuffle=True, num_workers=2, augment=False, fixed_seconds=None, enable_mp3_aug=False):
    ds = FoRDataset(root_dir=root_dir, feature_type=feature_type, split=split,
                    augment=augment, fixed_seconds=fixed_seconds, enable_mp3_aug=enable_mp3_aug)
    collate = None if fixed_seconds else pad_collate
    effective_workers = 0 if platform.system() == "Windows" and num_workers == 2 else num_workers
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=effective_workers, collate_fn=collate)
