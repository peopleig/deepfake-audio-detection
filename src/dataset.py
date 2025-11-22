import os
from pathlib import Path
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import platform
import torchaudio
import soundfile as sf
import numpy as np
from .features import extract_features, TARGET_SR
from .augment import augment_audio


class FoRDataset(Dataset):
    def __init__(self, root_dir: str, feature_type: str = "logmel", split: str = "train",
                 train_ratio: float = 0.8, val_ratio: float = 0.1, augment: bool = False,
                 fixed_seconds: float = None, enable_mp3_aug: bool = False, 
                 max_files: int = None, file_selection: str = "first"):
        self.root_dir = Path(root_dir)
        self.feature_type = feature_type
        self.augment = augment
        self.fixed_samples = int(fixed_seconds * TARGET_SR) if fixed_seconds else None
        self.enable_mp3_aug = enable_mp3_aug
        self.max_files = max_files
        self.file_selection = file_selection

        # Scan files
        wavs, labels = self._scan_files(self.root_dir, max_files=max_files, file_selection=file_selection)
        
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
        
        # Pre-create resampler (reuse across samples)
        self.resampler = None

    def _scan_files(self, root: Path, max_files: int = None, file_selection: str = "first") -> Tuple[List[Path], List[int]]:
        """Scan audio files (assumes clean dataset)"""
        wavs, labels = [], []
        alias_to_label = {
            0: {"real", "bonafide", "genuine", "human", "authentic"},
            1: {"fake", "spoof", "ai", "synthetic", "tts", "vc", "replay"},
        }
        
        print(f"📂 Scanning files in {root}...")
        if max_files:
            print(f"⚠️  Limiting to {file_selection} {max_files} files")
        
        # Collect ALL files first
        all_files = []
        for p in root.rglob("*.wav"):
            parts = {q.name.lower() for q in p.parents}
            assigned = False
            label = None
            
            if alias_to_label[0].intersection(parts):
                label = 0
                assigned = True
            elif alias_to_label[1].intersection(parts):
                label = 1
                assigned = True
            
            if assigned:
                all_files.append((p, label))
        
        # Apply max_files selection
        if max_files and len(all_files) > max_files:
            if file_selection == "last":
                all_files = all_files[-max_files:]
            else:  # "first" or default
                all_files = all_files[:max_files]
        
        # Unpack into separate lists
        wavs = [f[0] for f in all_files]
        labels = [f[1] for f in all_files]
        
        print(f"✅ Found {len(wavs)} files (real: {labels.count(0)}, fake: {labels.count(1)})\n")
        
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
        wav_np, sr = sf.read(str(path), dtype='float32')
        wav = torch.from_numpy(wav_np)
        
        if wav.ndim == 1:
            wav = wav.unsqueeze(0) 
        else:
            wav = wav.transpose(0, 1)
        
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        
        # Resample if needed (use cached resampler)
        if sr != TARGET_SR:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            wav = self.resampler(wav)
        
        wav = self._pad_or_trim(wav)
        
        if self.augment:
            wav = augment_audio(wav, enable_mp3=self.enable_mp3_aug)
        
        feat = extract_features(wav, TARGET_SR, self.feature_type) 
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feat, label


def pad_collate(batch):
    feats, labels = zip(*batch)
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
                    shuffle=True, num_workers=0, augment=False, fixed_seconds=None, 
                    enable_mp3_aug=False, max_files=None, file_selection="first"):
    ds = FoRDataset(root_dir=root_dir, feature_type=feature_type, split=split,
                    augment=augment, fixed_seconds=fixed_seconds, enable_mp3_aug=enable_mp3_aug,
                    max_files=max_files, file_selection=file_selection)
    collate = None if fixed_seconds else pad_collate
    
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0,
        collate_fn=collate,
        pin_memory=True
    )
