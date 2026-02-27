import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.utils.data


class CHBMIT4ChDataset(torch.utils.data.Dataset):
    """
    Dataset for cleaned 4-channel CHB-MIT pickles.
    Each file is expected to store a dict with keys:
      - 'X': np.ndarray, shape (4, T)
      - 'Y': int or float label
    """

    def __init__(self, root: str, files: List[str]):
        self.root = root
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path = os.path.join(self.root, self.files[index])
        with open(path, "rb") as f:
            sample: Dict = pickle.load(f)

        X = sample["X"]
        y = int(sample["y"])

        # Replace NaNs with zero and mild per-channel normalization
        X = np.nan_to_num(X, copy=False)
        X = X / (np.quantile(np.abs(X), 0.95, axis=-1, keepdims=True) + 1e-8)

        return torch.as_tensor(X, dtype=torch.float32), y


def prepare_fourch_dataloaders(
    root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Build train/val/test dataloaders from a cleaned 4-channel dataset layout:
      root/train/*.pkl
      root/val/*.pkl
      root/test/*.pkl
    """
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    for d in (train_dir, val_dir, test_dir):
        if not os.path.exists(d):
            raise FileNotFoundError(f"Expected split directory missing: {d}")

    train_files = sorted(f for f in os.listdir(train_dir) if f.endswith(".pkl"))
    val_files = sorted(f for f in os.listdir(val_dir) if f.endswith(".pkl"))
    test_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".pkl"))

    train_ds = CHBMIT4ChDataset(train_dir, train_files)
    val_ds = CHBMIT4ChDataset(val_dir, val_files)
    test_ds = CHBMIT4ChDataset(test_dir, test_files)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, test_loader
