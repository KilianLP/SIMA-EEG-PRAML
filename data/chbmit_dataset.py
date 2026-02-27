import os
import pickle
from typing import List, Tuple, Optional

import torch
import torch.utils.data
import numpy as np
from scipy.signal import resample

from trainer import Config
from .validation import validate_pickle_files


def compute_class_weights_from_files(root, files):
    """
    Fast class weight computation by reading only labels from pickle files.
    """
    import pickle

    print(f"\nFast class weight computation (reading {len(files)} files)...")
    num_positives = 0
    num_negatives = 0

    for i, filename in enumerate(files):
        filepath = os.path.join(root, filename)
        try:
            with open(filepath, 'rb') as f:
                sample = pickle.load(f)

            # Extract label (support both 'y' and 'label' keys)
            if 'y' in sample:
                label = int(sample['y'])
            elif 'label' in sample:
                label = int(sample['label'])
            else:
                continue  # Skip files without labels

            if label == 1:
                num_positives += 1
            else:
                num_negatives += 1

        except Exception:
            # Skip corrupted files silently
            continue

        # Progress update every 10k files
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1}/{len(files)} files...")

    total = num_positives + num_negatives
    pos_weight = num_negatives / (num_positives + 1e-8)

    print(f"  Total files: {total:,}")
    print(f"  Positive samples (seizure): {num_positives:,} ({100*num_positives/total:.2f}%)")
    print(f"  Negative samples (no seizure): {num_negatives:,} ({100*num_negatives/total:.2f}%)")
    print(f"  Computed pos_weight: {pos_weight:.2f}\n")

    return pos_weight


def filter_files_by_patient(files, min_patient: int = 1, max_patient: int = 8, exclude_patients: Optional[List[int]] = None):
    """
    Filter files to only include patients in range [min_patient, max_patient].
    """
    exclude_set = set(exclude_patients or [])
    filtered_files = []
    for f in files:
        # Extract patient number from filename (e.g., chb01, chb02, etc.)
        if f.startswith('chb'):
            try:
                patient_num = int(f[3:5])
                if patient_num in exclude_set:
                    continue
                if min_patient <= patient_num <= max_patient:
                    filtered_files.append(f)
            except ValueError:
                continue
    return filtered_files


class CHBMITDataset(torch.utils.data.Dataset):

    def __init__(self, root, files, sampling_rate: int = 200, skip_resample: bool = False, sample_length: float = 10.0):
        self.root = root
        self.files = files
        self.default_rate = 256  # CHB-MIT native sampling rate
        self.sampling_rate = sampling_rate if not skip_resample else self.default_rate
        self.skip_resample = skip_resample
        self.sample_length = sample_length
        self.failed_files = set()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index):
        """
        Load and preprocess a single sample.
        """
        filepath = os.path.join(self.root, self.files[index])

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with open(filepath, 'rb') as f:
                    sample = pickle.load(f)

                # Handle both 'X'/'y' and 'data'/'label' key conventions
                if 'X' in sample and 'y' in sample:
                    X = sample['X']
                    y = int(sample['y'])
                elif 'data' in sample and 'label' in sample:
                    X = sample['data']
                    y = int(sample['label'])
                else:
                    raise ValueError(f"Unknown keys in pickle file: {list(sample.keys())}")

                # Optional resampling to target sampling rate
                # Input (process2): 4 channels, ~2048 samples at 256 Hz (8 s)
                target_len = int(self.sample_length * self.sampling_rate)
                if not self.skip_resample and self.sampling_rate != self.default_rate:
                    X = resample(X, target_len, axis=-1)

                # Normalize by 95th percentile per channel
                X = X / (np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True) + 1e-8)

                X = torch.FloatTensor(X)
                return X, y

            except Exception as e:
                if attempt == max_retries - 1:
                    # After max retries, log failure and return zero sample
                    if filepath not in self.failed_files:
                        print(f"\n⚠ Failed to load {filepath} after {max_retries} attempts: {type(e).__name__}")
                        self.failed_files.add(filepath)
                    # Return zeros with correct shape (16 channels, 2000 time steps for 10s at 200Hz)
                    return torch.zeros(16, 10 * self.sampling_rate), 0

        # Fallback (should never reach here)
        return torch.zeros(4, int(self.sample_length * self.sampling_rate)), 0


def prepare_chbmit_dataloaders(config,):
    """
    Prepare train, validation, and test dataloaders for CHB-MIT dataset.
    """
    root = config.data.root_path

    # Verify data directory exists
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Data directory not found: {train_dir}\n"
            f"Please verify that config.data.root_path is correct: {root}"
        )

    print("=" * 80)
    print("Preparing CHB-MIT Dataloaders")
    print("=" * 80)

    # List files from each split directory
    train_files = sorted(os.listdir(train_dir))
    val_files = sorted(os.listdir(val_dir)) if os.path.exists(val_dir) else []
    test_files = sorted(os.listdir(test_dir)) if os.path.exists(test_dir) else []

    print(f"Found {len(train_files)} files in {train_dir}")
    print(f"Found {len(val_files)} files in {val_dir}")
    print(f"Found {len(test_files)} files in {test_dir}")

    print(f"\nPatient splits:")
    print(f"  Train (patients {config.data.train_patients[0]}-{config.data.train_patients[1]}): {len(train_files)} files")
    print(f"  Val   (patients {config.data.val_patients[0]}-{config.data.val_patients[1]}): {len(val_files)} files")
    print(f"  Test  (patients {config.data.test_patients[0]}-{config.data.test_patients[1]}): {len(test_files)} files")

    # Apply patient filtering (cross-subject setup)
    train_files = filter_files_by_patient(
        train_files,
        min_patient=config.data.train_patients[0],
        max_patient=config.data.train_patients[1],
        exclude_patients=config.data.exclude_patients,
    )
    val_files = filter_files_by_patient(
        val_files,
        min_patient=config.data.val_patients[0],
        max_patient=config.data.val_patients[1],
        exclude_patients=config.data.exclude_patients,
    )
    test_files = filter_files_by_patient(
        test_files,
        min_patient=config.data.test_patients[0],
        max_patient=config.data.test_patients[1],
        exclude_patients=None,  # keep hold-out patient in test set
    )

    print("\nAfter patient filtering:")
    print(f"  Train: {len(train_files)} files (excluded: {config.data.exclude_patients})")
    print(f"  Val:   {len(val_files)} files (excluded: {config.data.exclude_patients})")
    print(f"  Test:  {len(test_files)} files (hold-out patients {config.data.test_patients[0]}-{config.data.test_patients[1]})")

    # Apply data fraction if specified
    if config.data.data_fraction < 1.0:
        original_train = len(train_files)
        original_val = len(val_files)
        original_test = len(test_files)

        train_files = train_files[:int(len(train_files) * config.data.data_fraction)]
        val_files = val_files[:int(len(val_files) * config.data.data_fraction)]
        test_files = test_files[:int(len(test_files) * config.data.data_fraction)]

        print(f"\nUsing {config.data.data_fraction*100:.1f}% of data:")
        print(f"  Train: {len(train_files)}/{original_train} files")
        print(f"  Val:   {len(val_files)}/{original_val} files")
        print(f"  Test:  {len(test_files)}/{original_test} files")

    # Validate files (with caching)
    if config.data.validate_files:
        print("\nValidating training files...")
        train_files = validate_pickle_files(
            train_dir,
            train_files,
            sample_validation=config.data.sample_validation,
            use_cache=True,
        )

        print("\nValidating validation files...")
        val_files = validate_pickle_files(
            val_dir,
            val_files,
            sample_validation=config.data.sample_validation,
            use_cache=True,
        )

        print("\nValidating test files...")
        test_files = validate_pickle_files(
            test_dir,
            test_files,
            sample_validation=config.data.sample_validation,
            use_cache=True,
        )

    # Final counts
    print(f"\nFinal dataset sizes:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")

    # Create datasets
    # Note: skip_resample can be enabled for faster loading (uses native 256Hz instead of 200Hz)
    skip_resample = getattr(config.data, 'skip_resample', False)
    train_dataset = CHBMITDataset(
        train_dir,
        train_files,
        config.data.sampling_rate,
        skip_resample=skip_resample,
        sample_length=config.data.sample_length,
    )
    val_dataset = CHBMITDataset(
        val_dir,
        val_files,
        config.data.sampling_rate,
        skip_resample=skip_resample,
        sample_length=config.data.sample_length,
    )
    test_dataset = CHBMITDataset(
        test_dir,
        test_files,
        config.data.sampling_rate,
        skip_resample=skip_resample,
        sample_length=config.data.sample_length,
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
        persistent_workers=True if config.data.num_workers > 0 else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
        persistent_workers=True if config.data.num_workers > 0 else False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=2 if config.data.num_workers > 0 else None,
        persistent_workers=True if config.data.num_workers > 0 else False,
    )

    print(f"\nDataLoader configuration:")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Num workers: {config.data.num_workers}")
    print(f"  Pin memory: {config.data.pin_memory}")
    print("=" * 80)
    print()

    return train_loader, val_loader, test_loader
