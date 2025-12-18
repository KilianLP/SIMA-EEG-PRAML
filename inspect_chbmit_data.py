import os
import torch
import numpy as np
from utils import CHBMITLoader


def inspect_dataset():
    """Inspect the CHB-MIT dataset to understand data dimensions"""
    
    root = "/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments"
    sampling_rate = 200
    
    # Get some train files
    train_files = os.listdir(os.path.join(root, "train"))
    
    # Filter for first patient only
    filtered_files = []
    for f in train_files[:100]:  # Check first 100 files
        if f.startswith('chb01'):
            filtered_files.append(f)
    
    print(f"Found {len(filtered_files)} files for chb01")
    print(f"Sample filenames: {filtered_files[:5]}")
    
    # Create a small dataloader
    dataset = CHBMITLoader(
        os.path.join(root, "train"), 
        filtered_files[:10],  # Just 10 files
        sampling_rate
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Load a few samples
    for i in range(min(5, len(dataset))):
        X, y = dataset[i]
        print(f"\nSample {i}:")
        print(f"  X shape: {X.shape}")
        print(f"  X dtype: {X.dtype}")
        print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
        print(f"  y value: {y}")
        print(f"  y type: {type(y)}")
        
        # Check if it's tensor or numpy
        if isinstance(X, torch.Tensor):
            print(f"  X is torch.Tensor")
        elif isinstance(X, np.ndarray):
            print(f"  X is numpy.ndarray")
    
    # Try loading a batch
    print("\n" + "="*80)
    print("Testing DataLoader:")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
    )
    
    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  X_batch shape: {X_batch.shape}")
        print(f"  y_batch shape: {y_batch.shape}")
        print(f"  X_batch dtype: {X_batch.dtype}")
        print(f"  y_batch dtype: {y_batch.dtype}")
        
        if batch_idx >= 2:  # Just check first 3 batches
            break
    
    print("\n" + "="*80)
    print("Expected input for CNNTransformer:")
    print("  - Shape should be: (batch_size, n_channels, time_steps)")
    print("  - n_channels should be 23 for CHB-MIT")
    print(f"  - time_steps should be: {sampling_rate * 10} = {sampling_rate * 10} for 10 second segments")
    
    # Verify expected dimensions
    expected_shape = (23, sampling_rate * 10)
    print(f"\nExpected single sample shape: {expected_shape}")
    
    if len(dataset) > 0:
        X, y = dataset[0]
        if hasattr(X, 'shape'):
            actual_shape = X.shape
            print(f"Actual single sample shape: {actual_shape}")
            
            if actual_shape != expected_shape:
                print("\n⚠️  WARNING: Shape mismatch detected!")
                print(f"Expected: {expected_shape}")
                print(f"Got: {actual_shape}")
            else:
                print("\n✓ Shape matches expected dimensions")


if __name__ == "__main__":
    print("="*80)
    print("CHB-MIT Dataset Inspector")
    print("="*80)
    inspect_dataset()
