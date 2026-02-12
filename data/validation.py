
import os
import pickle
import hashlib
import time
from typing import List, Optional, Set
from pathlib import Path

import numpy as np


def get_cache_path(root_dir):
    """
    Get path to validation cache file.
    """
    # Create a hash of the directory path for the cache filename
    dir_hash = hashlib.md5(root_dir.encode()).hexdigest()[:8]
    cache_dir = Path(root_dir).parent
    return cache_dir / f".validation_cache_{dir_hash}.pkl"


def load_validation_cache(root_dir):
    """
    Load cached validation results.
    """
    cache_path = get_cache_path(root_dir)
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)

        # Check if cache is still valid (data directory hasn't been modified)
        root_mtime = os.path.getmtime(root_dir)
        if cache['timestamp'] > root_mtime:
            print(f"✓ Loaded validation cache from {cache_path}")
            print(f"  Found {len(cache['valid_files'])} valid files")
            return set(cache['valid_files'])
        else:
            print(f"⚠ Validation cache is outdated, will re-validate")
            return None

    except Exception as e:
        print(f"⚠ Failed to load validation cache: {e}")
        return None


def save_validation_cache(root_dir, valid_files):
    """
    Save validation results to cache.

    """
    cache_path = get_cache_path(root_dir)
    cache = {
        'valid_files': valid_files,
        'timestamp': time.time(),
        'root_dir': root_dir,
        'num_files': len(valid_files),
    }

    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"✓ Saved validation cache to {cache_path}")
    except Exception as e:
        print(f"⚠ Failed to save validation cache: {e}")


def validate_pickle_files(root_dir, files, sample_validation=True, sample_rate=0.01, use_cache=True,):
    """
    Validate pickle files and remove corrupted ones.
    """
    # Try to load from cache first
    if use_cache:
        cached_valid = load_validation_cache(root_dir)
        if cached_valid is not None:
            # Filter files to only include those in cache
            valid_files = [f for f in files if f in cached_valid]
            if len(valid_files) > 0:
                return valid_files
            else:
                print("⚠ Cache doesn't contain any of the requested files, re-validating")

    # Sample validation for quick startup
    if sample_validation and len(files) > 100:
        sample_size = max(100, int(len(files) * sample_rate))
        sample_indices = np.random.choice(len(files), min(sample_size, len(files)), replace=False)
        sample_files = [files[i] for i in sample_indices]

        print(f"Quick validation: checking {len(sample_files)} out of {len(files)} files...")
        corrupted_in_sample = 0
        error_examples = []

        for i, filename in enumerate(sample_files):
            filepath = os.path.join(root_dir, filename)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                    # Accept both 'X'/'y' and 'data'/'label' key naming conventions
                    has_data = ('data' in data and 'label' in data) or ('X' in data and 'y' in data)
                    if not has_data:
                        corrupted_in_sample += 1
                        if len(error_examples) < 3:
                            error_examples.append(f"Missing keys in {filename}: keys={list(data.keys())}")
            except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
                corrupted_in_sample += 1
                if len(error_examples) < 3:
                    error_examples.append(f"{type(e).__name__} in {filename}: {str(e)[:100]}")
            except Exception as e:
                corrupted_in_sample += 1
                if len(error_examples) < 3:
                    error_examples.append(f"{type(e).__name__} in {filename}: {str(e)[:100]}")

            if i < 5:
                status = '✗ FAILED' if (i < corrupted_in_sample) else '✓ OK'
                print(f"  Checking file {i+1}: {filename} - {status}")

        corruption_rate = corrupted_in_sample / len(sample_files)

        if corruption_rate > 0.5:
            print(f"\n⚠ Examples of validation failures:")
            for example in error_examples:
                print(f"  {example}")

            raise RuntimeError(
                f"❌ CRITICAL: {corruption_rate*100:.1f}% of sampled files are corrupted!\n"
                f"   This suggests a serious data corruption issue.\n"
                f"   Please check your data directory: {root_dir}\n"
                f"   You may need to re-process or re-download the dataset."
            )

        if corruption_rate > 0:
            print(f"⚠ Found {corruption_rate*100:.1f}% corruption in sample")
            print(f"  Performing full validation of all {len(files)} files...")
        else:
            print(f"✓ Sample validation passed, assuming all files are valid")
            # Save to cache
            if use_cache:
                save_validation_cache(root_dir, files)
            return files

    # Full validation
    valid_files = []
    corrupted_count = 0
    error_types = {}

    print(f"Validating {len(files)} files...")
    for i, filename in enumerate(files):
        if i % 10000 == 0 and i > 0:
            print(f"  Progress: {i}/{len(files)} files checked...")

        filepath = os.path.join(root_dir, filename)
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                # Accept both 'X'/'y' and 'data'/'label' key naming conventions
                has_data = ('data' in data and 'label' in data) or ('X' in data and 'y' in data)
                if has_data:
                    valid_files.append(filename)
                else:
                    corrupted_count += 1
                    error_key = f"missing_keys:{list(data.keys())}"
                    error_types[error_key] = error_types.get(error_key, 0) + 1
        except (EOFError, pickle.UnpicklingError, FileNotFoundError) as e:
            corrupted_count += 1
            error_key = type(e).__name__
            error_types[error_key] = error_types.get(error_key, 0) + 1
        except Exception as e:
            corrupted_count += 1
            error_key = type(e).__name__
            error_types[error_key] = error_types.get(error_key, 0) + 1

    if corrupted_count > 0:
        print(f"⚠ Found {corrupted_count} corrupted/invalid files, skipping them")
        print(f"  Error types: {error_types}")

    # Check if we lost too many files
    corruption_rate = corrupted_count / len(files) if len(files) > 0 else 0
    if corruption_rate > 0.5:
        raise RuntimeError(
            f"❌ CRITICAL: {corruption_rate*100:.1f}% ({corrupted_count}/{len(files)}) of files are corrupted!\n"
            f"   Directory: {root_dir}\n"
            f"   Error breakdown: {error_types}\n"
            f"   This indicates a serious data corruption issue.\n"
            f"   Please check your data and consider re-processing the dataset."
        )

    if len(valid_files) == 0:
        raise RuntimeError(
            f"❌ CRITICAL: All {len(files)} files in {root_dir} are corrupted!\n"
            f"   Error breakdown: {error_types}\n"
            f"   No valid files remain for training.\n"
            f"   Please check your data directory and re-process the dataset."
        )

    print(f"✓ {len(valid_files)} valid files remain")

    # Save to cache
    if use_cache:
        save_validation_cache(root_dir, valid_files)

    return valid_files
