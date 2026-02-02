#!/usr/bin/env python3
"""
CHB-MIT Seizure Detection Training Script

This script trains transformer models for seizure detection on the CHB-MIT dataset.
It provides a clean interface for experimenting with different architectures.

Usage Examples:
    # Train CNNTransformer (default)
    python experiments/train_chbmit.py

    # Train BIOT model
    python experiments/train_chbmit.py --model BIOT

    # Quick test with 1% of data
    python experiments/train_chbmit.py --data-fraction 0.01 --epochs 2

    # Custom hyperparameters
    python experiments/train_chbmit.py --model STTransformer --lr 0.0001 --batch-size 64

    # Override data path
    python experiments/train_chbmit.py --data-path /custom/path/to/data
"""

import argparse
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config_for_model
from models.factory import model_registry
from data import prepare_chbmit_dataloaders, compute_class_weights_from_files
from training import SeizureDetectionModule, setup_trainer, compute_class_weights


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train transformer models for seizure detection on CHB-MIT dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="CNNTransformer",
        choices=model_registry.list_models(),
        help="Model architecture to use",
    )

    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to processed CHB-MIT data (overrides default)",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        help="Fraction of data to use (0-1), useful for quick testing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of data loading workers",
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay for optimizer",
    )

    # Model hyperparameters
    parser.add_argument(
        "--nhead",
        type=int,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--emb-size",
        type=int,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate",
    )
    parser.add_argument(
        "--pretrain-path",
        type=str,
        help="Path to pretrained model (for BIOT)",
    )

    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip test phase after training",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip file validation (faster startup, use only if data is known to be good)",
    )
    parser.add_argument(
        "--skip-resample",
        action="store_true",
        help="Skip resampling, use native 256Hz (faster data loading, ~30%% speedup)",
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training (16-bit, enabled by default on GPU)",
    )

    # Class weighting
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting in loss (not recommended for imbalanced data)",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        help="Manual positive class weight (overrides auto-computation)",
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Load default configuration for selected model
    print(f"Loading configuration for {args.model}...")
    config = get_config_for_model(args.model)

    # Override configuration with command-line arguments
    if args.data_path:
        config.data.root_path = args.data_path
    if args.data_fraction:
        config.data.data_fraction = args.data_fraction
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.no_validation:
        config.data.validate_files = False
    if args.skip_resample:
        config.data.skip_resample = True
        # Update sampling rate to native 256Hz when skipping resample
        config.data.sampling_rate = 256

    if args.epochs:
        config.training.epochs = args.epochs
    if args.lr:
        config.training.lr = args.lr
    if args.weight_decay:
        config.training.weight_decay = args.weight_decay

    if args.nhead:
        config.model.nhead = args.nhead
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.emb_size:
        config.model.emb_size = args.emb_size
    if args.dropout:
        config.model.dropout = args.dropout
    if args.pretrain_path:
        config.model.pretrain_model_path = args.pretrain_path

    if args.no_class_weights:
        config.training.use_class_weights = False
    if args.pos_weight is not None:
        config.training.auto_compute_weights = False
        config.training.pos_weight = args.pos_weight
    if args.no_mixed_precision:
        config.training.use_mixed_precision = False

    config.seed = args.seed

    # Print configuration summary
    print(config.summary())

    # Set random seed for reproducibility
    pl.seed_everything(config.seed, workers=True)
    print(f"Set random seed to {config.seed}")

    # Prepare data loaders
    print("\n" + "=" * 80)
    print("STEP 1: Preparing Data Loaders")
    print("=" * 80)
    train_loader, val_loader, test_loader = prepare_chbmit_dataloaders(config)

    # Build model
    print("=" * 80)
    print("STEP 2: Building Model")
    print("=" * 80)
    print(f"Creating {config.model.name} model...")

    # For EEGformer, update sequence length based on actual sampling rate
    if config.model.name == "EEGformer":
        actual_sr = 256 if config.data.skip_resample else config.data.sampling_rate
        seq_length = int(10 * actual_sr)  # 10 seconds of data
        # Temporarily add to config for model building
        config.model.__dict__['seq_length'] = seq_length
        print(f"  Sequence length: {seq_length} samples ({actual_sr}Hz × 10s)")

    model = model_registry.build(config.model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model created successfully")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Compute class weights if needed
    pos_weight = None
    if config.training.use_class_weights:
        print("\n" + "=" * 80)
        print("STEP 2.5: Computing Class Weights")
        print("=" * 80)
        if config.training.auto_compute_weights:
            # Use fast file-based computation (reads only labels from pickles)
            # This is much faster than iterating through the dataloader
            import os
            train_dir = os.path.join(config.data.root_path, "train")

            # Get list of training files
            from data import filter_files_by_patient
            all_files = os.listdir(train_dir)
            train_files = filter_files_by_patient(
                all_files,
                min_patient=config.data.train_patients[0],
                max_patient=config.data.train_patients[1]
            )

            # Apply data fraction if needed
            if config.data.data_fraction < 1.0:
                train_files = train_files[:int(len(train_files) * config.data.data_fraction)]

            pos_weight = compute_class_weights_from_files(train_dir, train_files)
        else:
            pos_weight = config.training.pos_weight
            if pos_weight is not None:
                print(f"Using manual pos_weight: {pos_weight:.2f}\n")
            else:
                print("Warning: use_class_weights=True but no weight provided. Using unweighted loss.\n")
    else:
        print("\n⚠ Class weighting disabled - model may struggle with imbalanced data")

    # Create Lightning module
    print("\n" + "=" * 80)
    print("STEP 3: Creating Lightning Module")
    print("=" * 80)
    print("Wrapping model in PyTorch Lightning module...")
    lightning_module = SeizureDetectionModule(model, config, pos_weight=pos_weight)
    print("✓ Lightning module created")

    # Setup trainer
    print("\n" + "=" * 80)
    print("STEP 4: Setting Up Trainer")
    print("=" * 80)
    trainer = setup_trainer(config)
    print(f"✓ Trainer configured")
    print(f"  Max epochs: {config.training.epochs}")
    print(f"  Early stopping patience: {config.training.early_stopping_patience}")
    print(f"  Checkpoint directory: {config.training.checkpoint_dir}/{config.experiment_name}")

    # Train
    print("\n" + "=" * 80)
    print("STEP 5: Training")
    print("=" * 80)
    print(f"Starting training for {config.training.epochs} epochs...")
    print(f"TensorBoard logs: {config.training.log_dir}/{config.experiment_name}")
    print("=" * 80 + "\n")

    trainer.fit(lightning_module, train_loader, val_loader)

    print("\n✓ Training complete!")

    # Test
    if not args.skip_test:
        print("\n" + "=" * 80)
        print("STEP 6: Testing")
        print("=" * 80)
        print("Evaluating best model on test set...")
        trainer.test(lightning_module, test_loader, ckpt_path="best")
    else:
        print("\n⊘ Skipping test phase (--skip-test flag set)")

    # Print final summary
    print("\n" + "=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print(f"Experiment name: {config.experiment_name}")
    print(f"Checkpoints saved to: {config.training.checkpoint_dir}/{config.experiment_name}")
    print(f"TensorBoard logs: {config.training.log_dir}/{config.experiment_name}")
    print("\nTo view training curves:")
    print(f"  tensorboard --logdir {config.training.log_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
