#!/usr/bin/env python3
"""
Usage Examples:
    # Train EEGformer (default)
    python main.py --data-path /path/to/data --model EEGformer

    # Quick test with 1% of data
    python main.py --data-path /path/to/data --model EEGformer --data-fraction 0.01 --epochs 2

    # Test a trained model
    python main.py --mode test --model EEGformer --checkpoint ./checkpoints/best.ckpt --data-path /path/to/data
"""

import argparse
import torch
import pytorch_lightning as pl

# Model imports
from model.cnn_transformer import CNNTransformer
from model.eegformer import EEGformer

# Training imports
from trainer import (
    Config, DataConfig, ModelConfig, TrainingConfig,
    SeizureDetectionModule, setup_trainer, compute_class_weights
)

# Data imports
from data import prepare_chbmit_dataloaders, compute_class_weights_from_files, filter_files_by_patient


# Model-specific default hyperparameters
MODEL_DEFAULTS = {
    "CNNTransformer": {"batch_size": 128, "lr": 1e-3, "nhead": 4, "emb_size": 256, "num_layers": 4, "dropout": 0.2, "depth": 4},
    "EEGformer": {"batch_size": 16, "lr": 5e-5, "nhead": 8, "emb_size": 32, "num_layers": 1, "dropout": 0.1, "depth": 4},
}

AVAILABLE_MODELS = list(MODEL_DEFAULTS.keys())


def create_model(model_name, config):
    """
    Create model based on name.
    """
    mc = config.model

    if model_name == "CNNTransformer":
        return CNNTransformer(
            in_channels=mc.in_channels,
            n_classes=mc.n_classes,
            fft=mc.token_size,
            steps=mc.hop_length // 5,
            dropout=mc.dropout,
            nhead=mc.nhead,
            emb_size=mc.emb_size,
            n_segments=5,
        )


    elif model_name == "EEGformer":
        # Compute sequence length based on sampling rate
        actual_sr = config.data.sampling_rate
        seq_length = int(config.data.sample_length * actual_sr)
        return EEGformer(
            in_channels=mc.in_channels,
            n_classes=mc.n_classes,
            seq_length=seq_length,
            embed_dim=mc.emb_size,
            num_heads=mc.nhead,
            num_layers=mc.num_layers,
            dim_feedforward=mc.emb_size * 4,
            kernel_size=5,
            dropout=mc.dropout,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {AVAILABLE_MODELS}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train or test seizure detection models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Mode: train or test",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="EEGformer",
        choices=AVAILABLE_MODELS,
        help="Model architecture",
    )

    # Checkpoint (required for test mode)
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint for testing",
    )

    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        # required=True,
        default="data/clean_segments", ###TO ADAPT###
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size (default depends on model)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of data to use (0-1)",
    )
    parser.add_argument(
        "--skip-resample",
        action="store_true",
        help="Skip resampling, use native 256Hz",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip file validation",
    )
    parser.add_argument(
        "--exclude-patients",
        nargs="+",
        type=int,
        default=None,
        help="Patient IDs to exclude from train/val (e.g., 1 for chb01)",
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate (default depends on model)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting",
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        help="Manual positive class weight",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    # Model hyperparameters
    parser.add_argument("--nhead", type=int, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, help="Number of transformer layers")
    parser.add_argument("--emb-size", type=int, help="Embedding dimension")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--pretrain-path", type=str, help="Path to pretrained weights (BIOT)")

    # Hardware
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Hardware accelerator",
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision")

    # Misc
    parser.add_argument("--seed", type=int, default=12345, help="Random seed")
    parser.add_argument("--experiment-name", type=str, help="Experiment name for logging")
    parser.add_argument("--skip-test", action="store_true", help="Skip test after training")

    return parser.parse_args()


def build_config(args):
    """Build configuration from command-line arguments."""
    # Get model-specific defaults
    defaults = MODEL_DEFAULTS[args.model]

    # Data config
    data_config = DataConfig(
        root_path=args.data_path,
        batch_size=args.batch_size or defaults["batch_size"],
        num_workers=args.num_workers,
        data_fraction=args.data_fraction,
        skip_resample=args.skip_resample,
        sampling_rate=256,  # process2 outputs 256 Hz
        validate_files=not args.no_validation,
        exclude_patients=args.exclude_patients or DataConfig().exclude_patients,
    )

    # Model config
    model_config = ModelConfig(
        name=args.model,
        nhead=args.nhead or defaults["nhead"],
        num_layers=args.num_layers or defaults["num_layers"],
        emb_size=args.emb_size or defaults["emb_size"],
        depth=args.num_layers or defaults["depth"],  # Some models use depth instead
        dropout=args.dropout if args.dropout is not None else defaults["dropout"],
        pretrain_model_path=args.pretrain_path or "",
    )

    # Training config
    training_config = TrainingConfig(
        epochs=args.epochs,
        lr=args.lr or defaults["lr"],
        weight_decay=args.weight_decay,
        use_class_weights=not args.no_class_weights,
        pos_weight=args.pos_weight,
        use_early_stopping=args.early_stopping,
        early_stopping_patience=args.patience,
        accelerator=args.accelerator,
        devices=args.devices,
        use_mixed_precision=not args.no_mixed_precision,
    )

    # Experiment name
    exp_name = args.experiment_name or f"{args.model}_lr{training_config.lr}_bs{data_config.batch_size}"

    return Config(
        data=data_config,
        model=model_config,
        training=training_config,
        experiment_name=exp_name,
        seed=args.seed,
    )


def main():
    """Main entry point."""
    args = parse_args()

    # Validate test mode requirements
    if args.mode == "test" and not args.checkpoint:
        raise ValueError("--checkpoint is required for test mode")

    # Build configuration
    config = build_config(args)

    # Print configuration
    print("=" * 60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Model: {config.model.name}")
    print(f"Data path: {config.data.root_path}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.lr}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Seed: {config.seed}")
    print("=" * 60)

    # Set seed
    pl.seed_everything(config.seed, workers=True)

    # Prepare data loaders
    print("\nPreparing data loaders...")
    train_loader, val_loader, test_loader = prepare_chbmit_dataloaders(config)

    if args.mode == "train":
        # Create model
        print(f"\nCreating {config.model.name} model...")
        model = create_model(args.model, config)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Compute class weights
        pos_weight = None
        if config.training.use_class_weights:
            if config.training.pos_weight is not None:
                pos_weight = config.training.pos_weight
                print(f"\nUsing manual pos_weight: {pos_weight:.2f}")
            else:
                # Fast file-based computation
                import os
                train_dir = os.path.join(config.data.root_path, "train")
                all_files = os.listdir(train_dir)
                train_files = filter_files_by_patient(
                    all_files,
                    min_patient=config.data.train_patients[0],
                    max_patient=config.data.train_patients[1],
                    exclude_patients=config.data.exclude_patients,
                )
                if config.data.data_fraction < 1.0:
                    train_files = train_files[:int(len(train_files) * config.data.data_fraction)]
                pos_weight = compute_class_weights_from_files(train_dir, train_files)

        # Create Lightning module
        lightning_module = SeizureDetectionModule(model, config, pos_weight)

        # Setup trainer
        trainer = setup_trainer(config)

        # Train
        print(f"\nStarting training for {config.training.epochs} epochs...")
        trainer.fit(lightning_module, train_loader, val_loader)
        print("\nTraining complete!")

        # Test
        if not args.skip_test:
            print("\nEvaluating on test set...")
            trainer.test(lightning_module, test_loader, ckpt_path="best")

        # Print summary
        print(f"\nCheckpoints saved to: {config.training.checkpoint_dir}/{config.experiment_name}")
        print(f"Logs saved to: {config.training.log_dir}/{config.experiment_name}")

    elif args.mode == "test":
        # Create model architecture
        print(f"\nCreating {config.model.name} model...")
        model = create_model(args.model, config)

        # Load from checkpoint
        print(f"Loading checkpoint: {args.checkpoint}")
        lightning_module = SeizureDetectionModule.load_from_checkpoint(
            args.checkpoint,
            model=model,
            config=config,
        )

        # Setup trainer and test
        trainer = setup_trainer(config)
        trainer.test(lightning_module, test_loader)


if __name__ == "__main__":
    main()
