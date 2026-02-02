"""
Trainer setup utilities for PyTorch Lightning.

This module provides functions to configure the PyTorch Lightning Trainer
with appropriate callbacks, loggers, and hardware acceleration.
"""

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from config.base_config import ExperimentConfig


def setup_trainer(config: ExperimentConfig) -> pl.Trainer:
    """
    Setup PyTorch Lightning Trainer with callbacks and logger.

    Args:
        config: Experiment configuration

    Returns:
        Configured Trainer instance
    """
    # Setup TensorBoard logger
    logger = TensorBoardLogger(
        save_dir=config.training.log_dir,
        name="",  # Don't create subdirectory
        version=config.experiment_name,
    )

    # Model checkpoint callback - save best model based on validation metric
    checkpoint_callback = ModelCheckpoint(
        monitor=config.training.checkpoint_metric,
        dirpath=f"{config.training.checkpoint_dir}/{config.experiment_name}",
        filename="best-model-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=config.training.save_top_k,
        mode=config.training.checkpoint_mode,
        verbose=True,
    )

    # Last model checkpoint - always save the final model
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.training.checkpoint_dir}/{config.experiment_name}",
        filename="last-model-{epoch:02d}",
        save_last=True,
        verbose=True,
    )

    # Early stopping callback (optional)
    callbacks = [checkpoint_callback, last_checkpoint_callback]

    if config.training.use_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=config.training.early_stopping_metric,
            patience=config.training.early_stopping_patience,
            verbose=True,
            mode=config.training.early_stopping_mode,
            check_on_train_epoch_end=False,  # Check after validation, not training
        )
        callbacks.append(early_stop_callback)
        print(f"✓ Early stopping enabled (patience={config.training.early_stopping_patience})")
    else:
        print("✓ Early stopping disabled - will train for all epochs")

    # Determine hardware acceleration
    if config.training.accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = config.training.devices
            print(f"✓ Using {devices} GPU(s)")
        else:
            accelerator = "cpu"
            devices = config.training.devices
            print("⚠ No GPU available, using CPU")
    else:
        accelerator = config.training.accelerator
        devices = config.training.devices
        print(f"✓ Using {accelerator} with {devices} device(s)")

    # Mixed precision training for faster computation on modern GPUs
    precision = "16-mixed" if config.training.use_mixed_precision and accelerator == "gpu" else "32"
    if precision == "16-mixed":
        print("✓ Mixed precision training enabled (16-bit)")

    # Create trainer
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        precision=precision,
        benchmark=True,  # cudnn.benchmark for better performance
        enable_checkpointing=True,
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
    )

    return trainer
