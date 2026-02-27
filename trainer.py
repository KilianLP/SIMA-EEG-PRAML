from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pyhealth.metrics import binary_metrics_fn

#Config

@dataclass
class DataConfig:
    root_path: str = "./data"  # provide via --data-path

    # Data splits (patient IDs)
    train_patients: Tuple[int, int] = (2, 24)  # cross-subject: exclude chb01 from train/val
    val_patients: Tuple[int, int] = (2, 24)
    test_patients: Tuple[int, int] = (1, 1)    # hold-out chb01 for evaluation
    exclude_patients: List[int] = field(default_factory=lambda: [1])

    # Data loading parameters
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    data_fraction: float = 1.0

    # Data preprocessing
    sampling_rate: int = 256  # process2 outputs 256 Hz segments
    sample_length: float = 8.0  # process2 uses 8-second windows
    skip_resample: bool = False

    # File validation
    validate_files: bool = True
    sample_validation: bool = True


@dataclass
class ModelConfig:
    name: str = "EEGformer"
    in_channels: int = 4  # process1_bis/process2 keep 4 bipolar channels
    n_classes: int = 1

    # Signal processing parameters
    token_size: int = 200
    hop_length: int = 100

    # Transformer architecture parameters
    nhead: int = 4
    num_layers: int = 4
    emb_size: int = 256
    depth: int = 4
    dropout: float = 0.2

    pretrain_model_path: str = ""


@dataclass
class TrainingConfig:

    # Hyperparameters
    epochs: int = 50
    lr: float = 5e-5
    weight_decay: float = 1e-5

    # Class imbalance handling
    use_class_weights: bool = True
    pos_weight: Optional[float] = None

    # Early stopping
    use_early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_auroc"
    early_stopping_mode: str = "max"

    # Checkpointing
    checkpoint_metric: str = "val_auroc"
    checkpoint_mode: str = "max"
    save_top_k: int = 1

    # Directories
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"

    # Accelerator
    accelerator: str = "auto"
    devices: int = 1
    use_mixed_precision: bool = True


@dataclass
class Config:

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    experiment_name: str = "seizure_detection"
    seed: int = 12345


#Loss function
def binary_cross_entropy_loss(y_hat, y, pos_weight= None):
    """
    Stable Binary cross entropy loss with optional class weighting.
    https://medium.com/@sahilcarterr/why-nn-bcewithlogitsloss-numerically-stable-6a04f3052967
    """
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)

    loss = (-y * y_hat + torch.log(1 + torch.exp(-torch.abs(y_hat))) + torch.max(y_hat, torch.zeros_like(y_hat)))

    if pos_weight is not None:
        weight = torch.where(y > 0.5, pos_weight, 1.0)
        loss = loss * weight

    return loss.mean()

# Class weight


def compute_class_weights(dataloader, sample_fraction = 0.1):
    """
    Compute class weight for positive class based on class imbalance.

    """
    print(f"\nComputing class weights (sampling {sample_fraction*100:.0f}%)...")
    num_positives = 0
    num_negatives = 0

    total_batches = len(dataloader)
    sample_batches = max(1, int(total_batches * sample_fraction))

    import math
    step = max(1, math.ceil(total_batches / sample_batches))

    for batch_idx, (_, labels) in enumerate(dataloader):
        if batch_idx % step == 0:
            num_positives += labels.sum().item()
            num_negatives += (labels.size(0) - labels.sum().item())

        if batch_idx >= sample_batches * step:
            break

    total = num_positives + num_negatives
    pos_weight = num_negatives / (num_positives + 1e-8)

    print(f"  Positive samples: {int(num_positives):,} ({100*num_positives/total:.2f}%)")
    print(f"  Negative samples: {int(num_negatives):,} ({100*num_negatives/total:.2f}%)")
    print(f"  Computed pos_weight: {pos_weight:.2f}\n")

    return pos_weight

# Module lightning

class SeizureDetectionModule(pl.LightningModule):
    def __init__(self, model, config, pos_weight = None):
        super().__init__()
        self.model = model
        self.config = config
        self.pos_weight = pos_weight
        self.threshold = 0.5
        self.validation_step_outputs = []
        self.test_step_outputs = []

        if self.pos_weight is not None:
            print(f"Lightning module initialized with pos_weight={self.pos_weight:.2f}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = binary_cross_entropy_loss(logits, y, pos_weight=self.pos_weight)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self(X)
            loss = binary_cross_entropy_loss(logits, y, pos_weight=self.pos_weight)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()

        self.validation_step_outputs.append((probs, labels))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return probs, labels

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            logits = self(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()

        self.test_step_outputs.append((probs, labels))
        return probs, labels

    def _compute_epoch_metrics(self, outputs, stage, update_threshold = False,):
        all_predictions_prob = []
        all_gt = []
        for probs, labels in outputs:
            all_predictions_prob.extend(probs)
            all_gt.extend(labels)

        result_array = np.array(all_predictions_prob)
        gt = np.array(all_gt)

        if update_threshold and sum(gt) * (len(gt) - sum(gt)) != 0:
            self.threshold = float(np.sort(result_array)[-int(np.sum(gt))])

        if sum(gt) * (len(gt) - sum(gt)) != 0:
            result = binary_metrics_fn(
                gt,
                result_array,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }

        predictions = (result_array >= self.threshold).astype(int)
        true_positives = np.sum((predictions == 1) & (gt == 1))
        false_positives = np.sum((predictions == 1) & (gt == 0))
        true_negatives = np.sum((predictions == 0) & (gt == 0))
        false_negatives = np.sum((predictions == 0) & (gt == 1))

        total_samples = len(gt)
        total_hours = (total_samples * self.config.data.sample_length) / 3600.0
        fp_per_hour = false_positives / total_hours if total_hours > 0 else 0.0

        self.log(f"{stage}_acc", result["accuracy"], sync_dist=True)
        self.log(f"{stage}_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log(f"{stage}_pr_auc", result["pr_auc"], sync_dist=True)
        self.log(f"{stage}_auroc", result["roc_auc"], sync_dist=True)
        self.log(f"{stage}_false_positives", float(false_positives), sync_dist=True)
        self.log(f"{stage}_fp_per_hour", fp_per_hour, sync_dist=True)

        metrics = {
            **result,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "fp_per_hour": fp_per_hour,
            "total_hours": total_hours,
            "threshold": self.threshold,
        }

        return metrics

    def on_validation_epoch_end(self):
        metrics = self._compute_epoch_metrics(
            self.validation_step_outputs,
            stage="val",
            update_threshold=True,
        )

        print(
            f"Validation - Accuracy: {metrics['accuracy']:.4f}, "
            f"AUROC: {metrics['roc_auc']:.4f}, "
            f"FP: {metrics['false_positives']:.0f}, "
            f"FP/hour: {metrics['fp_per_hour']:.2f}"
        )

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        metrics = self._compute_epoch_metrics(
            self.test_step_outputs,
            stage="test",
            update_threshold=False,
        )

        print(f"\n{'='*60}")
        print(f"Test Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  PR AUC: {metrics['pr_auc']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  True Positives: {metrics['true_positives']:.0f}")
        print(f"  False Positives: {metrics['false_positives']:.0f}")
        print(f"  True Negatives: {metrics['true_negatives']:.0f}")
        print(f"  False Negatives: {metrics['false_negatives']:.0f}")
        print(f"  FP per hour: {metrics['fp_per_hour']:.2f}")
        print(f"{'='*60}\n")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config.training.lr, weight_decay=self.config.training.weight_decay,)
        return [optimizer]


# Trainer setup

def setup_trainer(config):

    # Create directories
    os.makedirs(config.training.log_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=config.training.log_dir,
        name="",
        version=config.experiment_name,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor=config.training.checkpoint_metric,
        dirpath=f"{config.training.checkpoint_dir}/{config.experiment_name}",
        filename="best-{epoch:02d}-{val_auroc:.4f}",
        save_top_k=config.training.save_top_k,
        mode=config.training.checkpoint_mode,
        verbose=True,
    )

    last_checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.training.checkpoint_dir}/{config.experiment_name}",
        filename="last-{epoch:02d}",
        save_last=True,
        verbose=True,
    )

    callbacks = [checkpoint_callback, last_checkpoint_callback]

    # Early stopping (optional)
    if config.training.use_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor=config.training.early_stopping_metric,
            patience=config.training.early_stopping_patience,
            verbose=True,
            mode=config.training.early_stopping_mode,
            check_on_train_epoch_end=False,
        )
        callbacks.append(early_stop_callback)
        print(f"Early stopping enabled (patience={config.training.early_stopping_patience})")

    if config.training.accelerator == "auto":
        if torch.cuda.is_available():
            accelerator = "gpu"
            devices = config.training.devices
            print(f"Using {devices} CUDA GPU(s)")
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
            print("Using Apple MPS GPU")
        else:
            accelerator = "cpu"
            devices = config.training.devices
            print("No GPU available, using CPU")
    else:
        accelerator = config.training.accelerator
        devices = config.training.devices

    # Mixed precision
    precision = "16-mixed" if config.training.use_mixed_precision and accelerator in ("gpu", "mps") else "32"
    if precision == "16-mixed":
        print("Mixed precision training enabled (16-bit)")

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        precision=precision,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=config.training.epochs,
        callbacks=callbacks,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
    )

    return trainer
