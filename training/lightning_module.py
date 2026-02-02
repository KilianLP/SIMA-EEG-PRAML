"""
PyTorch Lightning module for seizure detection.

This module provides a refactored Lightning module that eliminates
duplicate code between validation and test epoch end methods.
"""

from typing import Dict, Tuple, List, Optional

import torch
import pytorch_lightning as pl
import numpy as np
from pyhealth.metrics import binary_metrics_fn

from config.base_config import ExperimentConfig


def compute_class_weights(dataloader: torch.utils.data.DataLoader, sample_fraction: float = 0.1) -> float:
    """
    Compute class weight for positive class based on class imbalance.

    Uses the formula: pos_weight = num_negatives / num_positives
    This makes the positive class equally important as the negative class.

    Args:
        dataloader: Training dataloader to compute statistics from
        sample_fraction: Fraction of data to sample for estimation (default: 0.1 = 10%)
                        Use 1.0 for exact count (slower)

    Returns:
        Positive class weight as a float
    """
    print(f"\nComputing class weights from training data (sampling {sample_fraction*100:.0f}%)...")
    num_positives = 0
    num_negatives = 0

    # Calculate how many batches to sample
    total_batches = len(dataloader)
    sample_batches = max(1, int(total_batches * sample_fraction))

    print(f"  Sampling {sample_batches}/{total_batches} batches for class balance estimation...")

    # Sample batches uniformly across the dataset
    import math
    step = max(1, math.ceil(total_batches / sample_batches))

    for batch_idx, (_, labels) in enumerate(dataloader):
        if batch_idx % step == 0:
            num_positives += labels.sum().item()
            num_negatives += (labels.size(0) - labels.sum().item())

        # Stop after we've collected enough samples
        if batch_idx >= sample_batches * step:
            break

    total = num_positives + num_negatives
    pos_weight = num_negatives / (num_positives + 1e-8)  # Avoid division by zero

    print(f"  Sampled samples: {total:,}")
    print(f"  Positive samples (seizure): {int(num_positives):,} ({100*num_positives/total:.2f}%)")
    print(f"  Negative samples (no seizure): {int(num_negatives):,} ({100*num_negatives/total:.2f}%)")
    print(f"  Computed pos_weight: {pos_weight:.2f}")
    print(f"  This means positive samples will be weighted {pos_weight:.2f}x more in the loss\n")

    return pos_weight


def binary_cross_entropy_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    pos_weight: Optional[float] = None
) -> torch.Tensor:
    """
    Binary cross entropy loss (numerically stable implementation) with optional class weighting.

    Args:
        y_hat: Predicted logits of shape (N,) or (N, 1)
        y: True labels of shape (N,) or (N, 1)
        pos_weight: Optional weight for positive class to handle imbalance

    Returns:
        Mean BCE loss
    """
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)

    # Standard BCE: -y*log(sigmoid(y_hat)) - (1-y)*log(1-sigmoid(y_hat))
    # Numerically stable form
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )

    # Apply class weighting if specified
    if pos_weight is not None:
        # Weight positive samples more heavily
        weight = torch.where(y > 0.5, pos_weight, 1.0)
        loss = loss * weight

    return loss.mean()


class SeizureDetectionModule(pl.LightningModule):
    """
    PyTorch Lightning module for binary seizure detection.

    This module wraps any PyTorch model for seizure detection training
    and evaluation. It computes relevant metrics including accuracy,
    balanced accuracy, PR-AUC, ROC-AUC, and false positives per hour.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: ExperimentConfig,
        pos_weight: Optional[float] = None
    ):
        """
        Initialize the Lightning module.

        Args:
            model: PyTorch model for seizure detection
            config: Experiment configuration
            pos_weight: Optional positive class weight for handling class imbalance
        """
        super().__init__()
        self.model = model
        self.config = config
        self.pos_weight = pos_weight
        self.threshold = 0.5  # Initial threshold, updated during validation
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Log the configuration
        if self.pos_weight is not None:
            print(f"Lightning module initialized with pos_weight={self.pos_weight:.2f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Tuple of (signals, labels)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        X, y = batch
        logits = self(X)
        loss = binary_cross_entropy_loss(logits, y, pos_weight=self.pos_weight)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Validation step.

        Args:
            batch: Tuple of (signals, labels)
            batch_idx: Batch index

        Returns:
            Tuple of (predictions, ground_truth)
        """
        X, y = batch
        with torch.no_grad():
            logits = self(X)
            loss = binary_cross_entropy_loss(logits, y, pos_weight=self.pos_weight)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()

        self.validation_step_outputs.append((probs, labels))
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return probs, labels

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Test step.

        Args:
            batch: Tuple of (signals, labels)
            batch_idx: Batch index

        Returns:
            Tuple of (predictions, ground_truth)
        """
        X, y = batch
        with torch.no_grad():
            logits = self(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()

        self.test_step_outputs.append((probs, labels))
        return probs, labels

    def _compute_epoch_metrics(
        self,
        outputs: List[Tuple[np.ndarray, np.ndarray]],
        stage: str,
        update_threshold: bool = False,
    ) -> Dict[str, float]:
        """
        Compute metrics at epoch end (shared logic for validation and test).

        Args:
            outputs: List of (predictions, ground_truth) tuples from all steps
            stage: Stage name ("val" or "test") for logging
            update_threshold: Whether to update the threshold based on validation set

        Returns:
            Dictionary of computed metrics
        """
        # Aggregate all predictions and ground truth
        all_predictions_prob = []
        all_gt = []
        for probs, labels in outputs:
            all_predictions_prob.extend(probs)
            all_gt.extend(labels)

        result_array = np.array(all_predictions_prob)
        gt = np.array(all_gt)

        # Update threshold on validation set
        if update_threshold and sum(gt) * (len(gt) - sum(gt)) != 0:
            # Set threshold to the Nth highest prediction where N = number of positives
            self.threshold = float(np.sort(result_array)[-int(np.sum(gt))])

        # Compute metrics
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

        # Calculate confusion matrix elements
        predictions = (result_array >= self.threshold).astype(int)
        true_positives = np.sum((predictions == 1) & (gt == 1))
        false_positives = np.sum((predictions == 1) & (gt == 0))
        true_negatives = np.sum((predictions == 0) & (gt == 0))
        false_negatives = np.sum((predictions == 0) & (gt == 1))

        # Calculate false positives per hour
        total_samples = len(gt)
        total_hours = (total_samples * self.config.data.sample_length) / 3600.0
        fp_per_hour = false_positives / total_hours if total_hours > 0 else 0.0

        # Log metrics
        self.log(f"{stage}_acc", result["accuracy"], sync_dist=True)
        self.log(f"{stage}_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log(f"{stage}_pr_auc", result["pr_auc"], sync_dist=True)
        self.log(f"{stage}_auroc", result["roc_auc"], sync_dist=True)
        self.log(f"{stage}_false_positives", float(false_positives), sync_dist=True)
        self.log(f"{stage}_fp_per_hour", fp_per_hour, sync_dist=True)

        # Return extended metrics
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
        """Validation epoch end - compute metrics and update threshold."""
        metrics = self._compute_epoch_metrics(
            self.validation_step_outputs,
            stage="val",
            update_threshold=True,
        )

        print(
            f"Validation - Accuracy: {metrics['accuracy']:.4f}, "
            f"AUROC: {metrics['roc_auc']:.4f}, "
            f"FP: {metrics['false_positives']:.0f}, "
            f"FP/hour: {metrics['fp_per_hour']:.2f}, "
            f"Threshold: {metrics['threshold']:.4f}"
        )

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Test epoch end - compute final metrics and print summary."""
        metrics = self._compute_epoch_metrics(
            self.test_step_outputs,
            stage="test",
            update_threshold=False,  # Use threshold from validation
        )

        # Print detailed test results
        print(f"\n{'='*80}")
        print(f"Test Results Summary:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  PR AUC: {metrics['pr_auc']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  True Positives: {metrics['true_positives']:.0f}")
        print(f"  False Positives: {metrics['false_positives']:.0f}")
        print(f"  True Negatives: {metrics['true_negatives']:.0f}")
        print(f"  False Negatives: {metrics['false_negatives']:.0f}")
        print(f"  Total hours of data: {metrics['total_hours']:.2f}")
        print(f"  False Positives per hour: {metrics['fp_per_hour']:.2f}")
        print(f"  Threshold: {metrics['threshold']:.4f}")
        print(f"{'='*80}\n")

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
        )
        return [optimizer]
