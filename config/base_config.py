"""
Configuration dataclasses for BIOT seizure detection experiments.

This module defines the configuration structure for data loading, model architecture,
and training hyperparameters.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
import os


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Data paths
    root_path: str = "/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments"

    # Patient splits (min_patient, max_patient) inclusive
    train_patients: Tuple[int, int] = (1, 5)
    val_patients: Tuple[int, int] = (6, 6)
    test_patients: Tuple[int, int] = (7, 8)

    # Data loading parameters
    batch_size: int = 16
    num_workers: int = 4
    pin_memory: bool = True
    data_fraction: float = 1.0  # Fraction of data to use (0-1)

    # Data preprocessing
    sampling_rate: int = 200  # Hz
    sample_length: float = 10.0  # seconds
    skip_resample: bool = False  # If True, use native 256Hz (faster!)

    # File validation
    validate_files: bool = True
    sample_validation: bool = True  # Quick validation (sample 1% of files)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.data_fraction <= 1.0:
            raise ValueError(f"data_fraction must be in (0, 1], got {self.data_fraction}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.sampling_rate < 1:
            raise ValueError(f"sampling_rate must be positive, got {self.sampling_rate}")


@dataclass
class ModelConfig:
    """Configuration for model architecture."""

    # Model selection
    name: str = "EEGformer"  # "CNNTransformer", "BIOT", "STTransformer"

    # Input/output dimensions
    in_channels: int = 16
    n_classes: int = 1  # Binary classification

    # Signal processing parameters
    token_size: int = 200  # n_fft for STFT
    hop_length: int = 100  # STFT hop length

    # Transformer architecture parameters
    nhead: int = 4  # Number of attention heads
    num_layers: int = 4  # Number of transformer layers
    emb_size: int = 256  # Embedding dimension
    depth: int = 4  # Depth for BIOT/STTransformer
    dropout: float = 0.2

    # Pretrained model (optional)
    pretrain_model_path: str = ""

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_models = ["CNNTransformer", "BIOT", "STTransformer", "EEGformer"]
        if self.name not in valid_models:
            raise ValueError(f"model name must be one of {valid_models}, got {self.name}")
        if self.nhead < 1:
            raise ValueError(f"nhead must be positive, got {self.nhead}")
        if self.emb_size % self.nhead != 0:
            raise ValueError(f"emb_size ({self.emb_size}) must be divisible by nhead ({self.nhead})")


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    # Optimization
    epochs: int = 50
    lr: float = 5e-5
    weight_decay: float = 1e-5

    # Class imbalance handling
    use_class_weights: bool = True  # Use class weighting in loss
    auto_compute_weights: bool = True  # Auto-compute from training data
    pos_weight: Optional[float] = None  # Manual positive class weight (if auto_compute_weights=False)

    # Early stopping
    use_early_stopping: bool = False  # Enable/disable early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_auroc"
    early_stopping_mode: str = "max"

    # Checkpointing
    checkpoint_metric: str = "val_auroc"
    checkpoint_mode: str = "max"
    save_top_k: int = 1

    # Directories
    log_dir: str = "./log_seizure_detection"
    checkpoint_dir: str = "./checkpoints"
    save_dir: str = "./saved_models"
    results_dir: str = "./results"

    # Accelerator
    accelerator: str = "auto"  # "auto", "gpu", "cpu"
    devices: int = 1
    use_mixed_precision: bool = True  # Use 16-bit mixed precision (faster on modern GPUs)

    def __post_init__(self):
        """Validate configuration and create directories."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")

        # Create directories if they don't exist
        for dir_path in [self.log_dir, self.checkpoint_dir, self.save_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all components."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Experiment metadata
    experiment_name: str = "chbmit_seizure"
    seed: int = 12345

    def __post_init__(self):
        """Update experiment name based on model."""
        if self.experiment_name == "chbmit_seizure":
            # Auto-generate experiment name
            self.experiment_name = f"CHB_MIT_8patients-{self.model.name}-lr{self.training.lr}-bs{self.data.batch_size}-sr{self.data.sampling_rate}"

    def summary(self) -> str:
        """Generate a summary string of the configuration."""
        lines = [
            "=" * 80,
            f"Experiment: {self.experiment_name}",
            "=" * 80,
            "",
            "Data Configuration:",
            f"  Root path: {self.data.root_path}",
            f"  Train patients: {self.data.train_patients[0]}-{self.data.train_patients[1]}",
            f"  Val patients: {self.data.val_patients[0]}-{self.data.val_patients[1]}",
            f"  Test patients: {self.data.test_patients[0]}-{self.data.test_patients[1]}",
            f"  Batch size: {self.data.batch_size}",
            f"  Num workers: {self.data.num_workers}",
            f"  Data fraction: {self.data.data_fraction*100:.1f}%",
            "",
            "Model Configuration:",
            f"  Model: {self.model.name}",
            f"  Channels: {self.model.in_channels}",
            f"  Attention heads: {self.model.nhead}",
            f"  Layers: {self.model.num_layers}",
            f"  Embedding size: {self.model.emb_size}",
            f"  Dropout: {self.model.dropout}",
            "",
            "Training Configuration:",
            f"  Epochs: {self.training.epochs}",
            f"  Learning rate: {self.training.lr}",
            f"  Weight decay: {self.training.weight_decay}",
            f"  Early stopping patience: {self.training.early_stopping_patience}",
            f"  Accelerator: {self.training.accelerator}",
            "",
            f"Random seed: {self.seed}",
            "=" * 80,
        ]
        return "\n".join(lines)
