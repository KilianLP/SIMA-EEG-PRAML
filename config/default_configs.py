"""
Default configurations for different model architectures.

This module provides factory functions to create default configurations
for CNNTransformer, BIOT, and STTransformer models.
"""

from .base_config import DataConfig, ModelConfig, TrainingConfig, ExperimentConfig


def get_cnntransformer_config() -> ExperimentConfig:
    """
    Get default configuration for CNNTransformer model.

    CNNTransformer uses vanilla PyTorch transformer with ResNet-style CNN backbone.

    Returns:
        ExperimentConfig: Configuration with CNNTransformer defaults
    """
    return ExperimentConfig(
        data=DataConfig(
            root_path="/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments",
            train_patients=(1, 5),
            val_patients=(6, 6),
            test_patients=(7, 8),
            batch_size=128,
            num_workers=4,
            sampling_rate=200,
        ),
        model=ModelConfig(
            name="CNNTransformer",
            in_channels=16,
            n_classes=1,
            token_size=200,
            hop_length=100,
            nhead=4,
            num_layers=4,
            emb_size=256,
            dropout=0.2,
        ),
        training=TrainingConfig(
            epochs=50,
            lr=1e-3,
            weight_decay=1e-5,
            early_stopping_patience=10,
        ),
        experiment_name="chbmit_seizure",
        seed=12345,
    )


def get_biot_config() -> ExperimentConfig:
    """
    Get default configuration for BIOT model.

    BIOT uses Linear Attention Transformer for efficient processing of long sequences.
    It typically uses more attention heads than vanilla transformers.

    Returns:
        ExperimentConfig: Configuration with BIOT defaults
    """
    return ExperimentConfig(
        data=DataConfig(
            root_path="/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments",
            train_patients=(1, 5),
            val_patients=(6, 6),
            test_patients=(7, 8),
            batch_size=128,
            num_workers=4,
            sampling_rate=200,
        ),
        model=ModelConfig(
            name="BIOT",
            in_channels=16,
            n_classes=1,
            token_size=200,
            hop_length=100,
            nhead=8,  # BIOT uses more heads
            depth=4,
            emb_size=256,
            dropout=0.2,
        ),
        training=TrainingConfig(
            epochs=50,
            lr=1e-3,
            weight_decay=1e-5,
            early_stopping_patience=10,
        ),
        experiment_name="chbmit_seizure",
        seed=12345,
    )


def get_sttransformer_config() -> ExperimentConfig:
    """
    Get default configuration for STTransformer model.

    STTransformer (Spatial-Temporal Transformer) uses custom attention mechanisms
    for joint spatial and temporal modeling.

    Returns:
        ExperimentConfig: Configuration with STTransformer defaults
    """
    return ExperimentConfig(
        data=DataConfig(
            root_path="/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments",
            train_patients=(1, 5),
            val_patients=(6, 6),
            test_patients=(7, 8),
            batch_size=128,
            num_workers=4,
            sampling_rate=200,
        ),
        model=ModelConfig(
            name="STTransformer",
            in_channels=16,
            n_classes=1,
            token_size=200,
            hop_length=100,
            nhead=8,
            depth=3,  # STTransformer typically uses fewer layers
            emb_size=256,
            dropout=0.2,
        ),
        training=TrainingConfig(
            epochs=50,
            lr=1e-3,
            weight_decay=1e-5,
            early_stopping_patience=10,
        ),
        experiment_name="chbmit_seizure",
        seed=12345,
    )


def get_eegformer_config() -> ExperimentConfig:
    """
    Get default configuration for EEGformer model.

    EEGformer uses 1D convolutions for embedding, positional encoding,
    and standard transformer encoder blocks. Designed for EEG classification
    following the paper architecture specifications.

    Returns:
        ExperimentConfig: Configuration with EEGformer defaults
    """
    return ExperimentConfig(
        data=DataConfig(
            root_path="/Brain/private/DT_Reve_tmp/CHBMIT_processed/clean_segments",
            train_patients=(1, 5),
            val_patients=(6, 6),
            test_patients=(7, 8),
            batch_size=16,
            num_workers=4,
            sampling_rate=200,
        ),
        model=ModelConfig(
            name="EEGformer",
            in_channels=16,
            n_classes=1,
            token_size=200,  # Not directly used, but kept for compatibility
            hop_length=100,  # Not directly used, but kept for compatibility
            nhead=8,  # From paper: H=8
            num_layers=1,  # From paper: 1 transformer encoder block
            emb_size=32,  # From paper: E=32 (smaller than other models)
            dropout=0.1,
        ),
        training=TrainingConfig(
            epochs=50,
            lr=5e-5,
            weight_decay=1e-5,
        ),
        experiment_name="chbmit_seizure",
        seed=12345,
    )


# Model registry mapping for easy access
MODEL_CONFIGS = {
    "CNNTransformer": get_cnntransformer_config,
    "BIOT": get_biot_config,
    "STTransformer": get_sttransformer_config,
    "EEGformer": get_eegformer_config,
}


def get_config_for_model(model_name: str) -> ExperimentConfig:
    """
    Get default configuration for a specific model.

    Args:
        model_name: Name of the model ("CNNTransformer", "BIOT", or "STTransformer")

    Returns:
        ExperimentConfig: Default configuration for the specified model

    Raises:
        ValueError: If model_name is not recognized
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_CONFIGS.keys())}"
        )
    return MODEL_CONFIGS[model_name]()
