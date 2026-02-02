"""Configuration module for BIOT seizure detection experiments."""

from .base_config import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    ExperimentConfig,
)
from .default_configs import (
    get_cnntransformer_config,
    get_biot_config,
    get_sttransformer_config,
    get_config_for_model,
    MODEL_CONFIGS,
)

__all__ = [
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "get_cnntransformer_config",
    "get_biot_config",
    "get_sttransformer_config",
    "get_config_for_model",
    "MODEL_CONFIGS",
]
