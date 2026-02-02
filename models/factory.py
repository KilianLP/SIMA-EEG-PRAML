"""
Model factory for creating transformer architectures.

This module provides a registry pattern for creating different model architectures
from configuration objects. It enables easy switching between models without
modifying training code.
"""

from typing import Dict, Callable
import torch
import torch.nn as nn

# Import from model/ directory (singular - where existing models are)
from model.cnn_transformer import CNNTransformer
from model.biot import BIOTClassifier
from model.st_transformer import STTransformer
from model.eegformer import EEGformer
from config.base_config import ModelConfig


class ModelRegistry:
    """Registry for model builder functions."""

    def __init__(self):
        self._builders: Dict[str, Callable[[ModelConfig], nn.Module]] = {}

    def register(self, name: str):
        """
        Decorator to register a model builder function.

        Args:
            name: Name of the model

        Example:
            @model_registry.register("MyModel")
            def build_my_model(config: ModelConfig) -> nn.Module:
                return MyModel(...)
        """
        def decorator(builder_fn: Callable[[ModelConfig], nn.Module]):
            self._builders[name] = builder_fn
            return builder_fn
        return decorator

    def build(self, config: ModelConfig) -> nn.Module:
        """
        Build a model from configuration.

        Args:
            config: Model configuration

        Returns:
            Instantiated model

        Raises:
            ValueError: If model name is not registered
        """
        if config.name not in self._builders:
            raise ValueError(
                f"Unknown model: {config.name}. "
                f"Available models: {list(self._builders.keys())}"
            )
        return self._builders[config.name](config)

    def list_models(self):
        """List all registered model names."""
        return list(self._builders.keys())


# Global registry instance
model_registry = ModelRegistry()


@model_registry.register("CNNTransformer")
def build_cnn_transformer(config: ModelConfig) -> CNNTransformer:
    """
    Build CNNTransformer model.

    CNNTransformer uses ResNet-style CNN blocks for feature extraction
    followed by vanilla PyTorch Transformer encoder.

    Args:
        config: Model configuration

    Returns:
        CNNTransformer instance
    """
    return CNNTransformer(
        in_channels=config.in_channels,
        n_classes=config.n_classes,
        fft=config.token_size,
        steps=config.hop_length // 5,  # Original script uses hop_length // 5
        dropout=config.dropout,
        nhead=config.nhead,
        emb_size=config.emb_size,
        n_segments=5,  # Fixed parameter
    )


@model_registry.register("BIOT")
def build_biot(config: ModelConfig) -> BIOTClassifier:
    """
    Build BIOT (Linear Attention Transformer) model.

    BIOT uses Linear Attention Transformer for efficient processing
    of long sequences. Supports loading pretrained encoder weights.

    Args:
        config: Model configuration

    Returns:
        BIOTClassifier instance
    """
    model = BIOTClassifier(
        emb_size=config.emb_size,
        heads=config.nhead,
        depth=config.depth,
        n_classes=config.n_classes,
        n_channels=config.in_channels,
        n_fft=config.token_size,
        hop_length=config.hop_length,
    )

    # Load pretrained encoder weights if specified
    if config.pretrain_model_path:
        try:
            state_dict = torch.load(config.pretrain_model_path, map_location='cpu')
            model.biot.load_state_dict(state_dict)
            print(f"✓ Loaded pretrained BIOT encoder from {config.pretrain_model_path}")
        except Exception as e:
            print(f"⚠ Warning: Failed to load pretrained model from {config.pretrain_model_path}")
            print(f"  Error: {e}")
            print("  Continuing with random initialization...")

    return model


@model_registry.register("STTransformer")
def build_st_transformer(config: ModelConfig) -> STTransformer:
    """
    Build Spatial-Temporal Transformer model.

    STTransformer uses custom attention mechanisms for joint
    spatial (channel) and temporal modeling.

    Args:
        config: Model configuration

    Returns:
        STTransformer instance
    """
    # Calculate channel_length based on sampling rate and sample length
    # Assuming 10 seconds at 200 Hz = 2000 time points
    channel_length = int(config.in_channels * 125)  # Default: 16 * 125 = 2000

    return STTransformer(
        emb_size=config.emb_size,
        depth=config.depth,
        n_classes=config.n_classes,
        channel_legnth=channel_length,  # Note: typo in original code
        n_channels=config.in_channels,
    )


@model_registry.register("EEGformer")
def build_eegformer(config: ModelConfig) -> EEGformer:
    """
    Build EEGformer model.

    EEGformer uses 1D convolutions for embedding followed by
    standard transformer encoder with positional encoding.
    Designed specifically for EEG classification tasks.

    Args:
        config: Model configuration

    Returns:
        EEGformer instance
    """
    # Get sequence length from config (set by training script) or use default
    seq_length = getattr(config, 'seq_length', 2000)

    return EEGformer(
        in_channels=config.in_channels,
        n_classes=config.n_classes,
        seq_length=seq_length,
        embed_dim=config.emb_size,
        num_heads=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.emb_size * 4,  # Standard transformer ratio
        kernel_size=5,  # From paper specification
        dropout=config.dropout,
    )
