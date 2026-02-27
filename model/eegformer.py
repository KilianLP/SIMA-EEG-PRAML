import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    """
    def __init__(self, d_model, max_len = 5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # pe shape: (1, max_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer encoder block with:
    - Multi-head self-attention
    - Layer normalization
    - Feed-forward network (Dense0 -> Dense1)
    - Residual connections
    """
    def __init__(self, d_model = 32, nhead = 8, dim_feedforward = 128, dropout = 0):
        super().__init__()

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,  # Input format: (batch, seq, feature)
        )

        # Layer normalizations
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network (MLP)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),  # Dense0
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),  # Dense1
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Forward pass through transformer encoder block.
        """
        # Multi-head attention with residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class EEGformer(nn.Module):
    """
    EEGformer: Transformer-based architecture for EEG classification.

    Architecture:
    1. Embedding stage: Two 1D convolutions to reduce temporal dimension
    2. Positional encoding: Add learnable position information
    3. Encoder stage: Stack of transformer encoder blocks
    4. Classification stage: Global average pooling + MLP
    """
    def __init__(
        self,
        in_channels: int = 16,
        n_classes: int = 1,
        seq_length: int = 2000,  # 10s at 200Hz
        embed_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 1,
        dim_feedforward: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.seq_length = seq_length
        self.embed_dim = embed_dim

        # Calculate sequence lengths after convolutions
        # After Conv0: S' = (W - K) / K + 1 = (seq_length - kernel_size) / kernel_size + 1
        self.seq_after_conv0 = (seq_length - kernel_size) // kernel_size + 1
        # After Conv1: S = (S' - K) / K + 1
        self.seq_after_conv1 = (self.seq_after_conv0 - kernel_size) // kernel_size + 1

        # ============================================================
        # EMBEDDING STAGE
        # ============================================================

        # Convolution0: 4×W×1 -> E×S'×1
        # Input: (batch, in_channels, seq_length)
        # Output: (batch, embed_dim, seq_after_conv0)
        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
        )

        # Convolution1: E×S'×1 -> E×S×1
        # Input: (batch, embed_dim, seq_after_conv0)
        # Output: (batch, embed_dim, seq_after_conv1)
        self.conv1 = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=0,
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=embed_dim,
            max_len=self.seq_after_conv1 + 10,  # Add margin
        )

        # Layer norm after positional encoding
        self.norm_embed = nn.LayerNorm(embed_dim)

        # ============================================================
        # ENCODER STAGE
        # ============================================================

        # Stack of transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # ============================================================
        # CLASSIFICATION STAGE
        # ============================================================

        # Global average pooling (Reduce Mean)
        # Reduces (batch, seq, embed_dim) -> (batch, embed_dim)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final classification layer (Dense2)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        """
        Forward pass through EEGformer.
        """
        # ============================================================
        # EMBEDDING STAGE
        # ============================================================

        # Input: (batch, in_channels, seq_length)
        # Apply Conv0
        x = self.conv0(x)  # (batch, embed_dim, seq_after_conv0)

        # Apply Conv1
        x = self.conv1(x)  # (batch, embed_dim, seq_after_conv1)

        # Transpose for transformer: (batch, seq, embed_dim)
        x = x.transpose(1, 2)  # (batch, seq_after_conv1, embed_dim)

        # Add positional encoding
        x = self.pos_encoder(x)  # (batch, seq, embed_dim)

        # Layer normalization
        x = self.norm_embed(x)  # (batch, seq, embed_dim)

        # ============================================================
        # ENCODER STAGE
        # ============================================================

        # Apply transformer encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)  # (batch, seq, embed_dim)

        # ============================================================
        # CLASSIFICATION STAGE
        # ============================================================

        # Global average pooling over sequence dimension
        # Transpose back: (batch, embed_dim, seq)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq)

        # Apply global pooling: (batch, embed_dim, 1)
        x = self.global_pool(x)  # (batch, embed_dim, 1)

        # Remove last dimension: (batch, embed_dim)
        x = x.squeeze(-1)  # (batch, embed_dim)

        # Classification layer
        x = self.classifier(x)  # (batch, n_classes)

        # For binary classification with n_classes=1, squeeze to (batch,)
        if self.n_classes == 1:
            x = x.squeeze(-1)

        return x

    def get_attention_maps(self, x):
        """
        Get attention maps from all encoder blocks (useful for visualization).
        """
        attention_maps = []

        # Embedding stage (same as forward)
        x = self.conv0(x)
        x = self.conv1(x)
        x = x.transpose(1, 2)
        x = self.pos_encoder(x)
        x = self.norm_embed(x)

        # Encoder stage - collect attention weights
        for encoder_block in self.encoder_blocks:
            # Get attention weights
            attn_output, attn_weights = encoder_block.self_attn(x, x, x)
            attention_maps.append(attn_weights)

            # Continue forward pass
            x = encoder_block.norm1(x + attn_output)
            ffn_output = encoder_block.ffn(x)
            x = encoder_block.norm2(x + ffn_output)

        return attention_maps


if __name__ == "__main__":
    # Test the model
    print("Testing EEGformer architecture...")

    # Create model with CHB-MIT settings
    model = EEGformer(
        in_channels=16,
        n_classes=1,
        seq_length=2000,  # 10s at 200Hz
        embed_dim=32,
        num_heads=8,
        num_layers=1,
        dim_feedforward=128,
        kernel_size=5,
        dropout=0.1,
    )

    # Print model info
    print(f"\nModel: {model.__class__.__name__}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 16, 2000)
    print(f"\nInput shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")

    print("\n✓ EEGformer test passed!")
