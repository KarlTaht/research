"""Transformer encoder layers and stack."""

from typing import Optional
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward
from .layer_norm import ManualLayerNorm


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer.

    Architecture: Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = ManualLayerNorm(d_model)
        self.norm2 = ManualLayerNorm(d_model)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder layer.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional padding mask [batch, 1, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)

        # Manual dropout
        if self.training and self.dropout_p > 0:
            dropout_mask = torch.bernoulli(torch.full_like(attn_output, 1 - self.dropout_p))
            attn_output = attn_output * dropout_mask / (1 - self.dropout_p)

        x = self.norm1(x + attn_output)  # Add & Norm

        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)

        # Manual dropout
        if self.training and self.dropout_p > 0:
            dropout_mask = torch.bernoulli(torch.full_like(ff_output, 1 - self.dropout_p))
            ff_output = ff_output * dropout_mask / (1 - self.dropout_p)

        x = self.norm2(x + ff_output)  # Add & Norm

        return x


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers.

    Args:
        n_layers: Number of encoder layers
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = ManualLayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through encoder stack.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            mask: Optional padding mask [batch, 1, seq_len]

        Returns:
            Encoder output [batch, seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
