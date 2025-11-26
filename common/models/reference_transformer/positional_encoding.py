"""Positional encoding for transformer models."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (fixed, not learned).

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout_p = dropout

        # Create positional encoding matrix [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # Compute div_term for scaling
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model] for broadcasting

        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            x + positional encoding with dropout applied
        """
        # Add positional encoding (broadcasting over batch dimension)
        x = x + self.pe[:, :x.size(1), :]

        # Manual dropout implementation
        if self.training and self.dropout_p > 0:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout_p))
            x = x * mask / (1 - self.dropout_p)

        return x
