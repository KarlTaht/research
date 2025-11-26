"""Manual layer normalization implementation."""

import torch
import torch.nn as nn


class ManualLayerNorm(nn.Module):
    """Layer normalization using manual tensor operations.

    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta

    Args:
        d_model: Model dimension
        eps: Small constant for numerical stability
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

        # Learnable scale and shift parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Normalized tensor [batch, seq_len, d_model]
        """
        # Compute mean and variance over last dimension (d_model)
        mean = x.mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)  # [batch, seq_len, 1]

        # Normalize
        normalized = (x - mean) / torch.sqrt(variance + self.eps)

        # Apply learnable scale and shift
        output = self.gamma * normalized + self.beta

        return output
