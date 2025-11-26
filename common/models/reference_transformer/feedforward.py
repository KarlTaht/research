"""Position-wise feed-forward network."""

import math
import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """Position-wise feed-forward network using manual tensor operations.

    FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2

    Args:
        d_model: Model dimension
        d_ff: Hidden dimension (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.dropout_p = dropout

        # Manual weight matrices and biases
        # First layer: d_model -> d_ff
        self.W1 = nn.Parameter(torch.randn(d_model, d_ff) * math.sqrt(2.0 / d_model))
        self.b1 = nn.Parameter(torch.zeros(d_ff))

        # Second layer: d_ff -> d_model
        self.W2 = nn.Parameter(torch.randn(d_ff, d_model) * math.sqrt(2.0 / d_ff))
        self.b2 = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply position-wise feed-forward network.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # First layer with ReLU activation
        # [batch, seq_len, d_model] @ [d_model, d_ff] -> [batch, seq_len, d_ff]
        hidden = torch.matmul(x, self.W1) + self.b1
        hidden = torch.relu(hidden)

        # Apply dropout
        if self.training and self.dropout_p > 0:
            mask = torch.bernoulli(torch.full_like(hidden, 1 - self.dropout_p))
            hidden = hidden * mask / (1 - self.dropout_p)

        # Second layer
        # [batch, seq_len, d_ff] @ [d_ff, d_model] -> [batch, seq_len, d_model]
        output = torch.matmul(hidden, self.W2) + self.b2

        return output
