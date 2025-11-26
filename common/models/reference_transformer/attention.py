"""Attention mechanisms for transformer models."""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism.

    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

    Args:
        dropout: Dropout probability for attention weights
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout_p = dropout

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention.

        Args:
            query: Query tensor [batch, n_heads, seq_len_q, d_k]
            key: Key tensor [batch, n_heads, seq_len_k, d_k]
            value: Value tensor [batch, n_heads, seq_len_v, d_v]
            mask: Optional mask [batch, 1, 1, seq_len_k] or [batch, 1, seq_len_q, seq_len_k]
                  True/1 positions will be masked (set to -inf before softmax)

        Returns:
            output: Attention output [batch, n_heads, seq_len_q, d_v]
            attention_weights: Attention weights [batch, n_heads, seq_len_q, seq_len_k]
        """
        d_k = query.size(-1)

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        # [batch, n_heads, seq_len_q, d_k] @ [batch, n_heads, d_k, seq_len_k]
        # -> [batch, n_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        # Apply softmax to get attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply dropout to attention weights
        if self.training and self.dropout_p > 0:
            dropout_mask = torch.bernoulli(
                torch.full_like(attention_weights, 1 - self.dropout_p)
            )
            attention_weights = attention_weights * dropout_mask / (1 - self.dropout_p)

        # Apply attention weights to values
        # [batch, n_heads, seq_len_q, seq_len_k] @ [batch, n_heads, seq_len_k, d_v]
        # -> [batch, n_heads, seq_len_q, d_v]
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism using pure tensor operations.

    Implements parallel attention heads with manual weight matrices.
    No nn.Linear - all projections are manual matmuls with nn.Parameter weights.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        # Manual weight matrices for Q, K, V projections
        # [d_model, d_model] - will split into n_heads later
        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * math.sqrt(2.0 / d_model))
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * math.sqrt(2.0 / d_model))
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * math.sqrt(2.0 / d_model))

        # Output projection weight
        self.W_O = nn.Parameter(torch.randn(d_model, d_model) * math.sqrt(2.0 / d_model))

        self.attention = ScaledDotProductAttention(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional mask [batch, 1, seq_len_k] or [batch, seq_len_q, seq_len_k]

        Returns:
            output: Attention output [batch, seq_len_q, d_model]
        """
        batch_size = query.size(0)

        # Linear projections using manual matmul
        # [batch, seq_len, d_model] @ [d_model, d_model] -> [batch, seq_len, d_model]
        Q = torch.matmul(query, self.W_Q)
        K = torch.matmul(key, self.W_K)
        V = torch.matmul(value, self.W_V)

        # Split into multiple heads and reshape
        # [batch, seq_len, d_model] -> [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Adjust mask dimensions if needed
        if mask is not None:
            # Add head dimension: [batch, 1, ...] for broadcasting
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len_q, seq_len_k]

        # Apply attention
        # output: [batch, n_heads, seq_len_q, d_k]
        attn_output, _ = self.attention(Q, K, V, mask)

        # Concatenate heads
        # [batch, n_heads, seq_len, d_k] -> [batch, seq_len, n_heads, d_k] -> [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Apply output projection
        # [batch, seq_len, d_model] @ [d_model, d_model] -> [batch, seq_len, d_model]
        output = torch.matmul(attn_output, self.W_O)

        return output
