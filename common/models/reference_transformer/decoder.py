"""Transformer decoder layers and stack."""

from typing import Optional
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward
from .layer_norm import ManualLayerNorm


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture:
        Masked Self-Attention -> Add & Norm ->
        Cross-Attention (to encoder) -> Add & Norm ->
        Feed-Forward -> Add & Norm

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = ManualLayerNorm(d_model)
        self.norm2 = ManualLayerNorm(d_model)
        self.norm3 = ManualLayerNorm(d_model)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through decoder layer.

        Args:
            x: Target input [batch, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch, src_seq_len, d_model]
            tgt_mask: Target causal mask [batch, tgt_seq_len, tgt_seq_len]
            memory_mask: Encoder padding mask [batch, 1, src_seq_len]

        Returns:
            Output tensor [batch, tgt_seq_len, d_model]
        """
        # Masked self-attention (can't attend to future positions)
        self_attn_output = self.self_attn(x, x, x, tgt_mask)

        # Manual dropout
        if self.training and self.dropout_p > 0:
            dropout_mask = torch.bernoulli(
                torch.full_like(self_attn_output, 1 - self.dropout_p)
            )
            self_attn_output = self_attn_output * dropout_mask / (1 - self.dropout_p)

        x = self.norm1(x + self_attn_output)  # Add & Norm

        # Cross-attention to encoder output
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, memory_mask)

        # Manual dropout
        if self.training and self.dropout_p > 0:
            dropout_mask = torch.bernoulli(
                torch.full_like(cross_attn_output, 1 - self.dropout_p)
            )
            cross_attn_output = cross_attn_output * dropout_mask / (1 - self.dropout_p)

        x = self.norm2(x + cross_attn_output)  # Add & Norm

        # Feed-forward
        ff_output = self.feed_forward(x)

        # Manual dropout
        if self.training and self.dropout_p > 0:
            dropout_mask = torch.bernoulli(torch.full_like(ff_output, 1 - self.dropout_p))
            ff_output = ff_output * dropout_mask / (1 - self.dropout_p)

        x = self.norm3(x + ff_output)  # Add & Norm

        return x


class TransformerDecoder(nn.Module):
    """Stack of transformer decoder layers.

    Args:
        n_layers: Number of decoder layers
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
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = ManualLayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through decoder stack.

        Args:
            x: Target input [batch, tgt_seq_len, d_model]
            encoder_output: Encoder output [batch, src_seq_len, d_model]
            tgt_mask: Target causal mask [batch, tgt_seq_len, tgt_seq_len]
            memory_mask: Encoder padding mask [batch, 1, src_seq_len]

        Returns:
            Decoder output [batch, tgt_seq_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)

        return self.norm(x)
