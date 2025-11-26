"""Reference Transformer implementation with pure tensor operations.

This module provides a complete encoder-decoder Transformer architecture
based on "Attention Is All You Need" (Vaswani et al., 2017), implemented
using only low-level PyTorch tensor operations.

Main Components:
    - ReferenceTransformer: Main encoder-decoder model
    - PositionalEncoding: Sinusoidal position embeddings
    - MultiHeadAttention: Multi-head attention mechanism
    - TransformerEncoder: Stack of encoder layers
    - TransformerDecoder: Stack of decoder layers
    - ManualLayerNorm: Layer normalization
    - PositionWiseFeedForward: Feed-forward network

All components use pure tensor operations (no nn.Linear, nn.LayerNorm, etc.)
for educational purposes.
"""

from .model import ReferenceTransformer
from .positional_encoding import PositionalEncoding
from .attention import ScaledDotProductAttention, MultiHeadAttention
from .layer_norm import ManualLayerNorm
from .feedforward import PositionWiseFeedForward
from .encoder import TransformerEncoderLayer, TransformerEncoder
from .decoder import TransformerDecoderLayer, TransformerDecoder

__all__ = [
    # Main model
    "ReferenceTransformer",
    # Core components
    "PositionalEncoding",
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "ManualLayerNorm",
    "PositionWiseFeedForward",
    # Encoder components
    "TransformerEncoderLayer",
    "TransformerEncoder",
    # Decoder components
    "TransformerDecoderLayer",
    "TransformerDecoder",
]
