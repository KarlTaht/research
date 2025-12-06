"""Configuration classes for CustomTransformer."""

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class CustomTransformerConfig:
    """Configuration for CustomTransformer.

    This config class supports both attribute access and dict-style get() access
    to maintain compatibility with the original CustomTransformer interface.

    Example:
        config = CustomTransformerConfig(vocab_size=50257, d_model=256)
        model = CustomTransformer(config)

        # Or use dict directly:
        model = CustomTransformer({'vocab_size': 50257, 'd_model': 256})
    """

    vocab_size: int = 128
    max_seq_len: int = 128
    n_blocks: int = 8
    n_heads: int = 4
    d_model: int = 128
    d_ffn: int = 128
    device: Optional[str] = None
    dtype: Optional[torch.dtype] = None

    def get(self, key: str, default=None):
        """Dict-style access for compatibility."""
        return getattr(self, key, default)

    def get_device(self) -> Optional[str]:
        """Legacy method for device access."""
        return self.device

    def get_dtype(self) -> Optional[torch.dtype]:
        """Legacy method for dtype access."""
        return self.dtype

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'n_blocks': self.n_blocks,
            'n_heads': self.n_heads,
            'd_model': self.d_model,
            'd_ffn': self.d_ffn,
            'device': self.device,
            'dtype': self.dtype,
        }
