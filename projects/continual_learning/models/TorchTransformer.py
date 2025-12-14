"""
Leverage PyTorch to implement a higher-level abstraction
of the CustomTransformer, decoder-only style. 

TODO:
* Pre-norm (you have post-norm currently—norm before attention/FFN, not after)
* RMSNorm instead of LayerNorm (LLaMA-style)
* Rotary positional embeddings (RoPE) instead of learned absolute
* SwiGLU instead of GELU for FFN (LLaMA-style)
* Weight tying between embedding and output projection
"""

import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.models import BaseLanguageModel


class TorchTransformer(nn.module):
    def __init__(self, config):
        """
        Initialize model with specified layer sizes.

        Args:
            config: Configuration object or dict with optional keys:
                - vocab_size (default: 128)
                - max_seq_len (default: 128)
                - n_blocks (default: 8)
                - n_heads (default: 4)
                - d_model (default: 128)
                - d_ffn (default: 128)
                - device (default: auto-detect)
                - dtype (default: bfloat16)
        """
        self._init_config(config)
        self._init_transformer_network()
    
    def _init_config(self, config):
        def get_config(key, default):
            if hasattr(config, 'get'):
                return config.get(key, default)
            return getattr(config, key, default)

        self.vocab_size = get_config('vocab_size', 128)
        self.max_seq_len = get_config('max_seq_len', 128)

        self.n_blocks = get_config('n_blocks', 8)
        self.n_heads = get_config('n_heads', 4)
        self.d_model = get_config('d_model', 128)
        self.d_ffn = get_config('d_ffn', 128)

        self.d_head = self.d_model // self.n_heads

        self.device = self._resolve_device(config)
        self.dtype = get_config('dtype', None) or torch.bfloat16

        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout

    def _init_transformer_network(self):
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.d_ffn, self.n_heads) 
            for i in range(self.n_blocks)
        ])

        # TODO: Output project
        self.output_projection = None

    
    @staticmethod
    def _resolve_device(config) -> str:
        # Support both dict-style and object-style config access
        device = None
        if hasattr(config, 'get'):
            device = config.get('device', None)
        elif hasattr(config, 'get_device'):
            device = config.get_device()
        elif hasattr(config, 'device'):
            device = config.device

        if device:
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class TransformerBlock(nn.module):
    """
    Implements a full transformer block of attention + FFN + Normalization
    """

    def __init__(self, d_model, d_ffn, n_heads):
        super().__init__()
    
    def forward(self):
        pass


class MultiHeadAttention(nn.module):
    """
    Implements the attention mechanism (QKV)
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_head = d_model // n_heads

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Lienar(d_model, d_model, bias=False)

    def forward(self):
        pass

class FeedForward(nn.module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
    
    def forward(self):
        return self.w2(F.gelu(self.w1(x)))







