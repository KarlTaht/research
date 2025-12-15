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


class TorchTransformer(nn.Module):
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
        super().__init__()
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

    def _init_transformer_network(self):
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Embedding(self.max_seq_len, self.d_model)

        # Transformer Blocks
        self.register_buffer('causal_mask', self._get_causal_mask())
        self.blocks = nn.ModuleList([
            TransformerBlock(self.d_model, self.d_ffn, self.n_heads)
            for i in range(self.n_blocks)
        ])

        # Final Normalization (often ln_f in community)
        self.final_norm = nn.RMSNorm(self.d_model)

        # Output Projection (does not include weight-tieing)
        self.output_projection = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # Weight Sync (Embedding and Projection are inverse operations)
        self.output_projection.weight = self.token_embedding.weight

        # Apply proper weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with GPT-2 style scaling.

        Critical for tied embeddings: default nn.Embedding init (std=1.0) causes
        logit explosion when d_model is large. We use std=0.02 like GPT-2.
        """
        init_std = 0.02

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _get_causal_mask(self):
        return torch.triu(
            torch.full(
                (self.max_seq_len, self.max_seq_len),
                float('-inf'),
            ),
            diagonal=1,
        )
    
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

    def forward(self, input_ids, labels=None):
        """Forward pass with optional loss computation.

        Args:
            input_ids: batched tokens [batch_size, seq_len]
            labels: target tokens for loss computation [batch_size, seq_len]

        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        logits = self._forward_logits(input_ids)

        output = {'logits': logits}
        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output['loss'] = loss
        return output

    def _forward_logits(self, X):
        """Internal forward pass returning raw logits.

        Args:
            X: batched tokens [batch_size, seq_len] populated with integer indices of tokens.
        """
        # Embed the input
        # shape = [batch_size, seq_len, d_model]
        latent_result = self.embed(X)

        # Pass through transfomer blocks
        # shape = [batch_size, seq_len, d_model]
        latent_result = self.backbone(latent_result)

        # Project the output
        # shape = [batch_size, seq_len, d_vocab]
        logits = self.project(latent_result)

        return logits

    def get_model_info(self) -> Dict[str, Any]:
        """Return model info for logging."""
        params = sum(p.numel() for p in self.parameters())
        return {
            'model_type': 'TorchTransformer',
            'parameters': params,
            'parameters_millions': params / 1e6,
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_blocks': self.n_blocks,
            'n_heads': self.n_heads,
            'd_ffn': self.d_ffn,
            'max_seq_len': self.max_seq_len,
        }

    def embed(self, X):
        # Embeddings are obfuscated as a function
        tokens = self.token_embedding(X)
        
        # This is equivalent to slicing the array in the older method
        # Note that X.device ensures consistency stronger than self.device
        seq_len = X.shape[1]
        pos_indicies = torch.arange(seq_len, device=X.device)
        positions = self.pos_embedding(pos_indicies)
        return tokens + positions

    def backbone(self, X):
        seq_len = X.shape[1]
        mask = self.causal_mask[:seq_len, :seq_len]
        for transformer_block in self.blocks:
            X = transformer_block.forward(X, mask)
        return X
    
    def project(self, X):
        # shape = [batch_size, seq_len, d_model]
        latent_result = self.final_norm(X)
        return self.output_projection(latent_result)


class TransformerBlock(nn.Module):
    """
    Implements a full transformer block of attention + FFN + Normalization.
    """

    def __init__(self, d_model, d_ffn, n_heads):
        super().__init__()

        self.attention_pre_norm = nn.RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads)

        self.ffn_pre_norm = nn.RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ffn)

    def forward(self, X, causal_mask):
        # Attention with Residual Connection (pre-norm style)
        pre_norm = self.attention_pre_norm(X)
        X = X + self.attention(pre_norm, causal_mask)

        # FFN with Residual Connection
        pre_norm = self.ffn_pre_norm(X)
        X = X + self.ffn(pre_norm)
        return X
        

class MultiHeadAttention(nn.Module):
    """
    Implements the attention mechanism (QKV).
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X, causal_mask):
        batch_size = X.shape[0]
        seq_len = X.shape[1]

        # Linear projections of the input onto queries, keys, values, then splitting into attn heads
        Q = self.Q(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)
        K = self.K(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)
        V = self.V(X).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1,2)

        # Compute attention scores and apply mask
        scores = self._stabilize_scores(Q @ K.transpose(-2,-1)) + causal_mask

        # Compute attention probabilities
        probabilities = F.softmax(scores, dim=-1)

        # Apply attention to values
        retrieved_knowledge = probabilities @ V

        # Re-combine attention heads
        retrieved_knowledge = retrieved_knowledge.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        return self.W_o(retrieved_knowledge)

    def _stabilize_scores(self, X):
        return X / (self.d_head ** 0.5)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, X):
        return self.w2(F.gelu(self.w1(X)))


def create_model(config: dict) -> TorchTransformer:
    """Factory function to create a TorchTransformer from config dict.

    Args:
        config: Dict with keys: vocab_size, d_model, n_heads, n_blocks, d_ffn, max_seq_len

    Returns:
        TorchTransformer instance
    """
    return TorchTransformer(config)





