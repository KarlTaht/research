"""Decoder-only Transformer using PyTorch's native implementation.

A clean, educational transformer that uses nn.TransformerDecoder for the
heavy lifting while maintaining full control over embeddings and output.

Target architecture (from plan.md):
- 6 layers, 512 embed dim, 8 heads, 2048 FFN dim
- 1024 context length, 32768 vocab size
- ~25-30M parameters
"""

import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.models import BaseLanguageModel


class TorchTransformer(BaseLanguageModel):
    """Decoder-only transformer using PyTorch's nn.TransformerDecoder.

    This model wraps PyTorch's optimized transformer implementation with:
    - Token + positional embeddings
    - Causal (autoregressive) masking
    - Language model head for next-token prediction

    Args:
        vocab_size: Size of vocabulary
        d_model: Model/embedding dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        n_layers: Number of transformer layers (default: 6)
        d_ff: Feed-forward dimension (default: 2048)
        max_seq_len: Maximum sequence length (default: 1024)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            is_encoder_decoder=False,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            **kwargs,
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout

        # === Embeddings ===
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # === Transformer Decoder ===
        # TODO: Initialize transformer decoder layers
        # decoder_layer = nn.TransformerDecoderLayer(
        #     d_model=d_model,
        #     nhead=n_heads,
        #     dim_feedforward=d_ff,
        #     dropout=dropout,
        #     batch_first=True,
        #     norm_first=True,  # Pre-norm (more stable training)
        # )
        # self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # === Output ===
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but common)
        # self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self._init_weights()

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            self._create_causal_mask(max_seq_len),
            persistent=False,
        )

    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal attention mask.

        Returns:
            Boolean mask where True = masked (cannot attend)
        """
        # TODO: Implement causal mask creation
        # Upper triangular = future tokens = masked
        mask = torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)
        return mask

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        # TODO: Implement weight initialization
        # Common approach: normal init with small std
        # for module in self.modules():
        #     if isinstance(module, nn.Linear):
        #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        #         if module.bias is not None:
        #             torch.nn.init.zeros_(module.bias)
        #     elif isinstance(module, nn.Embedding):
        #         torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for language modeling.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            labels: Target IDs for loss computation [batch_size, seq_len]

        Returns:
            Dict with:
                - 'logits': [batch_size, seq_len, vocab_size]
                - 'loss': scalar (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # === Embeddings ===
        # TODO: Implement embedding lookup
        # positions = torch.arange(seq_len, device=device).unsqueeze(0)
        # x = self.token_embedding(input_ids) + self.position_embedding(positions)
        # x = self.dropout(x)

        # Placeholder
        x = self.token_embedding(input_ids)

        # === Causal Mask ===
        # TODO: Get appropriate slice of causal mask
        # mask = self.causal_mask[:seq_len, :seq_len]

        # === Transformer Layers ===
        # TODO: Pass through transformer decoder
        # For decoder-only, we use the same input as both tgt and memory
        # Or use nn.TransformerEncoder with is_causal=True
        # x = self.transformer(x, memory=x, tgt_mask=mask)

        # === Output ===
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # === Loss ===
        loss = None
        if labels is not None:
            # Shift so we predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore padding
            )

        return {"logits": logits, "loss": loss}

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for logging."""
        return {
            "model_type": "TorchTransformer",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout_rate,
            "parameters": self.count_parameters(),
            "parameters_millions": self.count_parameters() / 1e6,
        }


def create_model(
    vocab_size: int = 32768,
    d_model: int = 512,
    n_heads: int = 8,
    n_layers: int = 6,
    d_ff: int = 2048,
    max_seq_len: int = 1024,
    dropout: float = 0.1,
) -> TorchTransformer:
    """Factory function with default config from plan.md."""
    return TorchTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )


if __name__ == "__main__":
    # Quick sanity check
    print("Creating TorchTransformer with default config...")
    model = create_model()

    info = model.get_model_info()
    print(f"\nModel Info:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    # Test forward pass
    print("\nTesting forward pass...")
    batch = torch.randint(0, 32768, (2, 128))
    output = model(batch, labels=batch)

    print(f"  Input shape: {batch.shape}")
    print(f"  Logits shape: {output['logits'].shape}")
    print(f"  Loss: {output['loss']:.4f}" if output["loss"] else "  Loss: None")

    print("\n✓ Skeleton model works!")
