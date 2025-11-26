"""Reference implementation of encoder-decoder Transformer model."""

import math
from typing import Optional, Dict
import torch
import torch.nn as nn

from ..base import BaseLanguageModel
from .positional_encoding import PositionalEncoding
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder


class ReferenceTransformer(BaseLanguageModel):
    """Reference implementation of encoder-decoder Transformer architecture.

    This is an ENCODER-DECODER model for sequence-to-sequence tasks (translation,
    summarization, etc.). Based on "Attention Is All You Need" (Vaswani et al., 2017).

    Implemented using pure tensor operations (no nn.Linear, nn.LayerNorm, etc.) for
    educational purposes. This is a reference implementation meant to be manually
    rewritten later.

    Architecture:
        - Encoder: Processes source sequence with self-attention
        - Decoder: Generates target sequence with masked self-attention + cross-attention

    Input Format (sequence packing):
        Since BaseLanguageModel expects single input_ids, we pack source and target:
        input_ids = [SRC tokens] [SEP] [TGT tokens]

        Example: [5, 234, 89, SEP_ID, 8, 156, 201]
                 └─ source ─┘  sep  └─ target ─┘

    Args:
        vocab_size: Vocabulary size (shared for source and target)
        d_model: Model dimension (default: 512)
        n_heads: Number of attention heads (default: 8)
        n_encoder_layers: Number of encoder layers (default: 6)
        n_decoder_layers: Number of decoder layers (default: 6)
        d_ff: Feed-forward hidden dimension (default: 2048)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length (default: 512)
        sep_token_id: Separator token ID for splitting src/tgt (default: vocab_size - 1)
        pad_token_id: Padding token ID (default: 0)
        **kwargs: Additional arguments passed to BaseLanguageModel

    Example:
        >>> model = ReferenceTransformer(vocab_size=30000, d_model=512, n_heads=8)
        >>> # Pack sequences: source + sep + target
        >>> input_ids = torch.tensor([[100, 200, 300, 29999, 400, 500, 600]])
        >>> output = model(input_ids)
        >>> print(output['logits'].shape)  # [1, 3, 30000] (target sequence length)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        sep_token_id: Optional[int] = None,
        pad_token_id: int = 0,
        **kwargs
    ):
        super().__init__(vocab_size, **kwargs)

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.sep_token_id = sep_token_id if sep_token_id is not None else vocab_size - 1
        self.pad_token_id = pad_token_id

        # Manual embedding matrices (replaces nn.Embedding)
        # Shared embeddings for source and target (as in original paper)
        self.embedding = nn.Parameter(
            torch.randn(vocab_size, d_model) * math.sqrt(1.0 / d_model)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder and decoder stacks
        self.encoder = TransformerEncoder(n_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(n_decoder_layers, d_model, n_heads, d_ff, dropout)

        # Output projection: d_model -> vocab_size (manual weight matrix)
        self.output_projection = nn.Parameter(
            torch.randn(d_model, vocab_size) * math.sqrt(1.0 / d_model)
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        # Embeddings and output projection already initialized in __init__
        # Layer weights are initialized in their respective classes
        pass

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for decoder (prevents attending to future positions).

        Args:
            seq_len: Sequence length
            device: Device to create mask on

        Returns:
            Causal mask [1, seq_len, seq_len] where True indicates masked position
        """
        # Upper triangular matrix (above diagonal = True = masked)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool().unsqueeze(0)  # [1, seq_len, seq_len]

    def _create_padding_mask(
        self,
        input_ids: torch.Tensor,
        pad_token_id: int
    ) -> torch.Tensor:
        """Create padding mask.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            pad_token_id: Padding token ID

        Returns:
            Padding mask [batch, 1, seq_len] where True indicates padding
        """
        # Padding mask: True where input is pad token
        return (input_ids == pad_token_id).unsqueeze(1)  # [batch, 1, seq_len]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for encoder-decoder transformer.

        Input format (sequence packing): [SRC tokens] [SEP] [TGT tokens]
        The sequence is split on SEP token to get source and target.

        Args:
            input_ids: Packed input [batch, total_seq_len]
                      Format: [src_1, ..., src_n, SEP, tgt_1, ..., tgt_m]
            labels: Target labels [batch, total_seq_len] (optional, for loss computation)
                    Should match input_ids for autoregressive training

        Returns:
            Dictionary containing:
                - 'logits': Output logits [batch, tgt_seq_len, vocab_size]
                - 'loss': Computed loss (if labels provided)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Split input into source and target based on separator token
        # Find separator positions for each sequence in batch
        src_ids_list = []
        tgt_ids_list = []

        for i in range(batch_size):
            seq = input_ids[i]
            # Find first occurrence of separator
            sep_positions = (seq == self.sep_token_id).nonzero(as_tuple=True)[0]

            if len(sep_positions) == 0:
                raise ValueError(
                    f"Separator token {self.sep_token_id} not found in sequence {i}. "
                    "Input must be in format: [SRC] [SEP] [TGT]"
                )

            sep_pos = sep_positions[0].item()
            src_ids_list.append(seq[:sep_pos])
            tgt_ids_list.append(seq[sep_pos + 1:])

        # Pad sequences to same length within batch
        max_src_len = max(len(s) for s in src_ids_list)
        max_tgt_len = max(len(t) for t in tgt_ids_list)

        src_ids = torch.full(
            (batch_size, max_src_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device
        )
        tgt_ids = torch.full(
            (batch_size, max_tgt_len),
            self.pad_token_id,
            dtype=torch.long,
            device=device
        )

        for i in range(batch_size):
            src_len = len(src_ids_list[i])
            tgt_len = len(tgt_ids_list[i])
            src_ids[i, :src_len] = src_ids_list[i]
            tgt_ids[i, :tgt_len] = tgt_ids_list[i]

        # --- ENCODER ---
        # Embed source tokens (manual embedding lookup)
        src_embedded = self.embedding[src_ids] * math.sqrt(self.d_model)  # Scale embeddings
        src_embedded = self.pos_encoder(src_embedded)

        # Create source padding mask
        src_padding_mask = self._create_padding_mask(src_ids, self.pad_token_id)

        # Encode source sequence
        encoder_output = self.encoder(src_embedded, src_padding_mask)

        # --- DECODER ---
        # Embed target tokens (manual embedding lookup)
        tgt_embedded = self.embedding[tgt_ids] * math.sqrt(self.d_model)  # Scale embeddings
        tgt_embedded = self.pos_encoder(tgt_embedded)

        # Create target causal mask (can't attend to future positions)
        tgt_causal_mask = self._create_causal_mask(max_tgt_len, device)

        # Decode with cross-attention to encoder output
        decoder_output = self.decoder(
            tgt_embedded,
            encoder_output,
            tgt_mask=tgt_causal_mask,
            memory_mask=src_padding_mask
        )

        # Project to vocabulary (manual matmul)
        # [batch, tgt_seq_len, d_model] @ [d_model, vocab_size] -> [batch, tgt_seq_len, vocab_size]
        logits = torch.matmul(decoder_output, self.output_projection)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Extract target labels (same splitting logic)
            tgt_labels_list = []
            for i in range(batch_size):
                seq = labels[i]
                sep_positions = (seq == self.sep_token_id).nonzero(as_tuple=True)[0]
                sep_pos = sep_positions[0].item()
                tgt_labels_list.append(seq[sep_pos + 1:])

            tgt_labels = torch.full(
                (batch_size, max_tgt_len),
                self.pad_token_id,
                dtype=torch.long,
                device=device
            )
            for i in range(batch_size):
                tgt_len = len(tgt_labels_list[i])
                tgt_labels[i, :tgt_len] = tgt_labels_list[i]

            # Reshape for CrossEntropyLoss
            # logits: [batch * tgt_seq_len, vocab_size]
            # labels: [batch * tgt_seq_len]
            loss = self.criterion(
                logits.reshape(-1, self.vocab_size),
                tgt_labels.reshape(-1)
            )

        return {
            'logits': logits,
            'loss': loss
        }

    def get_model_info(self) -> Dict[str, any]:
        """Get model configuration information."""
        return {
            'architecture': 'Encoder-Decoder Transformer (Reference Implementation)',
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_encoder_layers': self.n_encoder_layers,
            'n_decoder_layers': self.n_decoder_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout,
            'max_seq_len': self.max_seq_len,
            'parameters': self.count_parameters(),
            'implementation': 'Pure tensor operations (manual matmuls, no nn.Linear/LayerNorm)'
        }
