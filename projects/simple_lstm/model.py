"""Simple LSTM language model."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import sys
from pathlib import Path

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models import BaseLanguageModel


class SimpleLSTM(BaseLanguageModel):
    """Simple LSTM-based language model.

    Architecture:
        - Embedding layer
        - LSTM layers
        - Linear output projection

    This is a minimal but complete implementation for testing the infrastructure.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        """
        Initialize LSTM language model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of token embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )

        # Layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embeddings
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Initialize output layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len]

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Embed input tokens
        embedded = self.embedding(input_ids)  # [batch, seq_len, embedding_dim]

        # LSTM forward
        lstm_out, _ = self.lstm(embedded)  # [batch, seq_len, hidden_dim]

        # Apply dropout
        lstm_out = self.dropout(lstm_out)

        # Project to vocabulary
        logits = self.fc(lstm_out)  # [batch, seq_len, vocab_size]

        # Prepare output
        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # Reshape for loss computation
            loss = self.criterion(
                logits.view(-1, self.vocab_size), labels.view(-1)
            )
            output["loss"] = loss

        return output

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            "model_type": "SimpleLSTM",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.model_config["embedding_dim"],
            "hidden_dim": self.model_config["hidden_dim"],
            "num_layers": self.model_config["num_layers"],
            "dropout": self.model_config["dropout"],
            "parameters": self.count_parameters(),
        }
