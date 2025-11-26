"""Base model classes for language models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn


class BaseLanguageModel(nn.Module, ABC):
    """Abstract base class for language models.

    All language models in this repository should extend this class
    to ensure consistent interface for training, evaluation, and inference.
    """

    def __init__(self, vocab_size: int, **kwargs):
        """
        Initialize base language model.

        Args:
            vocab_size: Size of the vocabulary
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.model_config = kwargs

    @abstractmethod
    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target token IDs for computing loss [batch_size, seq_len]

        Returns:
            Dictionary containing:
                - 'logits': Model output logits [batch_size, seq_len, vocab_size]
                - 'loss': Computed loss (if labels provided)
        """
        pass

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token IDs [batch_size, max_length]
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Start with input_ids
        generated = input_ids

        for _ in range(max_length - input_ids.size(1)):
            # Get predictions for last token
            outputs = self.forward(generated)
            logits = outputs["logits"]

            # Get logits for next token
            next_token_logits = logits[:, -1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[
                    0
                ][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[Any] = None,
        epoch: Optional[int] = None,
        **kwargs
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            **kwargs: Additional metadata to save (e.g., loss, metrics)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_config": self.model_config,
            "vocab_size": self.vocab_size,
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        # Add any additional metadata
        for key, value in kwargs.items():
            checkpoint[key] = value

        torch.save(checkpoint, path)
        print(f"✓ Checkpoint saved: {path}")

    def load_checkpoint(
        self, path: Union[str, Path], optimizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
            optimizer: Optional optimizer to load state into

        Returns:
            Dictionary with checkpoint metadata (epoch, etc.)
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location="cpu")

        self.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"✓ Checkpoint loaded: {path}")

        return {
            "epoch": checkpoint.get("epoch"),
            "model_config": checkpoint.get("model_config"),
        }

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            "vocab_size": self.vocab_size,
            **self.model_config,
        }
