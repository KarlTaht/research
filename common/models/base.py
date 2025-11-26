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

    Subclasses must implement:
        - forward(): Model forward pass
        - get_model_info(): Return model metadata for logging

    Subclasses may override:
        - generate(): Text generation (default implementation for decoder-only)
        - _init_weights(): Weight initialization
    """

    def __init__(
        self,
        vocab_size: int,
        is_encoder_decoder: bool = False,
        **kwargs
    ):
        """
        Initialize base language model.

        Args:
            vocab_size: Size of the vocabulary
            is_encoder_decoder: Whether this is an encoder-decoder model.
                If True, generate() will raise NotImplementedError unless overridden.
            **kwargs: Additional model-specific arguments
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.is_encoder_decoder = is_encoder_decoder
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

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging.

        Returns:
            Dictionary containing at least:
                - 'model_type': str - Name of the model architecture
                - 'vocab_size': int - Vocabulary size
                - 'parameters': int - Number of trainable parameters
        """
        pass

    def _init_weights(self) -> None:
        """Initialize model weights.

        Override this method in subclasses to provide custom initialization.
        Called automatically if defined in subclass __init__.
        """
        pass

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        This default implementation is for decoder-only models. Encoder-decoder
        models should override this method with their own generation logic.

        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            eos_token_id: If set, stop generation when this token is produced

        Returns:
            Generated token IDs [batch_size, generated_length]

        Raises:
            NotImplementedError: If called on an encoder-decoder model that
                hasn't overridden this method.
        """
        if self.is_encoder_decoder:
            raise NotImplementedError(
                f"{self.__class__.__name__} is an encoder-decoder model. "
                "Override generate() with architecture-specific generation logic."
            )

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

            # Check for EOS token
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

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
            "is_encoder_decoder": self.is_encoder_decoder,
            "model_class": self.__class__.__name__,
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

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        device: Optional[Union[str, torch.device]] = None,
        **override_kwargs
    ) -> "BaseLanguageModel":
        """
        Reconstruct model from a checkpoint file.

        This class method creates a new model instance and loads the saved
        state dict. Use this when you want to load a model without having
        to manually specify all constructor arguments.

        Args:
            path: Path to checkpoint file
            device: Device to load model onto (default: CPU)
            **override_kwargs: Arguments to override from saved config

        Returns:
            Loaded model instance

        Example:
            >>> model = SimpleLSTM.from_checkpoint("model.pt")
            >>> model = SimpleLSTM.from_checkpoint("model.pt", device="cuda")
        """
        path = Path(path)
        checkpoint = torch.load(path, map_location=device or "cpu")

        # Build constructor arguments from checkpoint
        config = checkpoint.get("model_config", {}).copy()
        config.update(override_kwargs)

        # Handle is_encoder_decoder for backwards compatibility
        is_encoder_decoder = checkpoint.get("is_encoder_decoder", False)

        # Create model instance
        model = cls(
            vocab_size=checkpoint["vocab_size"],
            is_encoder_decoder=is_encoder_decoder,
            **config
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Move to device if specified
        if device is not None:
            model = model.to(device)

        print(f"✓ Model loaded from checkpoint: {path}")
        return model

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "is_encoder_decoder": self.is_encoder_decoder,
            **self.model_config,
        }
