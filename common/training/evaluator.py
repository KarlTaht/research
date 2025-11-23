"""Evaluation framework for language models."""

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm


class Evaluator:
    """Standard evaluator for language models.

    Computes common metrics like perplexity, loss, and can generate
    sample outputs for qualitative assessment.
    """

    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize evaluator.

        Args:
            model: Language model to evaluate (should have forward() method)
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device
        self.model.to(device)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        max_batches: Optional[int] = None,
        return_samples: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader with evaluation data
            max_batches: Optional limit on number of batches to evaluate
            return_samples: If True, include sample generations in results

        Returns:
            Dictionary with metrics:
                - loss: Average loss
                - perplexity: Perplexity score
                - num_batches: Number of batches evaluated
                - num_tokens: Total tokens evaluated
                - samples (optional): Generated text samples
        """
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break

            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            # Forward pass
            outputs = self.model(input_ids, labels=labels)

            # Accumulate loss
            loss = outputs["loss"]
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            batch_tokens = batch_size * seq_len

            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

            # Update progress
            current_loss = total_loss / total_tokens
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        # Compute final metrics
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        results = {
            "loss": avg_loss,
            "perplexity": perplexity,
            "num_batches": num_batches,
            "num_tokens": total_tokens,
        }

        return results

    @torch.no_grad()
    def generate_samples(
        self,
        prompts: List[str],
        tokenizer: Any,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
    ) -> List[Dict[str, str]]:
        """
        Generate text samples from prompts.

        Args:
            prompts: List of text prompts
            tokenizer: Tokenizer for encoding/decoding
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering

        Returns:
            List of dicts with 'prompt' and 'generated' keys
        """
        self.model.eval()
        samples = []

        for prompt in prompts:
            # Tokenize prompt
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            # Generate
            if hasattr(self.model, "generate"):
                generated_ids = self.model.generate(
                    input_ids, max_length=max_length, temperature=temperature, top_k=top_k
                )
            else:
                # Fallback: manual generation
                generated_ids = self._generate_fallback(
                    input_ids, max_length, temperature, top_k
                )

            # Decode
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            samples.append({"prompt": prompt, "generated": generated_text})

        return samples

    def _generate_fallback(
        self, input_ids: torch.Tensor, max_length: int, temperature: float, top_k: Optional[int]
    ) -> torch.Tensor:
        """Fallback generation for models without generate() method."""
        generated = input_ids

        for _ in range(max_length - input_ids.size(1)):
            outputs = self.model(generated)
            logits = outputs["logits"]
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[
                    0
                ][..., -1, None]
                next_token_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def create_metrics_dataframe(
        self, metrics: Dict[str, Any], epoch: Optional[int] = None, split: str = "test"
    ) -> pd.DataFrame:
        """
        Convert metrics to pandas DataFrame for experiment storage.

        Args:
            metrics: Dictionary of metrics from evaluate()
            epoch: Optional epoch number
            split: Data split name ('train', 'val', 'test')

        Returns:
            DataFrame with one row per metric
        """
        data = {
            "split": split,
            "loss": metrics["loss"],
            "perplexity": metrics["perplexity"],
            "num_tokens": metrics["num_tokens"],
        }

        if epoch is not None:
            data["epoch"] = epoch

        return pd.DataFrame([data])


def compute_perplexity(model: nn.Module, dataloader: DataLoader, device: str = "cuda") -> float:
    """
    Quick utility to compute perplexity.

    Args:
        model: Language model
        dataloader: DataLoader with evaluation data
        device: Device to use

    Returns:
        Perplexity score
    """
    evaluator = Evaluator(model, device)
    results = evaluator.evaluate(dataloader)
    return results["perplexity"]
