#!/usr/bin/env python3
"""Evaluation script for Simple LSTM model."""

import sys
from pathlib import Path
import yaml
import torch
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from torch.utils.data import DataLoader
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model import SimpleLSTM
from common.data import get_datasets_dir
from common.training import Evaluator
from common.utils import save_experiment


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dataset(dataset, tokenizer, max_length=128):
    """Tokenize dataset and prepare for evaluation."""

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    return tokenized_dataset


def main():
    # Load config
    config = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print("Loading model checkpoint...")
    checkpoint_path = "checkpoints/lstm_final.pt"

    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first by running: python train.py")
        sys.exit(1)

    model = SimpleLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )

    checkpoint_info = model.load_checkpoint(checkpoint_path)
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")

    # Load dataset
    print("\nLoading TinyStories validation set...")
    dataset_path = get_datasets_dir() / "roneneldan" / "TinyStories"
    dataset = load_from_disk(dataset_path)

    # Prepare validation data
    val_dataset = prepare_dataset(
        dataset["validation"], tokenizer, max_length=config["data"]["max_length"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Evaluate
    print("\nEvaluating...")
    evaluator = Evaluator(model, device)
    metrics = evaluator.evaluate(val_loader)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.2f}")
    print(f"Tokens evaluated: {metrics['num_tokens']:,}")
    print("=" * 50)

    # Generate samples
    print("\nGenerating sample outputs...")
    prompts = [
        "Once upon a time",
        "The little girl",
        "In a magical forest",
    ]

    samples = evaluator.generate_samples(prompts, tokenizer, max_length=50, temperature=0.8)

    print("\nSample Generations:")
    print("-" * 50)
    for i, sample in enumerate(samples, 1):
        print(f"\n{i}. Prompt: {sample['prompt']}")
        print(f"   Generated: {sample['generated']}")

    # Save evaluation results
    print("\nSaving evaluation results...")
    results_df = pd.DataFrame([{
        "split": "validation",
        "loss": metrics["loss"],
        "perplexity": metrics["perplexity"],
        "num_tokens": metrics["num_tokens"],
    }])

    metadata = {
        "model_type": "SimpleLSTM",
        "dataset": "TinyStories",
        "checkpoint": str(checkpoint_path),
        **config["model"],
    }

    save_experiment("simple_lstm_eval", results_df, metadata=metadata)

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
