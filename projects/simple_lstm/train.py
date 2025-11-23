#!/usr/bin/env python3
"""Training script for Simple LSTM model on TinyStories."""

import sys
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from tqdm import tqdm
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
    """Tokenize dataset and prepare for training."""

    def tokenize_function(examples):
        # Tokenize text
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # For language modeling, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].clone()

        return tokenized

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset.column_names
    )

    tokenized_dataset.set_format(type="torch", columns=["input_ids", "labels"])

    return tokenized_dataset


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move to device
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / num_batches


def main():
    # Load config
    config = load_config()
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading TinyStories dataset...")
    dataset_path = get_datasets_dir() / "roneneldan" / "TinyStories"
    dataset = load_from_disk(dataset_path)

    # Use subset for quick testing
    if config["data"].get("use_subset", True):
        subset_size = config["data"].get("subset_size", 10000)
        dataset["train"] = dataset["train"].select(range(subset_size))
        dataset["validation"] = dataset["validation"].select(range(1000))
        print(f"Using subset: {subset_size} training examples")

    # Tokenize
    print("Tokenizing...")
    train_dataset = prepare_dataset(
        dataset["train"], tokenizer, max_length=config["data"]["max_length"]
    )
    val_dataset = prepare_dataset(
        dataset["validation"], tokenizer, max_length=config["data"]["max_length"]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Create model
    print("\nInitializing model...")
    model = SimpleLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=config["model"]["embedding_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    model = model.to(device)

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model info: {model.get_model_info()}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    print(f"\nTraining for {config['training']['num_epochs']} epochs...")

    experiment_results = []
    evaluator = Evaluator(model, device)

    for epoch in range(config["training"]["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate
        val_metrics = evaluator.evaluate(val_loader, max_batches=50)
        print(f"Val loss: {val_metrics['loss']:.4f}")
        print(f"Val perplexity: {val_metrics['perplexity']:.2f}")

        # Save metrics
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_perplexity": val_metrics["perplexity"],
        }
        experiment_results.append(epoch_results)

        # Save checkpoint
        if (epoch + 1) % config["training"].get("save_every", 5) == 0:
            checkpoint_path = f"checkpoints/lstm_epoch_{epoch+1}.pt"
            model.save_checkpoint(checkpoint_path, optimizer, epoch + 1)

    # Save final checkpoint
    print("\nSaving final checkpoint...")
    model.save_checkpoint("checkpoints/lstm_final.pt", optimizer, config["training"]["num_epochs"])

    # Save experiment results
    print("\nSaving experiment results...")
    results_df = pd.DataFrame(experiment_results)
    metadata = {
        "model_type": "SimpleLSTM",
        "dataset": "TinyStories",
        **config["model"],
        **config["training"],
    }

    save_experiment("simple_lstm_tinystories", results_df, metadata=metadata)

    print("\n✓ Training complete!")
    print(f"Final validation perplexity: {val_metrics['perplexity']:.2f}")


if __name__ == "__main__":
    main()
