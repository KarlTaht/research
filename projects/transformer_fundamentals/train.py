#!/usr/bin/env python
"""Training script for ReferenceTransformer on SAMSum dialogue summarization.

This script validates the encoder-decoder ReferenceTransformer implementation
by training on the SAMSum dataset (dialogue → summary pairs).

Usage:
    python train.py --config config.yaml
    python -m projects.transformer_fundamentals.train
"""

import argparse
import sys
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models import ReferenceTransformer
from common.data import get_datasets_dir
from common.training import Evaluator
from common.utils import save_experiment


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(config: dict, tokenizer):
    """Load and preprocess SAMSum dataset."""
    print("Loading SAMSum dataset...")

    # Load from disk
    dataset_path = get_datasets_dir() / config['data']['dataset_name']
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run: python tools/download_hf_dataset.py --name samsum"
        )

    dataset = load_from_disk(dataset_path)
    print(f"Dataset loaded: {dataset}")

    # Get SEP token ID (last token in vocab)
    sep_token_id = config['model']['vocab_size'] - 1
    max_dialogue_len = config['data']['max_dialogue_length']
    max_summary_len = config['data']['max_summary_length']

    def preprocess_function(examples):
        """Tokenize and pack dialogue + summary with SEP token."""
        input_ids_list = []
        labels_list = []

        for dialogue, summary in zip(examples['dialogue'], examples['summary']):
            # Tokenize dialogue (source) and summary (target)
            dialogue_ids = tokenizer.encode(dialogue, add_special_tokens=False)
            summary_ids = tokenizer.encode(summary, add_special_tokens=False)

            # Truncate if necessary
            dialogue_ids = dialogue_ids[:max_dialogue_len]
            summary_ids = summary_ids[:max_summary_len]

            # Pack: [dialogue] [SEP] [summary]
            input_ids = dialogue_ids + [sep_token_id] + summary_ids

            # Labels are same as input_ids for teacher forcing
            labels = input_ids.copy()

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {
            'input_ids': input_ids_list,
            'labels': labels_list
        }

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )

    # Use subset if specified
    if config['data'].get('use_subset', False):
        subset_size = config['data']['subset_size']
        print(f"Using subset of {subset_size} examples...")
        tokenized_dataset['train'] = tokenized_dataset['train'].select(range(subset_size))
        val_subset = min(subset_size // 10, len(tokenized_dataset['validation']))
        tokenized_dataset['validation'] = tokenized_dataset['validation'].select(range(val_subset))

    print(f"Train size: {len(tokenized_dataset['train'])}")
    print(f"Validation size: {len(tokenized_dataset['validation'])}")

    return tokenized_dataset


def collate_fn(batch):
    """Custom collate function to pad sequences to same length in batch."""
    # Find max length in batch
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids_list = []
    labels_list = []

    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']

        # Pad to max_len with 0 (pad token)
        padding_len = max_len - len(input_ids)
        input_ids = input_ids + [0] * padding_len
        labels = labels + [0] * padding_len

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'labels': torch.tensor(labels_list, dtype=torch.long)
    }


def train_epoch(model, train_loader, optimizer, device, gradient_clip):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train ReferenceTransformer on SAMSum")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print(f"Config loaded from {args.config}")
    print(yaml.dump(config, default_flow_style=False))

    # Set random seed
    torch.manual_seed(config['training']['seed'])

    # Initialize tokenizer
    print("Initializing GPT-2 tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Vocab size: {len(tokenizer)}")

    # Load and preprocess dataset
    tokenized_dataset = prepare_dataset(config, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(
        tokenized_dataset['train'],
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        tokenized_dataset['validation'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model
    print("Initializing ReferenceTransformer...")
    model = ReferenceTransformer(
        vocab_size=config['model']['vocab_size'],
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_encoder_layers=config['model']['n_encoder_layers'],
        n_decoder_layers=config['model']['n_decoder_layers'],
        d_ff=config['model']['d_ff'],
        dropout=config['model']['dropout'],
        max_seq_len=config['model']['max_seq_len'],
        pad_token_id=tokenizer.pad_token_id
    )

    # Move model to device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print(f"\nModel info:")
    model_info = model.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    print(f"\nDevice: {device}")

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Initialize evaluator
    evaluator = Evaluator(model, device=device)

    # Training loop
    print(f"\nStarting training for {config['training']['num_epochs']} epochs...")
    results = []

    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            config['training']['gradient_clip']
        )
        print(f"Train loss: {train_loss:.4f}")

        # Validate
        print("Validating...")
        val_metrics = evaluator.evaluate(val_loader, max_batches=None)
        print(f"Validation metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Perplexity: {val_metrics['perplexity']:.2f}")
        print(f"  Tokens: {val_metrics['num_tokens']}")

        # Track results
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_perplexity': val_metrics['perplexity'],
            'num_tokens': val_metrics['num_tokens']
        })

        # Save checkpoint
        if epoch % config['training']['save_every'] == 0:
            checkpoint_dir = Path(__file__).parent.parent.parent / 'assets' / 'models'
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"transformer_samsum_epoch_{epoch}.pt"

            model.save_checkpoint(
                checkpoint_path,
                optimizer=optimizer,
                epoch=epoch,
                loss=val_metrics['loss'],
                training_config=config  # Save full training config
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint = checkpoint_dir / f"transformer_samsum_final.pt"
    model.save_checkpoint(
        final_checkpoint,
        optimizer=optimizer,
        epoch=config['training']['num_epochs'],
        loss=results[-1]['val_loss'],
        training_config=config  # Save full training config
    )
    print(f"\nFinal checkpoint saved: {final_checkpoint}")

    # Save experiment results
    print("\nSaving experiment results...")
    results_df = pd.DataFrame(results)
    metadata = {
        'model': 'ReferenceTransformer',
        'dataset': 'samsum',
        'architecture': 'encoder-decoder',
        'implementation': 'pure_tensor_ops',
        **config['model'],
        **config['training'],
        **config['experiment']
    }

    save_experiment(config['experiment']['name'], results_df, metadata=metadata)

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Final validation perplexity: {results[-1]['val_perplexity']:.2f}")
    print(f"Final validation loss: {results[-1]['val_loss']:.4f}")
    print(f"\nCheckpoint: {final_checkpoint}")
    print(f"Experiment: {config['experiment']['name']}")


if __name__ == '__main__':
    main()
