#!/usr/bin/env python3
"""Training script for CustomTransformer on tiny-textbooks."""

import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models.custom_transfromer.wrapper import CustomTransformerWrapper
from common.data import get_datasets_dir
from common.utils import save_experiment


def load_config(config_path: str = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "tiny-textbooks-small.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_dataset(dataset, tokenizer, max_length, text_column='textbook'):
    """Tokenize dataset for language modeling."""

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt',
        )
        # For language modeling, labels = input_ids
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format(type='torch', columns=['input_ids', 'labels'])
    return tokenized


def evaluate(model, dataloader, max_batches=None):
    """Evaluate model on dataloader."""
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches and batch_idx >= max_batches:
            break

        input_ids = batch['input_ids']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model.forward(input_ids, labels=labels)

        total_loss += outputs['loss'].item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {'loss': avg_loss, 'perplexity': perplexity}


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train CustomTransformer')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("Loading tiny-textbooks dataset...")
    dataset_path = get_datasets_dir() / 'nampdn-ai' / 'tiny-textbooks'

    if dataset_path.exists():
        print(f"Loading from disk: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    else:
        print("Downloading dataset...")
        dataset = load_dataset(config['data']['dataset'])
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(dataset_path)

    # Determine train/val splits
    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        if 'train' in dataset:
            train_data = dataset['train']
        else:
            # Use the first available split
            first_key = list(dataset.keys())[0]
            train_data = dataset[first_key]
    else:
        train_data = dataset

    # Use subset for faster iteration
    if config['data'].get('use_subset', True):
        subset_size = config['data'].get('subset_size', 5000)
        train_data = train_data.select(range(min(subset_size, len(train_data))))

    # Create validation set from first 10% or 500 samples
    val_size = min(500, len(train_data) // 10)
    val_data = train_data.select(range(val_size))
    print(f"Using: {len(train_data)} training, {len(val_data)} validation samples")

    # Tokenize
    print("Tokenizing...")
    train_dataset = prepare_dataset(
        train_data,
        tokenizer,
        config['data']['max_length'],
        config['data'].get('text_column', 'textbook'),
    )
    val_dataset = prepare_dataset(
        val_data,
        tokenizer,
        config['data']['max_length'],
        config['data'].get('text_column', 'textbook'),
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
    )

    # Create model
    print("\nInitializing CustomTransformer...")
    model = CustomTransformerWrapper(
        vocab_size=len(tokenizer),
        max_seq_len=config['model']['max_seq_len'],
        n_blocks=config['model']['n_blocks'],
        n_heads=config['model']['n_heads'],
        d_model=config['model']['d_model'],
        d_ffn=config['model']['d_ffn'],
    )

    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model info: {model.get_model_info()}")

    # Training loop
    print(f"\nTraining for {config['training']['num_epochs']} epochs...")

    experiment_results = []
    learning_rate = config['training']['learning_rate']
    log_every = config['training'].get('log_every', 100)
    eval_every = config['training'].get('eval_every', 500)

    global_step = 0

    for epoch in range(config['training']['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['num_epochs']}")

        epoch_losses = []
        progress_bar = tqdm(train_loader, desc='Training')

        for batch in progress_bar:
            input_ids = batch['input_ids']
            labels = batch['labels']

            # Manual training step
            result = model.train_step(input_ids, labels, learning_rate)
            epoch_losses.append(result['loss'])
            global_step += 1

            # Update progress bar
            if global_step % log_every == 0:
                recent_loss = sum(epoch_losses[-log_every:]) / min(log_every, len(epoch_losses))
                progress_bar.set_postfix({'loss': f'{recent_loss:.4f}'})

            # Periodic evaluation
            if global_step % eval_every == 0:
                val_metrics = evaluate(model, val_loader, max_batches=50)
                print(
                    f"\n  Step {global_step}: val_loss={val_metrics['loss']:.4f}, "
                    f"val_ppl={val_metrics['perplexity']:.2f}"
                )

        # End of epoch evaluation
        train_loss = sum(epoch_losses) / len(epoch_losses)
        val_metrics = evaluate(model, val_loader)

        print(
            f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_ppl={val_metrics['perplexity']:.2f}"
        )

        # Save results
        epoch_results = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_perplexity': val_metrics['perplexity'],
        }
        experiment_results.append(epoch_results)

        # Save checkpoint
        if (epoch + 1) % config['training'].get('save_every', 1) == 0:
            checkpoint_dir = Path(__file__).parent / 'checkpoints'
            checkpoint_dir.mkdir(exist_ok=True)
            checkpoint_path = checkpoint_dir / f'custom_transformer_epoch_{epoch+1}.pt'
            model.save_checkpoint(str(checkpoint_path), epoch=epoch + 1)

    # Save experiment results
    print("\nSaving experiment results...")
    results_df = pd.DataFrame(experiment_results)
    metadata = {
        'model_type': 'CustomTransformer',
        'dataset': config['data']['dataset'],
        **config['model'],
        **config['training'],
    }
    save_experiment('custom_transformer_tiny_textbooks', results_df, metadata=metadata)

    print("\nTraining complete!")
    print(f"Final validation perplexity: {val_metrics['perplexity']:.2f}")


if __name__ == '__main__':
    main()
