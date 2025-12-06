#!/usr/bin/env python3
"""Evaluation script for CustomTransformer."""

import sys
from pathlib import Path
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models.custom_transfromer.wrapper import CustomTransformerWrapper
from common.data import get_datasets_dir
from train import prepare_dataset, load_config


def evaluate_full(model, dataloader):
    """Full evaluation with perplexity."""
    total_loss = 0.0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc='Evaluating')

    for batch in progress_bar:
        input_ids = batch['input_ids']
        labels = batch['labels']

        with torch.no_grad():
            outputs = model.forward(input_ids, labels=labels)

        batch_tokens = input_ids.numel()
        total_loss += outputs['loss'].item() * batch_tokens
        total_tokens += batch_tokens

        current_loss = total_loss / total_tokens
        progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_tokens': total_tokens,
    }


def generate_samples(model, tokenizer, prompts, max_length=100):
    """Generate text samples from prompts."""
    samples = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors='pt')

        generated_ids = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.8,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        samples.append({
            'prompt': prompt,
            'generated': generated_text,
        })

    return samples


def main():
    parser = argparse.ArgumentParser(description='Evaluate CustomTransformer')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--generate', action='store_true', help='Generate sample text')
    parser.add_argument('--max-batches', type=int, default=None, help='Limit evaluation batches')
    args = parser.parse_args()

    config = load_config(args.config)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = CustomTransformerWrapper(
        vocab_size=len(tokenizer),
        max_seq_len=config['model']['max_seq_len'],
        n_blocks=config['model']['n_blocks'],
        n_heads=config['model']['n_heads'],
        d_model=config['model']['d_model'],
        d_ffn=config['model']['d_ffn'],
    )
    model.load_checkpoint(args.checkpoint)

    print(f"Model loaded: {model.count_parameters():,} parameters")

    # Load test dataset
    dataset_path = get_datasets_dir() / 'nampdn-ai' / 'tiny-textbooks'
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        print("Run train.py first to download the dataset.")
        return

    dataset = load_from_disk(dataset_path)

    # Use test split if available, else validation, else last 10% of train
    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        if 'test' in dataset:
            test_data = dataset['test']
        elif 'validation' in dataset:
            test_data = dataset['validation']
        elif 'train' in dataset:
            n = len(dataset['train'])
            test_data = dataset['train'].select(range(int(n * 0.9), n))
        else:
            first_key = list(dataset.keys())[0]
            n = len(dataset[first_key])
            test_data = dataset[first_key].select(range(int(n * 0.9), n))
    else:
        n = len(dataset)
        test_data = dataset.select(range(int(n * 0.9), n))

    print(f"Test set size: {len(test_data)}")

    test_dataset = prepare_dataset(
        test_data,
        tokenizer,
        config['data']['max_length'],
        config['data'].get('text_column', 'textbook'),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
    )

    # Evaluate
    print("\nEvaluating...")
    if args.max_batches:
        # Limited evaluation
        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= args.max_batches:
                break
            with torch.no_grad():
                outputs = model.forward(batch['input_ids'], labels=batch['labels'])
            total_loss += outputs['loss'].item()
            num_batches += 1
        metrics = {
            'loss': total_loss / num_batches,
            'perplexity': torch.exp(torch.tensor(total_loss / num_batches)).item(),
            'num_tokens': num_batches * config['training']['batch_size'] * config['data']['max_length'],
        }
    else:
        metrics = evaluate_full(model, test_loader)

    print(f"\nResults:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Tokens: {metrics['num_tokens']:,}")

    # Generate samples if requested
    if args.generate:
        print("\nGenerating samples...")
        prompts = [
            "The fundamental concept of",
            "In mathematics,",
            "Chapter 1:",
        ]

        samples = generate_samples(model, tokenizer, prompts, max_length=100)

        for sample in samples:
            print(f"\nPrompt: {sample['prompt']}")
            print(f"Generated: {sample['generated'][:200]}...")


if __name__ == '__main__':
    main()
