#!/usr/bin/env python3
"""Evaluation script for CustomTransformer.

Usage:
    # Evaluate a checkpoint
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/tinystories.yaml

    # Evaluate with text generation
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/tinystories.yaml --generate

    # Quick evaluation (limited batches)
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/tinystories.yaml --max-batches 50

    # Evaluate with custom tokenizer
    python evaluate.py --checkpoint checkpoints/best.pt --config configs/tinystories_custom_tokenizer.yaml --generate
"""

import sys
from pathlib import Path
import argparse
import yaml
import torch
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models.custom_transfromer.wrapper import CustomTransformerWrapper
from common.data import load_training_data, get_dataset_config, get_models_dir


def load_config(config_path: str = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "tinystories.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tokenizer(config: dict):
    """Load tokenizer based on config.

    Supports:
    - GPT-2 tokenizer (default): tokenizer: "gpt2" or omitted
    - Custom tokenizer path: tokenizer: "path/to/tokenizer" or "tokenizers/name"

    Args:
        config: Configuration dict with optional 'tokenizer' key in 'data' section

    Returns:
        Loaded tokenizer with pad_token set
    """
    tokenizer_config = config.get('data', {}).get('tokenizer', 'gpt2')

    if tokenizer_config == 'gpt2':
        print("  Using GPT-2 tokenizer (vocab_size=50257)")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Custom tokenizer - check if it's a relative path under assets/models/
        tokenizer_path = Path(tokenizer_config)
        if not tokenizer_path.is_absolute():
            # Try assets/models/tokenizers/ first
            assets_path = get_models_dir() / 'tokenizers' / tokenizer_config
            if assets_path.exists():
                tokenizer_path = assets_path
            else:
                # Try assets/models/ directly
                assets_path = get_models_dir() / tokenizer_config
                if assets_path.exists():
                    tokenizer_path = assets_path

        print(f"  Using custom tokenizer: {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

        # Ensure pad_token is set (custom tokenizers should have it, but fallback)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '<pad>'})

    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer


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

    # Load tokenizer (supports GPT-2 or custom tokenizers)
    print("Loading tokenizer...")
    tokenizer = load_tokenizer(config)

    # Load model from checkpoint
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

    # Load validation data using dataset registry
    dataset_name = config['data']['dataset']
    print(f"\nLoading dataset: {dataset_name}")
    dataset_config = get_dataset_config(dataset_name)
    print(f"  Description: {dataset_config['description']}")

    # Load validation split for evaluation
    _, val_loader = load_training_data(
        dataset_name,
        tokenizer,
        max_length=config['data']['max_length'],
        batch_size=config['training']['batch_size'],
        subset_size=config['data'].get('subset_size'),
        val_subset_size=config['data'].get('val_subset_size', 1000),
    )

    if val_loader is None:
        print("No validation data available for this dataset.")
        return

    print(f"  Val batches: {len(val_loader)}")

    # Evaluate
    print("\nEvaluating...")
    if args.max_batches:
        # Limited evaluation
        total_loss = 0.0
        num_batches = 0
        for batch_idx, batch in enumerate(val_loader):
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
        metrics = evaluate_full(model, val_loader)

    print(f"\nResults:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    print(f"  Tokens: {metrics['num_tokens']:,}")

    # Generate samples if requested
    if args.generate:
        print("\nGenerating samples...")
        # Use prompts from config if available
        prompts = config.get('evaluation', {}).get('generation_prompts', [
            "Once upon a time",
            "The little girl",
            "One day, a boy named",
        ])

        max_gen_length = config.get('evaluation', {}).get('max_generation_length', 100)
        samples = generate_samples(model, tokenizer, prompts, max_length=max_gen_length)

        for sample in samples:
            print(f"\nPrompt: {sample['prompt']}")
            print(f"Generated: {sample['generated'][:200]}...")


if __name__ == '__main__':
    main()
