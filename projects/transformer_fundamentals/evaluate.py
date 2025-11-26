#!/usr/bin/env python
"""Evaluation script for ReferenceTransformer on SAMSum.

Loads a trained checkpoint and generates summaries for dialogue examples.

Usage:
    python evaluate.py --checkpoint ../assets/models/transformer_samsum_final.pt
    python evaluate.py --checkpoint <path> --split test --num-examples 10
"""

import argparse
import sys
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
import yaml

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models import ReferenceTransformer
from common.data import get_datasets_dir


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get('model_config', {})

    # Initialize model
    model = ReferenceTransformer(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'unknown')}")

    return model, model_config


def generate_summary(
    model,
    tokenizer,
    dialogue: str,
    max_length: int = 64,
    temperature: float = 0.8,
    device: torch.device = torch.device('cpu')
):
    """Generate summary for a dialogue using the encoder-decoder model.

    Args:
        model: ReferenceTransformer model
        tokenizer: Tokenizer
        dialogue: Input dialogue text
        max_length: Maximum summary length
        temperature: Sampling temperature
        device: Device to run on

    Returns:
        Generated summary text
    """
    model.eval()

    # Tokenize dialogue (source)
    dialogue_ids = tokenizer.encode(dialogue, add_special_tokens=False)

    # Get SEP token ID
    sep_token_id = model.sep_token_id

    # For generation, we need to provide source + SEP + start of target
    # Start with just the dialogue + SEP, then generate summary autoregressively
    # Note: This is simplified - full implementation would use beam search

    # Pack: [dialogue] [SEP]
    input_ids = dialogue_ids + [sep_token_id]

    # Convert to tensor
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

    # Generate summary tokens one by one
    generated_summary_ids = []

    with torch.no_grad():
        for _ in range(max_length):
            # Current input: [dialogue] [SEP] [generated_so_far]
            current_input = dialogue_ids + [sep_token_id] + generated_summary_ids

            # Truncate if too long
            if len(current_input) > model.max_seq_len:
                break

            current_input_tensor = torch.tensor([current_input], dtype=torch.long).to(device)

            # Forward pass
            outputs = model(current_input_tensor)
            logits = outputs['logits']  # [1, seq_len, vocab_size]

            # Get logits for last generated token (in target sequence)
            last_token_logits = logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            last_token_logits = last_token_logits / temperature

            # Sample from distribution
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop if we generate EOS or pad token
            if next_token in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break

            generated_summary_ids.append(next_token)

    # Decode generated summary
    summary = tokenizer.decode(generated_summary_ids, skip_special_tokens=True)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReferenceTransformer on SAMSum")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'validation', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--num-examples', type=int, default=5, help='Number of examples to show')
    parser.add_argument('--max-length', type=int, default=64, help='Maximum summary length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to run on')
    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model from checkpoint
    model, model_config = load_model_from_checkpoint(args.checkpoint, device)

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"Loading SAMSum {args.split} set...")
    dataset_path = get_datasets_dir() / 'samsum'
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Run: python tools/download_hf_dataset.py --name samsum"
        )

    dataset = load_from_disk(dataset_path)
    split_data = dataset[args.split]

    print(f"Dataset loaded: {len(split_data)} examples")

    # Generate summaries for examples
    print(f"\nGenerating summaries for {args.num_examples} examples...\n")
    print("=" * 80)

    for i in range(min(args.num_examples, len(split_data))):
        example = split_data[i]
        dialogue = example['dialogue']
        reference_summary = example['summary']

        print(f"\nExample {i + 1}:")
        print("-" * 80)
        print(f"DIALOGUE:")
        print(dialogue)
        print(f"\nREFERENCE SUMMARY:")
        print(reference_summary)

        # Generate summary
        print(f"\nGENERATED SUMMARY:")
        try:
            generated_summary = generate_summary(
                model, tokenizer, dialogue,
                max_length=args.max_length,
                temperature=args.temperature,
                device=device
            )
            print(generated_summary)
        except Exception as e:
            print(f"Error generating summary: {e}")

        print("=" * 80)

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
