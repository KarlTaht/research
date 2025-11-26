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
    vocab_size = checkpoint.get('vocab_size')
    model_config = checkpoint.get('model_config', {})

    # If model_config is empty (old checkpoint), infer from state_dict
    if not model_config:
        print("Warning: Empty model_config, inferring from state_dict...")
        state_dict = checkpoint['model_state_dict']

        # Infer d_model from embedding shape
        d_model = state_dict['embedding'].shape[1]

        # Infer d_ff from encoder layer 0 feedforward W1 shape
        d_ff = state_dict['encoder.layers.0.feed_forward.W1'].shape[1]

        # Infer max_seq_len from positional encoding buffer shape
        # pos_encoder.pe has shape [1, max_seq_len, d_model]
        max_seq_len = state_dict['pos_encoder.pe'].shape[1]

        # Infer n_heads from d_model (assuming d_model is divisible by n_heads)
        # Check W_Q shape to determine head configuration
        # For multi-head attention, W_Q has shape [d_model, d_model]
        # We need to guess n_heads - common values are 4, 8, 16
        for n_heads in [4, 8, 16, 1]:
            if d_model % n_heads == 0:
                break

        # Count encoder and decoder layers
        n_encoder_layers = max(
            int(k.split('.')[2]) + 1
            for k in state_dict.keys()
            if k.startswith('encoder.layers.')
        )
        n_decoder_layers = max(
            int(k.split('.')[2]) + 1
            for k in state_dict.keys()
            if k.startswith('decoder.layers.')
        )

        model_config = {
            'd_model': d_model,
            'n_heads': n_heads,
            'd_ff': d_ff,
            'n_encoder_layers': n_encoder_layers,
            'n_decoder_layers': n_decoder_layers,
            'dropout': 0.1,  # Default (can't be inferred)
            'max_seq_len': max_seq_len,
        }

        print(f"Inferred config: {model_config}")

    # Initialize model
    model = ReferenceTransformer(vocab_size=vocab_size, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded successfully")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Loss: {checkpoint.get('loss', 'unknown')}")

    # Display training config if available
    if 'training_config' in checkpoint:
        print(f"\nTraining config found:")
        training_config = checkpoint['training_config']
        print(f"  Learning rate: {training_config.get('training', {}).get('learning_rate', 'N/A')}")
        print(f"  Batch size: {training_config.get('training', {}).get('batch_size', 'N/A')}")
        print(f"  Warmup steps: {training_config.get('training', {}).get('warmup_steps', 'N/A')}")
        print(f"  Max steps: {training_config.get('training', {}).get('max_steps', 'N/A')}")

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

    # For encoder-decoder generation, we need to seed the target with a BOS token
    # We'll use the EOS token as BOS (common practice)
    bos_token_id = tokenizer.eos_token_id

    # Start with BOS token as the seed for generation
    generated_summary_ids = [bos_token_id]

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
            logits = outputs['logits']  # [1, target_len, vocab_size]

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

    # Decode generated summary (skip the initial BOS token)
    summary = tokenizer.decode(generated_summary_ids[1:], skip_special_tokens=True)

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
