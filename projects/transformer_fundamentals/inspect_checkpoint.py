#!/usr/bin/env python
"""Utility script to inspect checkpoint contents.

Usage:
    python inspect_checkpoint.py path/to/checkpoint.pt
    python inspect_checkpoint.py --full path/to/checkpoint.pt  # Show full config
"""

import argparse
import sys
from pathlib import Path
import torch
from pprint import pprint

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def format_size(num_params):
    """Format parameter count in human-readable format."""
    if num_params >= 1e9:
        return f"{num_params/1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)


def count_parameters(state_dict):
    """Count total parameters in state dict."""
    total = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
    return total


def inspect_checkpoint(checkpoint_path: str, show_full: bool = False):
    """Inspect checkpoint file and display information."""
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"File size: {checkpoint_path.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 70)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Basic info
    print("\n📋 BASIC INFO")
    print("-" * 70)
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Loss: {checkpoint.get('loss', 'N/A')}")

    # Model config
    print("\n🏗️  MODEL ARCHITECTURE")
    print("-" * 70)
    vocab_size = checkpoint.get('vocab_size', 'N/A')
    print(f"Vocab size: {vocab_size}")

    model_config = checkpoint.get('model_config', {})
    if model_config:
        print(f"d_model: {model_config.get('d_model', 'N/A')}")
        print(f"n_heads: {model_config.get('n_heads', 'N/A')}")
        print(f"n_encoder_layers: {model_config.get('n_encoder_layers', 'N/A')}")
        print(f"n_decoder_layers: {model_config.get('n_decoder_layers', 'N/A')}")
        print(f"d_ff: {model_config.get('d_ff', 'N/A')}")
        print(f"dropout: {model_config.get('dropout', 'N/A')}")
        print(f"max_seq_len: {model_config.get('max_seq_len', 'N/A')}")
    else:
        print("⚠️  Model config is empty (old checkpoint format)")
        # Try to infer
        state_dict = checkpoint.get('model_state_dict', {})
        if 'embedding' in state_dict:
            d_model = state_dict['embedding'].shape[1]
            print(f"d_model (inferred): {d_model}")

    # Parameter count
    state_dict = checkpoint.get('model_state_dict', {})
    if state_dict:
        num_params = count_parameters(state_dict)
        print(f"\nTotal parameters: {format_size(num_params)} ({num_params:,})")

    # Training config
    training_config = checkpoint.get('training_config')
    if training_config:
        print("\n⚙️  TRAINING CONFIG")
        print("-" * 70)

        if 'training' in training_config:
            t = training_config['training']
            print(f"Learning rate: {t.get('learning_rate', 'N/A')}")
            print(f"Batch size: {t.get('batch_size', 'N/A')}")
            print(f"Warmup steps: {t.get('warmup_steps', 'N/A')}")
            print(f"Max steps: {t.get('max_steps', 'N/A')}")
            print(f"Gradient clip: {t.get('gradient_clip_val', 'N/A')}")

        if 'data' in training_config:
            d = training_config['data']
            print(f"\nDataset: {d.get('dataset_name', 'N/A')}")
            print(f"Max dialogue length: {d.get('max_dialogue_length', 'N/A')}")
            print(f"Max summary length: {d.get('max_summary_length', 'N/A')}")

        if show_full:
            print("\n📄 FULL TRAINING CONFIG")
            print("-" * 70)
            pprint(training_config, width=70)
    else:
        print("\n⚠️  No training config found (checkpoint saved before this feature)")

    # Optimizer state
    if 'optimizer_state_dict' in checkpoint:
        print("\n✓ Optimizer state present")
    else:
        print("\n✗ No optimizer state")

    # Keys in checkpoint
    print("\n🔑 CHECKPOINT KEYS")
    print("-" * 70)
    for key in checkpoint.keys():
        if key == 'model_state_dict':
            print(f"  {key}: {len(checkpoint[key])} tensors")
        elif key == 'optimizer_state_dict':
            print(f"  {key}: present")
        else:
            print(f"  {key}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint contents")
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--full', action='store_true',
                       help='Show full training config')
    args = parser.parse_args()

    inspect_checkpoint(args.checkpoint, show_full=args.full)


if __name__ == '__main__':
    main()
