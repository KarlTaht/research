#!/usr/bin/env python3
"""Display model configuration and parameter summary.

Usage:
    python config_summary.py configs/tiny-textbooks-small.yaml
    python config_summary.py  # Uses default config
"""

import sys
from pathlib import Path
import argparse
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models.custom_transfromer import CustomTransformerWrapper


def load_config(config_path: str = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "tiny-textbooks-small.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_number(n: int) -> str:
    """Format number with commas and shorthand."""
    if n >= 1_000_000_000:
        return f"{n:,} ({n/1e9:.2f}B)"
    elif n >= 1_000_000:
        return f"{n:,} ({n/1e6:.2f}M)"
    elif n >= 1_000:
        return f"{n:,} ({n/1e3:.1f}K)"
    return f"{n:,}"


def compute_parameter_breakdown(model) -> dict:
    """Compute detailed parameter breakdown."""
    m = model.model

    breakdown = {
        'Embeddings': {
            'vocab_embedding': m.vocab_embedding.numel(),
            'pos_embedding': m.pos_embedding.numel(),
        },
        'Attention (per block)': {
            'Q': m.Q.numel() // m.n_blocks,
            'K': m.K.numel() // m.n_blocks,
            'V': m.V.numel() // m.n_blocks,
            'W_o': m.W_o.numel() // m.n_blocks,
        },
        'FFN (per block)': {
            'W1': m.W1.numel() // m.n_blocks,
            'W2': m.W2.numel() // m.n_blocks,
        },
        'Layer Norm (per block)': {
            'attention_gamma': m.attention_gamma.numel() // m.n_blocks,
            'attention_beta': m.attention_beta.numel() // m.n_blocks,
            'ffn_gamma': m.ffn_gamma.numel() // m.n_blocks,
            'ffn_beta': m.ffn_beta.numel() // m.n_blocks,
        },
        'Output': {
            'output_projection': m.output_projection.numel(),
        },
    }

    # Compute totals
    totals = {
        'Embeddings': sum(breakdown['Embeddings'].values()),
        'Attention (all blocks)': sum(breakdown['Attention (per block)'].values()) * m.n_blocks,
        'FFN (all blocks)': sum(breakdown['FFN (per block)'].values()) * m.n_blocks,
        'Layer Norm (all blocks)': sum(breakdown['Layer Norm (per block)'].values()) * m.n_blocks,
        'Output': sum(breakdown['Output'].values()),
    }

    return breakdown, totals


def print_summary(config_path: str, vocab_size: int = 50257):
    """Print model configuration and parameter summary."""
    config = load_config(config_path)

    # Create model to compute parameters
    model = CustomTransformerWrapper(
        vocab_size=vocab_size,
        max_seq_len=config['model']['max_seq_len'],
        n_blocks=config['model']['n_blocks'],
        n_heads=config['model']['n_heads'],
        d_model=config['model']['d_model'],
        d_ffn=config['model']['d_ffn'],
    )

    total_params = model.count_parameters()
    breakdown, totals = compute_parameter_breakdown(model)

    # Print header
    print("=" * 60)
    print("MODEL CONFIGURATION SUMMARY")
    print("=" * 60)

    if config_path:
        print(f"Config: {config_path}")
    print()

    # Model architecture
    print("Architecture:")
    print("-" * 40)
    print(f"  {'vocab_size':<20} {vocab_size:>15,}")
    print(f"  {'max_seq_len':<20} {config['model']['max_seq_len']:>15,}")
    print(f"  {'d_model':<20} {config['model']['d_model']:>15,}")
    print(f"  {'d_ffn':<20} {config['model']['d_ffn']:>15,}")
    print(f"  {'n_blocks':<20} {config['model']['n_blocks']:>15,}")
    print(f"  {'n_heads':<20} {config['model']['n_heads']:>15,}")
    print(f"  {'d_head':<20} {config['model']['d_model'] // config['model']['n_heads']:>15,}")
    print()

    # Parameter breakdown
    print("Parameter Breakdown:")
    print("-" * 40)

    for category, params in totals.items():
        pct = 100 * params / total_params
        print(f"  {category:<25} {params:>12,} ({pct:>5.1f}%)")

    print("-" * 40)
    print(f"  {'TOTAL':<25} {total_params:>12,}")
    print()

    # Formatted total
    print(f"Total Parameters: {format_number(total_params)}")
    print()

    # Training config if present
    if 'training' in config:
        print("Training Config:")
        print("-" * 40)
        print(f"  {'batch_size':<20} {config['training'].get('batch_size', 'N/A'):>15}")
        print(f"  {'learning_rate':<20} {config['training'].get('learning_rate', 'N/A'):>15}")
        print(f"  {'num_epochs':<20} {config['training'].get('num_epochs', 'N/A'):>15}")
        print()

    # Data config if present
    if 'data' in config:
        print("Data Config:")
        print("-" * 40)
        print(f"  {'dataset':<20} {config['data'].get('dataset', 'N/A')}")
        print(f"  {'max_length':<20} {config['data'].get('max_length', 'N/A'):>15}")
        print(f"  {'subset_size':<20} {config['data'].get('subset_size', 'N/A'):>15}")
        print()

    print("=" * 60)

    return total_params


def main():
    parser = argparse.ArgumentParser(
        description='Display model configuration and parameter summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python config_summary.py configs/tiny-textbooks-small.yaml
  python config_summary.py --vocab-size 32000 configs/my-config.yaml
        """
    )
    parser.add_argument('config', nargs='?', default=None, help='Path to config.yaml')
    parser.add_argument('--vocab-size', type=int, default=50257,
                        help='Vocabulary size (default: 50257 for GPT-2)')
    args = parser.parse_args()

    print_summary(args.config, args.vocab_size)


if __name__ == '__main__':
    main()
