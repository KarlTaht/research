#!/usr/bin/env python3
"""Display model configuration and parameter summary for TorchTransformer.

Correctly accounts for weight tying between embedding and output projection.

Usage:
    python config_summary.py configs/automotive.yaml
    python config_summary.py configs/food.yaml
    python config_summary.py --vocab-size 32768 configs/tinystories.yaml
"""

import sys
from pathlib import Path
import argparse
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import TorchTransformer
from train import load_tokenizer


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_vocab_size(config: dict) -> tuple[int, str]:
    """Get vocab size from tokenizer config.

    Returns:
        Tuple of (vocab_size, tokenizer_name)
    """
    tokenizer_name = config.get('data', {}).get('tokenizer', 'combined_bpe_32768')
    tokenizer = load_tokenizer(tokenizer_name)
    return len(tokenizer), tokenizer_name


def estimate_flops_per_step(
    n_params: int,
    batch_size: int,
    seq_len: int,
) -> float:
    """Estimate FLOPs per training step (forward + backward).

    Uses the approximation: FLOPs ~ 6 * params * tokens_per_step
    - Forward pass: ~2 * params * tokens
    - Backward pass: ~4 * params * tokens (2x forward for gradients)
    """
    tokens_per_step = batch_size * seq_len
    return 6 * n_params * tokens_per_step


def estimate_runtime(
    total_flops: float,
    tflops_realistic: float = 15.0,
) -> dict:
    """Estimate training runtime.

    Args:
        total_flops: Total FLOPs for entire training run
        tflops_realistic: Realistic sustained TFLOPs (default 15 for RTX 5060 Ti)

    Returns:
        Dict with runtime estimates
    """
    seconds = total_flops / (tflops_realistic * 1e12)
    minutes = seconds / 60
    hours = minutes / 60

    return {
        'seconds': seconds,
        'minutes': minutes,
        'hours': hours,
        'formatted': f"{hours:.1f}h" if hours >= 1 else f"{minutes:.1f}m",
    }


def format_number(n: int) -> str:
    """Format number with commas and shorthand."""
    if n >= 1_000_000_000:
        return f"{n:,} ({n/1e9:.2f}B)"
    elif n >= 1_000_000:
        return f"{n:,} ({n/1e6:.2f}M)"
    elif n >= 1_000:
        return f"{n:,} ({n/1e3:.1f}K)"
    return f"{n:,}"


def compute_parameter_breakdown(model: TorchTransformer) -> tuple[dict, dict]:
    """Compute detailed parameter breakdown.

    Accounts for weight tying: embedding and output projection share weights.
    """
    d_model = model.d_model
    d_ffn = model.d_ffn
    n_blocks = model.n_blocks
    vocab_size = model.vocab_size
    max_seq_len = model.max_seq_len

    # Per-component breakdown
    breakdown = {
        'Embeddings': {
            'token_embedding': vocab_size * d_model,
            'pos_embedding': max_seq_len * d_model,
        },
        'Attention (per block)': {
            'Q': d_model * d_model,
            'K': d_model * d_model,
            'V': d_model * d_model,
            'W_o': d_model * d_model,
        },
        'FFN (per block)': {
            'w1': d_model * d_ffn,
            'w2': d_ffn * d_model,
        },
        'RMSNorm (per block)': {
            'attention_pre_norm': d_model,
            'ffn_pre_norm': d_model,
        },
        'Output': {
            'final_norm': d_model,
            'output_projection': 0,  # Tied to token_embedding
        },
    }

    # Compute totals
    totals = {
        'Embeddings': sum(breakdown['Embeddings'].values()),
        'Attention (all blocks)': sum(breakdown['Attention (per block)'].values()) * n_blocks,
        'FFN (all blocks)': sum(breakdown['FFN (per block)'].values()) * n_blocks,
        'RMSNorm (all blocks)': sum(breakdown['RMSNorm (per block)'].values()) * n_blocks,
        'Final Norm': breakdown['Output']['final_norm'],
        'Output Projection': 0,  # Tied - no additional params
    }

    return breakdown, totals


def print_summary(config_path: str, vocab_size: int = None, tflops: float = 15.0):
    """Print model configuration and parameter summary."""
    config = load_config(config_path)

    # Auto-detect vocab size from tokenizer if not specified
    if vocab_size is None:
        vocab_size, tokenizer_name = get_vocab_size(config)
    else:
        tokenizer_name = f"manual ({vocab_size})"

    # Build model config
    model_config = {
        'vocab_size': vocab_size,
        'd_model': config['model']['d_model'],
        'n_heads': config['model']['n_heads'],
        'n_blocks': config['model']['n_blocks'],
        'd_ffn': config['model']['d_ffn'],
        'max_seq_len': config['model']['max_seq_len'],
    }

    # Create model to compute parameters
    model = TorchTransformer(model_config)

    # Use PyTorch's parameter count (correctly handles weight tying)
    total_params = sum(p.numel() for p in model.parameters())
    breakdown, totals = compute_parameter_breakdown(model)

    # Print header
    print("=" * 60)
    print("TORCHTRANSFORMER CONFIGURATION SUMMARY")
    print("=" * 60)

    print(f"Config: {config_path}")
    print()

    # Model architecture
    print("Architecture:")
    print("-" * 40)
    print(f"  {'vocab_size':<20} {vocab_size:>15,}")
    print(f"  {'tokenizer':<20} {tokenizer_name:>15}")
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
        if category == 'Output Projection':
            # Special handling for tied weights
            print(f"  {category:<25} {'(tied to embedding)':>18}")
        else:
            pct = 100 * params / total_params
            print(f"  {category:<25} {params:>12,} ({pct:>5.1f}%)")

    print("-" * 40)
    print(f"  {'TOTAL (unique params)':<25} {total_params:>12,}")
    print()

    # Formatted total
    print(f"Total Parameters: {format_number(total_params)}")
    print()

    # Weight tying note
    print("Note: Output projection shares weights with token embedding")
    print(f"      (saves {vocab_size * config['model']['d_model']:,} parameters)")
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
        corpus_dir = config['data'].get('corpus_dir', config['data'].get('dataset', 'N/A'))
        print(f"  {'corpus/dataset':<20} {corpus_dir}")
        print(f"  {'max_length':<20} {config['data'].get('max_length', 'N/A'):>15}")
        subset = config['data'].get('subset_size')
        print(f"  {'subset_size':<20} {subset if subset else 'full dataset':>15}")
        print()

    # FLOP estimation if we have training config
    if 'training' in config:
        batch_size = config['training'].get('batch_size', 32)
        seq_len = config['data'].get('max_length', 256)
        num_epochs = config['training'].get('num_epochs', 1)
        subset_size = config['data'].get('subset_size')

        # Estimate dataset size
        if subset_size:
            dataset_size = subset_size
        else:
            dataset_size = 100_000  # Conservative estimate for domain corpus

        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch * num_epochs

        flops_per_step = estimate_flops_per_step(total_params, batch_size, seq_len)
        total_flops = flops_per_step * total_steps
        runtime = estimate_runtime(total_flops, tflops)

        print("Compute Estimation:")
        print("-" * 40)
        print(f"  {'FLOPs/step':<20} {flops_per_step/1e9:>12.2f} GFLOPs")
        print(f"  {'Steps/epoch':<20} {steps_per_epoch:>15,}")
        print(f"  {'Total steps':<20} {total_steps:>15,}")
        print(f"  {'Total FLOPs':<20} {total_flops/1e12:>12.2f} TFLOPs")
        print()
        print(f"  Runtime @ {tflops:.0f} TFLOP/s:  ~{runtime['formatted']}")
        print()

    print("=" * 60)

    return total_params


def main():
    parser = argparse.ArgumentParser(
        description='Display TorchTransformer configuration and parameter summary',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python config_summary.py configs/automotive.yaml
  python config_summary.py configs/food.yaml
  python config_summary.py --vocab-size 32000 configs/tinystories.yaml
  python config_summary.py --tflops 20 configs/automotive.yaml
        """
    )
    parser.add_argument('config', help='Path to config.yaml')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='Override vocabulary size (default: auto-detect from tokenizer)')
    parser.add_argument('--tflops', type=float, default=15.0,
                        help='Realistic sustained TFLOP/s for runtime estimate (default: 15)')
    args = parser.parse_args()

    print_summary(args.config, args.vocab_size, args.tflops)


if __name__ == '__main__':
    main()
