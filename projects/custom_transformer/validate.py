#!/usr/bin/env python3
"""Validation script for debugging CustomTransformer training vs evaluation.

Provides token-by-token analysis of model predictions on training data.
Useful for:
- Verifying the model is learning (overfitting check on training data)
- Debugging discrepancies between training loss and evaluation loss
- Understanding what the model predicts at each position

Usage:
    # Find "Once upon a time" examples in training data (random init)
    python validate.py --config configs/tinystories.yaml --find-prompt "Once upon a time"

    # With trained checkpoint
    python validate.py --config configs/tinystories.yaml --checkpoint checkpoints/best.pt

    # Control verbosity
    python validate.py --config configs/tinystories.yaml --max-positions 30
"""

import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn.functional as F
from typing import Generator, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from common.models.custom_transfromer.wrapper import CustomTransformerWrapper
from common.data import load_training_data, get_dataset_config, get_models_dir


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tokenizer(config: dict):
    """Load tokenizer based on config."""
    tokenizer_config = config.get('data', {}).get('tokenizer', 'gpt2')

    if tokenizer_config == 'gpt2':
        print("  Using GPT-2 tokenizer (vocab_size=50257)")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer_path = Path(tokenizer_config)
        if not tokenizer_path.is_absolute():
            assets_path = get_models_dir() / 'tokenizers' / tokenizer_config
            if assets_path.exists():
                tokenizer_path = assets_path
            else:
                assets_path = get_models_dir() / tokenizer_config
                if assets_path.exists():
                    tokenizer_path = assets_path

        print(f"  Using custom tokenizer: {tokenizer_path}")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '<pad>'})

    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer


def find_training_examples(
    train_loader,
    tokenizer,
    prefix: str,
) -> Generator[dict, None, None]:
    """Yield training examples starting with given prefix.

    Yields:
        dict with keys:
            - 'input_ids': tensor [seq_len]
            - 'labels': tensor [seq_len]
            - 'text': str (decoded text)
            - 'batch_idx': int
            - 'example_idx': int (index within batch)
            - 'batch': full batch dict for batch summary
    """
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids']  # [batch_size, seq_len]
        labels = batch['labels']

        for example_idx in range(input_ids.size(0)):
            # Decode this example
            example_ids = input_ids[example_idx]
            text = tokenizer.decode(example_ids, skip_special_tokens=True)

            # Check if it starts with the prefix
            if text.strip().startswith(prefix):
                yield {
                    'input_ids': example_ids,
                    'labels': labels[example_idx],
                    'text': text,
                    'batch_idx': batch_idx,
                    'example_idx': example_idx,
                    'batch': batch,
                }


def decode_token_readable(tokenizer, token_id: int) -> str:
    """Decode a single token to readable string with escapes for special chars."""
    text = tokenizer.decode([token_id])
    # Make whitespace visible
    text = text.replace('\n', '\\n').replace('\t', '\\t')
    return text


def compute_per_position_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Compute unreduced cross-entropy loss at each position.

    Applies the same shift as wrapper.py:104-112 (causal LM: predict next token).

    Args:
        logits: [seq_len, vocab_size] for single example
        labels: [seq_len] for single example
        vocab_size: vocabulary size

    Returns:
        per_position_loss: [seq_len-1] unreduced losses
    """
    # Shift: logits[i] predicts labels[i+1]
    shift_logits = logits[:-1, :]  # [seq_len-1, vocab_size]
    shift_labels = labels[1:]       # [seq_len-1]

    # Compute per-position cross-entropy (no reduction)
    per_position_loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        reduction='none',
    )
    return per_position_loss


def get_top_k_predictions(
    logits: torch.Tensor,
    tokenizer,
    k: int = 5,
) -> list[list[tuple]]:
    """Get top-k predictions with probabilities for each position.

    Args:
        logits: [seq_len, vocab_size]
        tokenizer: tokenizer for decoding
        k: number of top predictions

    Returns:
        List of lists, where each inner list contains (token_str, prob) tuples
    """
    probs = F.softmax(logits.float(), dim=-1)  # [seq_len, vocab_size]
    top_probs, top_indices = torch.topk(probs, k, dim=-1)  # [seq_len, k]

    results = []
    for pos in range(logits.size(0)):
        pos_preds = []
        for i in range(k):
            token_id = top_indices[pos, i].item()
            prob = top_probs[pos, i].item()
            token_str = decode_token_readable(tokenizer, token_id)
            pos_preds.append((token_str, prob, token_id))
        results.append(pos_preds)

    return results


def analyze_example(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    tokenizer,
    max_positions: int = 50,
) -> dict:
    """Full token-by-token analysis with predictions and losses.

    Args:
        model: CustomTransformerWrapper
        input_ids: [seq_len] single example
        labels: [seq_len] single example
        tokenizer: tokenizer for decoding
        max_positions: max positions to analyze

    Returns:
        dict with analysis results
    """
    # Move to model device
    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)

    # Add batch dimension for model
    input_ids_batch = input_ids.unsqueeze(0)  # [1, seq_len]

    with torch.no_grad():
        outputs = model.forward(input_ids_batch)
        logits = outputs['logits'][0]  # [seq_len, vocab_size]

    # Compute per-position loss
    per_pos_loss = compute_per_position_loss(logits, labels, model.vocab_size)

    # Get top-k predictions (for shifted positions)
    # logits[i] predicts the token at position i+1
    shift_logits = logits[:-1, :]  # [seq_len-1, vocab_size]
    top_k_preds = get_top_k_predictions(shift_logits, tokenizer, k=5)

    # Build analysis for each position
    num_positions = min(max_positions, len(per_pos_loss))
    positions = []

    correct_count = 0
    top5_correct_count = 0

    for pos in range(num_positions):
        input_token_id = input_ids[pos].item()
        target_token_id = labels[pos + 1].item()  # shifted target

        input_token_str = decode_token_readable(tokenizer, input_token_id)
        target_token_str = decode_token_readable(tokenizer, target_token_id)

        loss = per_pos_loss[pos].item()
        preds = top_k_preds[pos]  # list of (token_str, prob, token_id)

        # Check if prediction is correct
        top1_correct = (preds[0][2] == target_token_id)
        top5_correct = any(p[2] == target_token_id for p in preds)

        if top1_correct:
            correct_count += 1
        if top5_correct:
            top5_correct_count += 1

        positions.append({
            'position': pos,
            'input_token': input_token_str,
            'input_token_id': input_token_id,
            'target_token': target_token_str,
            'target_token_id': target_token_id,
            'predictions': preds,
            'loss': loss,
            'top1_correct': top1_correct,
            'top5_correct': top5_correct,
        })

    # Compute summary stats
    total_loss = per_pos_loss[:num_positions].sum().item()
    avg_loss = total_loss / num_positions
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        'positions': positions,
        'num_positions': num_positions,
        'total_loss': total_loss,
        'avg_loss': avg_loss,
        'perplexity': perplexity,
        'top1_accuracy': correct_count / num_positions,
        'top5_accuracy': top5_correct_count / num_positions,
        'correct_count': correct_count,
        'top5_correct_count': top5_correct_count,
    }


def compute_batch_summary(
    model,
    batch: dict,
    vocab_size: int,
) -> dict:
    """Compute average loss and stats across entire batch."""
    input_ids = batch['input_ids'].to(model.device)
    labels = batch['labels'].to(model.device)

    with torch.no_grad():
        outputs = model.forward(input_ids, labels=labels)
        batch_loss = outputs['loss'].item()

    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)

    return {
        'batch_loss': batch_loss,
        'batch_perplexity': torch.exp(torch.tensor(batch_loss)).item(),
        'batch_size': batch_size,
        'seq_len': seq_len,
        'total_tokens': batch_size * seq_len,
    }


def print_analysis_report(
    analysis: dict,
    example_info: dict,
    tokenizer,
    show_header: bool = True,
):
    """Pretty-print the analysis results."""
    if show_header:
        print("=" * 80)
        print("VALIDATION REPORT")
        print("=" * 80)
        print()
        print(f"Found example in batch {example_info['batch_idx']}, "
              f"example {example_info['example_idx']}")
        print()
        print("Original text (first 200 chars):")
        text_preview = example_info['text'][:200].replace('\n', ' ')
        print(f'"{text_preview}..."')
        print()

    print("=" * 80)
    print(f"TOKEN-BY-TOKEN ANALYSIS (showing {analysis['num_positions']} positions)")
    print("=" * 80)
    print()

    # Header
    print(f"{'Pos':>3} | {'Input Token':<16} | {'Target Token':<16} | "
          f"{'Top-3 Predictions':<36} | {'Loss':>6}")
    print("-" * 3 + "-+-" + "-" * 16 + "-+-" + "-" * 16 + "-+-" + "-" * 36 + "-+-" + "-" * 6)

    for pos_info in analysis['positions']:
        pos = pos_info['position']
        input_tok = repr(pos_info['input_token'])[:14]
        target_tok = repr(pos_info['target_token'])[:14]
        loss = pos_info['loss']
        correct_marker = " ok" if pos_info['top1_correct'] else ""

        # Format top-3 predictions
        preds = pos_info['predictions'][:3]
        pred_strs = []
        for tok_str, prob, tok_id in preds:
            tok_repr = repr(tok_str)[:8]
            pred_strs.append(f"{tok_repr}({prob:.2f})")
        preds_str = ", ".join(pred_strs)[:34]

        print(f"{pos:>3} | {input_tok:<16} | {target_tok:<16} | "
              f"{preds_str:<36} | {loss:>5.2f}{correct_marker}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Positions analyzed: {analysis['num_positions']}")
    print(f"Average loss: {analysis['avg_loss']:.4f}")
    print(f"Perplexity: {analysis['perplexity']:.2f}")
    print(f"Top-1 accuracy: {analysis['correct_count']}/{analysis['num_positions']} "
          f"({analysis['top1_accuracy']*100:.1f}%)")
    print(f"Top-5 accuracy: {analysis['top5_correct_count']}/{analysis['num_positions']} "
          f"({analysis['top5_accuracy']*100:.1f}%)")
    print()


def print_batch_summary(batch_summary: dict):
    """Print batch summary statistics."""
    print()
    print("=" * 80)
    print("BATCH SUMMARY")
    print("=" * 80)
    print(f"Batch size: {batch_summary['batch_size']}")
    print(f"Sequence length: {batch_summary['seq_len']}")
    print(f"Total tokens: {batch_summary['total_tokens']:,}")
    print(f"Batch average loss: {batch_summary['batch_loss']:.4f}")
    print(f"Batch perplexity: {batch_summary['batch_perplexity']:.2f}")
    print()


def interactive_prompt() -> str:
    """Wait for user input: n=next, b=batch, q=quit."""
    try:
        cmd = input("[n]ext example | [b]atch summary | [q]uit: ").strip().lower()
        if cmd in ('', 'n', 'next'):
            return 'n'
        elif cmd in ('b', 'batch'):
            return 'b'
        elif cmd in ('q', 'quit', 'exit'):
            return 'q'
        else:
            return 'n'  # Default to next
    except (EOFError, KeyboardInterrupt):
        return 'q'


def main():
    parser = argparse.ArgumentParser(
        description='Validate CustomTransformer with token-by-token analysis'
    )
    parser.add_argument('--config', type=str, required=True, help='Path to config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (optional, uses random init if not provided)')
    parser.add_argument('--find-prompt', type=str, default="Once upon a time",
                        help='Prefix to search for in training examples')
    parser.add_argument('--max-positions', type=int, default=50,
                        help='Maximum positions to analyze')
    parser.add_argument('--max-examples', type=int, default=None,
                        help='Maximum examples to find (None = unlimited)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(config)

    # Load training data (same as train.py)
    print(f"\nLoading training dataset: {config['data']['dataset']}")
    dataset_config = get_dataset_config(config['data']['dataset'])
    print(f"  Description: {dataset_config['description']}")

    train_loader, _ = load_training_data(
        config['data']['dataset'],
        tokenizer,
        max_length=config['data']['max_length'],
        batch_size=config['training']['batch_size'],
        subset_size=config['data'].get('subset_size'),
    )
    print(f"  Train batches: {len(train_loader)}")

    # Parse dtype
    dtype_str = config['model'].get('dtype', 'bfloat16')
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    model_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Initialize model
    print("\nInitializing CustomTransformer...")
    model = CustomTransformerWrapper(
        vocab_size=len(tokenizer),
        max_seq_len=config['model']['max_seq_len'],
        n_blocks=config['model']['n_blocks'],
        n_heads=config['model']['n_heads'],
        d_model=config['model']['d_model'],
        d_ffn=config['model']['d_ffn'],
        dtype=model_dtype,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_checkpoint(args.checkpoint)
    else:
        print("  No checkpoint provided - using random initialization (baseline)")

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")

    # Search for examples
    print(f"\nSearching for examples starting with: \"{args.find_prompt}\"")
    print()

    example_count = 0
    for example in find_training_examples(train_loader, tokenizer, args.find_prompt):
        example_count += 1

        if args.max_examples and example_count > args.max_examples:
            print(f"Reached max examples ({args.max_examples})")
            break

        # Analyze this example
        analysis = analyze_example(
            model,
            example['input_ids'],
            example['labels'],
            tokenizer,
            max_positions=args.max_positions,
        )

        # Print report
        print_analysis_report(analysis, example, tokenizer)

        # Interactive prompt
        while True:
            cmd = interactive_prompt()
            if cmd == 'n':
                break  # Go to next example
            elif cmd == 'b':
                batch_summary = compute_batch_summary(model, example['batch'], model.vocab_size)
                print_batch_summary(batch_summary)
                # After showing batch, ask again
            elif cmd == 'q':
                print("Exiting.")
                return

    if example_count == 0:
        print(f"No examples found starting with \"{args.find_prompt}\"")
        print("Try a different --find-prompt or check your dataset.")


if __name__ == '__main__':
    main()
