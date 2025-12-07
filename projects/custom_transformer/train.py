#!/usr/bin/env python3
"""Enhanced training script for CustomTransformer with multi-dataset support.

Features:
- Multi-dataset support via dataset registry (TinyStories, tiny-textbooks, etc.)
- Custom tokenizer support (e.g., dataset-specific BPE tokenizers)
- bfloat16 with precision mixing for numerical stability
- Checkpoint snapshotting with training resumption
- DuckDB-compatible experiment tracking with TFLOP estimation
- Advanced evaluation with optional Claude Haiku coherence scoring

Usage:
    # Train on TinyStories with GPT-2 tokenizer
    python train.py --config configs/tinystories.yaml

    # Train with custom tokenizer (smaller vocab = faster training)
    python train.py --config configs/tinystories_custom_tokenizer.yaml

    # Resume from checkpoint
    python train.py --config configs/tinystories.yaml --resume checkpoints/latest.pt

    # Skip Claude coherence evaluation
    python train.py --config configs/tinystories.yaml --no-coherence-eval
"""

import sys
from pathlib import Path
import yaml
import torch
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from tqdm import tqdm
import argparse
import time


class TeeLogger:
    """Write to both stdout and a log file."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', buffering=1)  # Line buffered

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models.custom_transfromer.wrapper import CustomTransformerWrapper
from common.data import load_training_data, get_dataset_config, get_models_dir
from common.training import CheckpointManager, AdvancedEvaluator
from common.utils import TrainingLogger, estimate_flops_per_step, format_flops


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


def main():
    parser = argparse.ArgumentParser(description='Train CustomTransformer')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint (path or "latest")')
    parser.add_argument('--no-coherence-eval', action='store_true', help='Skip Claude Haiku coherence evaluation')
    parser.add_argument('--log-file', type=str, default=None, help='Log output to file (can tail -f)')
    args = parser.parse_args()

    # Setup file logging if requested
    tee_logger = None
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tee_logger = TeeLogger(log_path)
        sys.stdout = tee_logger
        sys.stderr = tee_logger
        print(f"Logging to: {log_path}")

    config = load_config(args.config)
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Load tokenizer (supports GPT-2 or custom tokenizers)
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer(config)

    # Load data using dataset registry
    print(f"\nLoading dataset: {config['data']['dataset']}")
    dataset_config = get_dataset_config(config['data']['dataset'])
    print(f"  Description: {dataset_config['description']}")

    train_loader, val_loader = load_training_data(
        config['data']['dataset'],
        tokenizer,
        max_length=config['data']['max_length'],
        batch_size=config['training']['batch_size'],
        subset_size=config['data'].get('subset_size'),
        val_subset_size=config['data'].get('val_subset_size'),
    )

    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    # Create model with bfloat16 (precision mixing handled internally)
    print("\nInitializing CustomTransformer...")
    model = CustomTransformerWrapper(
        vocab_size=len(tokenizer),
        max_seq_len=config['model']['max_seq_len'],
        n_blocks=config['model']['n_blocks'],
        n_heads=config['model']['n_heads'],
        d_model=config['model']['d_model'],
        d_ffn=config['model']['d_ffn'],
        dtype=torch.bfloat16,
    )

    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {model.device}")
    print(f"  Dtype: {model.dtype}")

    # Setup checkpoint manager
    checkpoint_dir = Path(__file__).parent / 'checkpoints'
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=model,
        experiment_name=config.get('experiment_name', 'custom_transformer'),
        max_checkpoints=5,
    )

    # Setup training logger
    logger = TrainingLogger(
        experiment_name=config.get('experiment_name', 'custom_transformer'),
        model_config=config['model'],
        train_config=config['training'],
        log_every_n_steps=config['training'].get('log_every', 100),
    )

    # Setup advanced evaluator
    evaluator = AdvancedEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=model.device,
    )

    # Estimate FLOPs per step
    tflops_per_step = estimate_flops_per_step(
        batch_size=config['training']['batch_size'],
        seq_len=config['data']['max_length'],
        vocab_size=len(tokenizer),
        d_model=config['model']['d_model'],
        d_ffn=config['model']['d_ffn'],
        n_blocks=config['model']['n_blocks'],
        n_heads=config['model']['n_heads'],
    )
    print(f"  Estimated TFLOPs/step: {format_flops(tflops_per_step)}")

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    learning_rate = config['training']['learning_rate']
    best_val_loss = float('inf')

    if args.resume:
        resume_path = args.resume if args.resume != 'latest' else None
        resume_state = checkpoint_manager.load_checkpoint(resume_path)
        if resume_state:
            start_epoch = resume_state['epoch']
            global_step = resume_state['global_step']
            learning_rate = resume_state['learning_rate']
            best_val_loss = resume_state['metrics'].get('val_loss', float('inf'))
            print(f"\nResumed from epoch {start_epoch}, step {global_step}")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Best val loss: {best_val_loss:.4f}")

    # Training configuration
    num_epochs = config['training']['num_epochs']
    log_every = config['training'].get('log_every', 100)
    eval_every = config['training'].get('eval_every', 500)
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)
    max_nan_count = config['training'].get('max_nan_count', 10)

    # Generation prompts for evaluation
    generation_prompts = config.get('evaluation', {}).get(
        'generation_prompts',
        ['Once upon a time', 'The little girl', 'One day, a boy named']
    )

    # Training loop
    print(f"\nTraining for epochs {start_epoch + 1} to {num_epochs}...")
    nan_count = 0

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        epoch_losses = []
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for step, batch in enumerate(progress_bar):
            step_start_time = time.time()

            input_ids = batch['input_ids']
            labels = batch['labels']

            # Training step with stability features
            result = model.train_step(
                input_ids,
                labels,
                learning_rate,
                max_grad_norm=max_grad_norm,
            )

            batch_time_ms = (time.time() - step_start_time) * 1000
            tokens_per_second = input_ids.numel() / (batch_time_ms / 1000)

            # Handle NaN detection
            if result.get('status') in ['nan_detected', 'nan_gradient']:
                nan_count += 1
                print(f"\n  NaN detected at step {global_step}. Count: {nan_count}/{max_nan_count}")

                if nan_count >= max_nan_count:
                    print("  Too many NaNs. Halting training.")
                    logger.save()
                    return

                continue  # Skip this step

            global_step += 1
            epoch_losses.append(result['loss'])

            # Log step
            logged = logger.log_step(
                epoch=epoch,
                step=step,
                train_loss=result['loss'],
                learning_rate=learning_rate,
                approximate_tflops=tflops_per_step,
                tokens_per_second=tokens_per_second,
                batch_time_ms=batch_time_ms,
                grad_norm=result.get('grad_norm'),
            )

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{result['loss']:.4f}",
                'tok/s': f"{tokens_per_second:.0f}",
                'grad': f"{result.get('grad_norm', 0):.2f}",
            })

            # Periodic evaluation
            if global_step % eval_every == 0 and val_loader:
                val_metrics = evaluator.evaluate_perplexity(val_loader, max_batches=50)
                print(
                    f"\n  Step {global_step}: "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_ppl={val_metrics['perplexity']:.2f}"
                )

        # End of epoch
        epoch_time = time.time() - epoch_start_time
        train_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float('inf')

        print(f"\nEpoch {epoch + 1} complete in {epoch_time:.1f}s")
        print(f"  Train loss: {train_loss:.4f}")

        # Full evaluation at end of epoch
        if val_loader:
            print("  Running full evaluation...")
            eval_config = config.get('evaluation', {})
            eval_results = evaluator.full_evaluation(
                val_loader,
                generation_prompts=generation_prompts,
                max_eval_batches=100,
                max_generation_length=eval_config.get('max_generation_length', 100),
                temperature=eval_config.get('temperature', 0.8),
                top_k=eval_config.get('top_k', 50),
                evaluate_coherence=(
                    not args.no_coherence_eval and
                    eval_config.get('evaluate_coherence', True)
                ),
            )

            val_loss = eval_results['loss']
            val_perplexity = eval_results['perplexity']

            print(f"  Val loss: {val_loss:.4f}")
            print(f"  Val perplexity: {val_perplexity:.2f}")
            print(f"  Vocab diversity: {eval_results.get('heuristic_vocab_diversity', 0):.3f}")
            print(f"  Repetition rate: {eval_results.get('heuristic_repetition_rate', 0):.3f}")

            if 'claude_coherence_score_avg' in eval_results and eval_results['claude_coherence_score_avg']:
                print(f"  Claude coherence: {eval_results['claude_coherence_score_avg']:.1f}/5")

            # Print sample generations
            if eval_results.get('generated_samples'):
                print("\n  Sample generations:")
                for i, sample in enumerate(eval_results['generated_samples'][:2], 1):
                    continuation = sample['continuation'][:80]
                    print(f"    [{i}] \"{sample['prompt']}\" -> \"{continuation}...\"")

            # Log epoch metrics
            logger.log_epoch(
                epoch=epoch,
                val_loss=val_loss,
                val_perplexity=val_perplexity,
                learning_rate=learning_rate,
            )

            # Check if best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print(f"  New best val loss!")

            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                epoch=epoch + 1,
                global_step=global_step,
                train_config=config['training'],
                learning_rate=learning_rate,
                metrics={
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_perplexity': val_perplexity,
                },
                is_best=is_best,
            )

    # Save final experiment results
    print("\nSaving experiment logs...")
    logger.save()

    # Print final summary
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"  Total steps: {global_step}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  NaN occurrences: {nan_count}")
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    summary = logger.get_summary()
    print(f"  Run ID: {summary['run_id']}")
    print("="*50)

    # Cleanup file logger
    if tee_logger:
        sys.stdout = tee_logger.terminal
        sys.stderr = tee_logger.terminal
        tee_logger.close()


if __name__ == '__main__':
    main()
