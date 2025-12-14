#!/usr/bin/env python3
"""Training script for continual learning experiments.

Features:
- Supports pre-tokenized JSONL corpora (automotive, food)
- Uses PyTorch-native TorchTransformer
- bfloat16 training with mixed precision
- Checkpoint saving with training resumption
- DuckDB-compatible experiment tracking

Usage:
    # Train on automotive corpus
    python train.py --config configs/automotive.yaml

    # Train on food corpus
    python train.py --config configs/food.yaml

    # Resume from checkpoint
    python train.py --config configs/automotive.yaml --resume checkpoints/latest.pt
"""

import sys
from pathlib import Path
import math
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from datasets import load_from_disk
from tqdm import tqdm
import argparse
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import TorchTransformer, create_model
from common.data import get_models_dir
from common.training import CheckpointManager
from common.utils import TrainingLogger, estimate_flops_per_step, format_flops


class TeeLogger:
    """Write to both stdout and a log file."""

    def __init__(self, log_file: Path):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'w', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


def load_config(config_path: str = None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "automotive.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tokenizer(tokenizer_name: str):
    """Load tokenizer from assets/models/tokenizers/."""
    tokenizer_path = get_models_dir() / 'tokenizers' / tokenizer_name
    if not tokenizer_path.exists():
        tokenizer_path = get_models_dir() / tokenizer_name
    if not tokenizer_path.exists():
        tokenizer_path = Path(tokenizer_name)

    print(f"  Loading tokenizer from: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

    print(f"  Vocab size: {len(tokenizer)}")
    return tokenizer


def load_pretokenized_data(
    data_dir: Path,
    tokenizer_name: str,
    max_length: int,
    batch_size: int,
    subset_size: int = None,
):
    """Load pre-tokenized data from cache.

    Expects cache at: {data_dir}/train_tokenized/{tokenizer_name}_v{vocab}_len{max_length}/

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Get tokenizer to determine vocab size for cache key
    tokenizer = load_tokenizer(tokenizer_name)
    vocab_size = len(tokenizer)

    cache_key = f"{tokenizer_name}_v{vocab_size}_len{max_length}"
    train_cache = data_dir / "train_tokenized" / cache_key / "train"
    val_cache = data_dir / "train_tokenized" / cache_key / "validation"

    print(f"  Loading pre-tokenized train from: {train_cache}")
    if not train_cache.exists():
        raise FileNotFoundError(
            f"Pre-tokenized cache not found: {train_cache}\n"
            f"Run: python tools/pretokenize_dataset.py --jsonl {data_dir}/train.jsonl "
            f"--val-jsonl {data_dir}/val.jsonl --tokenizer {tokenizer_name} --max-length {max_length}"
        )

    train_dataset = load_from_disk(str(train_cache))
    print(f"  Loaded {len(train_dataset):,} train examples")

    # Apply subset if specified
    if subset_size and subset_size < len(train_dataset):
        train_dataset = train_dataset.select(range(subset_size))
        print(f"  Using subset: {len(train_dataset):,} examples")

    train_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    # Load validation
    val_loader = None
    if val_cache.exists():
        print(f"  Loading pre-tokenized val from: {val_cache}")
        val_dataset = load_from_disk(str(val_cache))
        print(f"  Loaded {len(val_dataset):,} val examples")
        val_dataset.set_format(type='torch', columns=['input_ids', 'labels'])
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    return train_loader, val_loader, tokenizer


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=100):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']

        # Count non-padding tokens
        num_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(min(avg_loss, 100))  # Cap to prevent overflow

    model.train()
    return {'loss': avg_loss, 'perplexity': perplexity, 'num_tokens': total_tokens}


def main():
    parser = argparse.ArgumentParser(description='Train TorchTransformer')
    parser.add_argument('--config', type=str, default=None, help='Path to config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--log-file', type=str, default=None, help='Log output to file')
    args = parser.parse_args()

    config = load_config(args.config)
    experiment_name = config.get('experiment_name', 'continual_learning')

    # Setup logging
    tee_logger = None
    log_path = Path(args.log_file) if args.log_file else Path(__file__).parent / 'logs' / f'{experiment_name}.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee_logger = TeeLogger(log_path)
    sys.stdout = tee_logger
    sys.stderr = tee_logger
    print(f"Logging to: {log_path}")

    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print(f"\nLoading data from: {config['data']['corpus_dir']}")
    corpus_dir = Path(__file__).parent / config['data']['corpus_dir']
    train_loader, val_loader, tokenizer = load_pretokenized_data(
        data_dir=corpus_dir,
        tokenizer_name=config['data']['tokenizer'],
        max_length=config['data']['max_length'],
        batch_size=config['training']['batch_size'],
        subset_size=config['data'].get('subset_size'),
    )

    print(f"  Train batches: {len(train_loader)}")
    if val_loader:
        print(f"  Val batches: {len(val_loader)}")

    # Parse dtype
    dtype_str = config['model'].get('dtype', 'bfloat16')
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    model_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    # Initialize model
    print("\nInitializing TorchTransformer...")
    model = TorchTransformer(
        vocab_size=len(tokenizer),
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_seq_len=config['model']['max_seq_len'],
        dropout=config['model'].get('dropout', 0.1),
    )

    # Move to device and dtype
    model = model.to(device=device, dtype=model_dtype)

    info = model.get_model_info()
    print(f"  Parameters: {info['parameters']:,} ({info['parameters_millions']:.1f}M)")
    print(f"  Dtype: {model_dtype}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01),
        betas=(0.9, 0.95),
    )

    # Setup checkpoint manager
    checkpoint_dir = Path(__file__).parent / 'checkpoints' / experiment_name
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        model=model,
        optimizer=optimizer,
        experiment_name=experiment_name,
        max_checkpoints=5,
    )

    # Setup training logger
    logger = TrainingLogger(
        experiment_name=experiment_name,
        model_config=config['model'],
        train_config=config['training'],
        log_every_n_steps=config['training'].get('log_every', 100),
    )

    # Estimate FLOPs
    tflops_per_step = estimate_flops_per_step(
        batch_size=config['training']['batch_size'],
        seq_len=config['data']['max_length'],
        vocab_size=len(tokenizer),
        d_model=config['model']['d_model'],
        d_ffn=config['model']['d_ff'],
        n_blocks=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
    )
    print(f"  Estimated TFLOPs/step: {format_flops(tflops_per_step)}")

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    if args.resume:
        resume_path = args.resume if args.resume != 'latest' else None
        resume_state = checkpoint_manager.load_checkpoint(resume_path)
        if resume_state:
            start_epoch = resume_state['epoch']
            global_step = resume_state['global_step']
            best_val_loss = resume_state['metrics'].get('val_loss', float('inf'))
            print(f"\nResumed from epoch {start_epoch}, step {global_step}")

    # Training configuration
    num_epochs = config['training']['num_epochs']
    log_every = config['training'].get('log_every', 100)
    eval_every = config['training'].get('eval_every', 500)
    max_grad_norm = config['training'].get('max_grad_norm', 1.0)

    # Learning rate schedule
    base_lr = config['training']['learning_rate']
    min_lr = config['training'].get('min_learning_rate', base_lr * 0.1)
    lr_decay = config['training'].get('lr_decay', None)
    total_steps = num_epochs * len(train_loader)

    warmup_ratio = config['training'].get('warmup_ratio', 0.0)
    warmup_steps = config['training'].get('warmup_steps', int(total_steps * warmup_ratio))
    if warmup_steps > 0:
        print(f"  Warmup: {warmup_steps} steps")

    # Use mixed precision for bfloat16
    use_amp = model_dtype == torch.bfloat16
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # Training loop
    print(f"\nTraining for epochs {start_epoch + 1} to {num_epochs}...")

    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        epoch_losses = []
        epoch_start_time = time.time()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        model.train()
        for step, batch in enumerate(progress_bar):
            step_start_time = time.time()

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Compute learning rate with warmup and optional decay
            if global_step < warmup_steps:
                learning_rate = base_lr * (global_step / warmup_steps) if warmup_steps > 0 else base_lr
            elif lr_decay == 'cosine':
                decay_steps = total_steps - warmup_steps
                decay_progress = (global_step - warmup_steps) / decay_steps if decay_steps > 0 else 1.0
                learning_rate = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * decay_progress))
            elif lr_decay == 'linear':
                decay_steps = total_steps - warmup_steps
                decay_progress = (global_step - warmup_steps) / decay_steps if decay_steps > 0 else 1.0
                learning_rate = base_lr - (base_lr - min_lr) * decay_progress
            else:
                learning_rate = base_lr

            # Update optimizer LR
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # Forward pass with autocast for mixed precision
            optimizer.zero_grad()

            if use_amp and device.type == 'cuda':
                with torch.amp.autocast('cuda', dtype=model_dtype):
                    outputs = model(input_ids, labels=labels)
                    loss = outputs['loss']
            else:
                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            batch_time_ms = (time.time() - step_start_time) * 1000
            tokens_per_second = input_ids.numel() / (batch_time_ms / 1000)

            global_step += 1
            epoch_losses.append(loss.item())

            # Log step
            logger.log_step(
                epoch=epoch,
                step=step,
                train_loss=loss.item(),
                learning_rate=learning_rate,
                approximate_tflops=tflops_per_step,
                tokens_per_second=tokens_per_second,
                batch_time_ms=batch_time_ms,
            )

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{learning_rate:.2e}",
                'tok/s': f"{tokens_per_second:.0f}",
            })

            # Periodic evaluation
            if global_step % eval_every == 0 and val_loader:
                val_metrics = evaluate(model, val_loader, device, max_batches=50)
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
            val_metrics = evaluate(model, val_loader, device, max_batches=None)
            val_loss = val_metrics['loss']
            val_perplexity = val_metrics['perplexity']

            print(f"  Val loss: {val_loss:.4f}")
            print(f"  Val perplexity: {val_perplexity:.2f}")

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
    print(f"  Checkpoints saved to: {checkpoint_dir}")
    print("="*50)

    # Cleanup file logger
    if tee_logger:
        sys.stdout = tee_logger.terminal
        sys.stderr = tee_logger.terminal
        tee_logger.close()


if __name__ == '__main__':
    main()
