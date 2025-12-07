"""Structured training logger compatible with experiment storage."""

import pandas as pd
from datetime import datetime
from typing import Optional, Dict, Any, List
import uuid
from pathlib import Path
import math

from .experiment_storage import save_experiment


# Canonical schema for training runs (for documentation)
TRAINING_LOG_SCHEMA = {
    # Identifiers
    'run_id': 'str - Unique run identifier (UUID)',
    'experiment_name': 'str - Human-readable experiment name',
    'timestamp': 'str - ISO format timestamp',

    # Progress
    'epoch': 'int - Current epoch',
    'step': 'int - Step within epoch (-1 for epoch summary)',
    'global_step': 'int - Total steps across all epochs',

    # Metrics
    'train_loss': 'float - Training loss (avg over log interval)',
    'train_perplexity': 'float - exp(train_loss)',
    'val_loss': 'float - Validation loss (null during training)',
    'val_perplexity': 'float - Validation perplexity',

    # Training state
    'learning_rate': 'float - Current learning rate',

    # Compute tracking
    'approximate_tflops': 'float - Estimated TFLOPs per step',
    'tokens_per_second': 'float - Throughput metric',
    'batch_time_ms': 'float - Time per batch in milliseconds',
}


class TrainingLogger:
    """
    Structured training logger that accumulates metrics and saves to Parquet.

    Logs metrics at configurable intervals and saves to the experiment storage
    system for querying with DuckDB.

    Example:
        logger = TrainingLogger(
            experiment_name='custom_transformer_tinystories',
            model_config={'d_model': 512, 'n_blocks': 6},
            train_config={'batch_size': 16, 'learning_rate': 0.001},
            log_every_n_steps=100,
        )

        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                # ... training step ...
                logger.log_step(
                    epoch=epoch,
                    step=step,
                    train_loss=loss,
                    learning_rate=lr,
                    approximate_tflops=tflops,
                    tokens_per_second=tok_per_sec,
                    batch_time_ms=batch_time,
                )

            # Log epoch-level validation metrics
            logger.log_epoch(
                epoch=epoch,
                val_loss=val_metrics['loss'],
                val_perplexity=val_metrics['perplexity'],
            )

        # Save to Parquet
        logger.save()
    """

    def __init__(
        self,
        experiment_name: str,
        model_config: dict,
        train_config: dict,
        log_every_n_steps: int = 100,
    ):
        """
        Initialize TrainingLogger.

        Args:
            experiment_name: Name for this experiment
            model_config: Model configuration (saved as metadata)
            train_config: Training configuration (saved as metadata)
            log_every_n_steps: Log metrics every N steps (0 = log every step)
        """
        self.run_id = str(uuid.uuid4())[:8]
        self.experiment_name = experiment_name
        self.model_config = model_config
        self.train_config = train_config
        self.log_every_n_steps = log_every_n_steps

        self.logs: List[Dict[str, Any]] = []
        self.global_step = 0
        self.epoch_losses: List[float] = []
        self._last_logged_step = -1

    def log_step(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        learning_rate: float,
        approximate_tflops: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
        batch_time_ms: Optional[float] = None,
        grad_norm: Optional[float] = None,
    ) -> bool:
        """
        Log a training step (only actually logs every N steps).

        Args:
            epoch: Current epoch
            step: Step within epoch
            train_loss: Loss for this step
            learning_rate: Current learning rate
            approximate_tflops: Estimated TFLOPs for this step
            tokens_per_second: Throughput (tokens/second)
            batch_time_ms: Time for this batch in milliseconds
            grad_norm: Gradient norm (optional, for debugging)

        Returns:
            True if this step was logged, False if skipped
        """
        self.global_step += 1
        self.epoch_losses.append(train_loss)

        # Skip if not at log interval
        if self.log_every_n_steps > 0 and self.global_step % self.log_every_n_steps != 0:
            return False

        # Compute average loss over recent steps
        window_size = self.log_every_n_steps if self.log_every_n_steps > 0 else 1
        recent_losses = self.epoch_losses[-window_size:]
        avg_loss = sum(recent_losses) / len(recent_losses)

        log_entry = {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            'train_loss': avg_loss,
            'train_perplexity': math.exp(min(avg_loss, 20)),  # Cap to avoid overflow
            'val_loss': None,
            'val_perplexity': None,
            'learning_rate': learning_rate,
            'approximate_tflops': approximate_tflops,
            'tokens_per_second': tokens_per_second,
            'batch_time_ms': batch_time_ms,
            'grad_norm': grad_norm,
        }

        self.logs.append(log_entry)
        self._last_logged_step = self.global_step
        return True

    def log_epoch(
        self,
        epoch: int,
        val_loss: Optional[float] = None,
        val_perplexity: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """
        Log end-of-epoch validation metrics.

        If the last log entry is from this epoch, updates it with validation
        metrics. Otherwise, creates a new epoch summary entry.

        Args:
            epoch: Completed epoch number
            val_loss: Validation loss
            val_perplexity: Validation perplexity
            learning_rate: Current learning rate (optional)
        """
        # Compute epoch average train loss
        epoch_avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else None

        # Check if we can update the last entry
        if self.logs and self.logs[-1]['epoch'] == epoch and self.logs[-1]['step'] != -1:
            # Update last entry with validation metrics
            self.logs[-1]['val_loss'] = val_loss
            self.logs[-1]['val_perplexity'] = val_perplexity
        else:
            # Create new epoch summary entry
            log_entry = {
                'run_id': self.run_id,
                'experiment_name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch,
                'step': -1,  # Indicates epoch-level entry
                'global_step': self.global_step,
                'train_loss': epoch_avg_loss,
                'train_perplexity': math.exp(min(epoch_avg_loss, 20)) if epoch_avg_loss else None,
                'val_loss': val_loss,
                'val_perplexity': val_perplexity,
                'learning_rate': learning_rate,
                'approximate_tflops': None,
                'tokens_per_second': None,
                'batch_time_ms': None,
                'grad_norm': None,
            }
            self.logs.append(log_entry)

        # Reset epoch losses for next epoch
        self.epoch_losses = []

    def save(self, output_dir: Optional[str] = None) -> Path:
        """
        Save logs to Parquet using experiment storage.

        Args:
            output_dir: Override output directory

        Returns:
            Path to saved Parquet file
        """
        if not self.logs:
            print("Warning: No logs to save")
            return None

        results_df = pd.DataFrame(self.logs)

        # Combine metadata
        metadata = {
            'run_id': self.run_id,
            'model_config': self.model_config,
            'train_config': self.train_config,
        }

        # Use run_id in filename to allow multiple runs
        experiment_full_name = f"{self.experiment_name}_{self.run_id}"

        filepath = save_experiment(
            experiment_full_name,
            results_df,
            metadata=metadata,
            output_dir=output_dir,
        )

        print(f"Training logs saved: {filepath}")
        return filepath

    def get_dataframe(self) -> pd.DataFrame:
        """Get current logs as DataFrame."""
        return pd.DataFrame(self.logs)

    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the most recent logged metrics."""
        if not self.logs:
            return {}
        return self.logs[-1].copy()

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the run."""
        if not self.logs:
            return {}

        df = pd.DataFrame(self.logs)
        return {
            'run_id': self.run_id,
            'experiment_name': self.experiment_name,
            'total_steps': self.global_step,
            'num_entries': len(self.logs),
            'final_train_loss': df['train_loss'].iloc[-1],
            'final_val_loss': df['val_loss'].dropna().iloc[-1] if df['val_loss'].notna().any() else None,
            'min_train_loss': df['train_loss'].min(),
            'min_val_loss': df['val_loss'].min() if df['val_loss'].notna().any() else None,
        }


def estimate_flops_per_step(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    d_model: int,
    d_ffn: int,
    n_blocks: int,
    n_heads: int,
) -> float:
    """
    Estimate FLOPs for one forward + backward pass of a transformer.

    This is an approximation based on standard transformer architecture.
    Forward pass FLOPs roughly equals backward pass FLOPs.

    Based on "Scaling Laws for Neural Language Models" and
    "Training Compute-Optimal Large Language Models" (Chinchilla).

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        vocab_size: Vocabulary size
        d_model: Model dimension
        d_ffn: Feed-forward network dimension
        n_blocks: Number of transformer blocks
        n_heads: Number of attention heads

    Returns:
        Approximate TFLOPs per step
    """
    # Tokens processed per step
    tokens = batch_size * seq_len

    # ===== Per-token FLOPs per block =====

    # Attention QKV projections: 3 * d_model * d_model = 3 * d_model^2
    qkv_flops = 3 * d_model * d_model

    # Attention scores: seq_len * d_model (per token, matmul with all keys)
    attn_scores_flops = seq_len * d_model

    # Attention output: seq_len * d_model (weighted sum of values)
    attn_output_flops = seq_len * d_model

    # Attention output projection: d_model * d_model
    attn_proj_flops = d_model * d_model

    # Total attention per token
    attn_flops_per_token = qkv_flops + attn_scores_flops + attn_output_flops + attn_proj_flops

    # FFN: 2 * d_model * d_ffn (up projection + down projection)
    ffn_flops_per_token = 2 * d_model * d_ffn

    # Total per block per token
    block_flops_per_token = attn_flops_per_token + ffn_flops_per_token

    # ===== Embedding and output =====

    # Embedding lookup is negligible (just indexing)
    # Output projection: d_model * vocab_size
    output_flops_per_token = d_model * vocab_size

    # ===== Total forward pass =====
    forward_flops_per_token = n_blocks * block_flops_per_token + output_flops_per_token
    forward_flops = forward_flops_per_token * tokens

    # Backward pass is roughly 2x forward (gradient computation)
    total_flops = forward_flops * 3  # 1x forward + 2x backward

    # Return as TFLOPs
    return total_flops / 1e12


def format_flops(tflops: float) -> str:
    """Format TFLOPs for display."""
    if tflops >= 1.0:
        return f"{tflops:.2f} TFLOPs"
    elif tflops >= 0.001:
        return f"{tflops * 1000:.2f} GFLOPs"
    else:
        return f"{tflops * 1e6:.2f} MFLOPs"
