"""TensorBoard logging utilities for experiment tracking."""

from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter
import json


def get_tensorboard_dir() -> Path:
    """Get the default TensorBoard logs directory."""
    from common.data import get_default_assets_dir

    tb_dir = get_default_assets_dir() / "outputs" / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return tb_dir


class TensorBoardLogger:
    """
    TensorBoard logger that mirrors TrainingLogger's interface.

    Logs scalars, hyperparameters, and text to TensorBoard event files.
    """

    def __init__(
        self,
        experiment_name: str,
        run_id: str,
        log_dir: Optional[Path] = None,
        model_config: Optional[Dict[str, Any]] = None,
        train_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize TensorBoard logger.

        Args:
            experiment_name: Name for this experiment
            run_id: Unique run identifier (e.g., from TrainingLogger)
            log_dir: Override log directory (default: assets/outputs/tensorboard/)
            model_config: Model configuration to log as hyperparameters
            train_config: Training configuration to log as hyperparameters
        """
        if log_dir is None:
            log_dir = get_tensorboard_dir()

        # Create subdirectory: tensorboard/{experiment_name}_{run_id}/
        self.log_path = log_dir / f"{experiment_name}_{run_id}"
        self.writer = SummaryWriter(log_dir=str(self.log_path))

        # Log hyperparameters
        if model_config or train_config:
            hparams = {}
            if model_config:
                for k, v in model_config.items():
                    if isinstance(v, (int, float, str, bool)):
                        hparams[f"model/{k}"] = v
            if train_config:
                for k, v in train_config.items():
                    if isinstance(v, (int, float, str, bool)):
                        hparams[f"train/{k}"] = v
            if hparams:
                # Log as text since add_hparams requires metrics
                self.writer.add_text("hyperparameters", json.dumps(hparams, indent=2), 0)

    def log_step(
        self,
        global_step: int,
        train_loss: float,
        learning_rate: float,
        train_perplexity: Optional[float] = None,
        tokens_per_second: Optional[float] = None,
        batch_time_ms: Optional[float] = None,
        grad_norm: Optional[float] = None,
        approximate_tflops: Optional[float] = None,
    ):
        """Log training step metrics."""
        self.writer.add_scalar("train/loss", train_loss, global_step)
        self.writer.add_scalar("train/learning_rate", learning_rate, global_step)

        if train_perplexity is not None:
            self.writer.add_scalar("train/perplexity", train_perplexity, global_step)
        if tokens_per_second is not None:
            self.writer.add_scalar("perf/tokens_per_second", tokens_per_second, global_step)
        if batch_time_ms is not None:
            self.writer.add_scalar("perf/batch_time_ms", batch_time_ms, global_step)
        if grad_norm is not None:
            self.writer.add_scalar("train/grad_norm", grad_norm, global_step)
        if approximate_tflops is not None:
            self.writer.add_scalar("perf/tflops", approximate_tflops, global_step)

    def log_epoch(
        self,
        epoch: int,
        global_step: int,
        val_loss: Optional[float] = None,
        val_perplexity: Optional[float] = None,
        train_loss: Optional[float] = None,
    ):
        """Log epoch-level validation metrics."""
        if val_loss is not None:
            self.writer.add_scalar("val/loss", val_loss, global_step)
        if val_perplexity is not None:
            self.writer.add_scalar("val/perplexity", val_perplexity, global_step)
        if train_loss is not None:
            self.writer.add_scalar("train/epoch_loss", train_loss, global_step)

    def log_text(self, tag: str, text: str, global_step: int):
        """Log text (e.g., generated samples)."""
        self.writer.add_text(tag, text, global_step)

    def close(self):
        """Close the writer and flush all pending events."""
        self.writer.close()
