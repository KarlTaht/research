"""Weights & Biases logging utilities for experiment tracking."""

from typing import Optional, Dict, Any


class WandbLogger:
    """
    W&B logger that mirrors TrainingLogger's interface.

    Logs scalars, hyperparameters, and metrics to Weights & Biases.
    Supports run resumption for crash recovery.
    """

    def __init__(
        self,
        experiment_name: str,
        run_id: str,
        project: str = "ml-research",
        config: Optional[Dict[str, Any]] = None,
        resume: str = "allow",
    ):
        """
        Initialize W&B logger.

        Args:
            experiment_name: Name for this experiment
            run_id: Unique run identifier (used for resumption)
            project: W&B project name
            config: Model and training configuration to log
            resume: Resume mode - "allow", "must", "never", or "auto"
        """
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=project,
            name=f"{experiment_name}_{run_id}",
            id=run_id,
            config=config,
            resume=resume,
        )

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
        metrics = {
            "train/loss": train_loss,
            "train/learning_rate": learning_rate,
        }

        if train_perplexity is not None:
            metrics["train/perplexity"] = train_perplexity
        if tokens_per_second is not None:
            metrics["perf/tokens_per_second"] = tokens_per_second
        if batch_time_ms is not None:
            metrics["perf/batch_time_ms"] = batch_time_ms
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
        if approximate_tflops is not None:
            metrics["perf/tflops"] = approximate_tflops

        self.wandb.log(metrics, step=global_step)

    def log_epoch(
        self,
        epoch: int,
        global_step: int,
        val_loss: Optional[float] = None,
        val_perplexity: Optional[float] = None,
        train_loss: Optional[float] = None,
    ):
        """Log epoch-level validation metrics."""
        metrics = {"epoch": epoch}

        if val_loss is not None:
            metrics["val/loss"] = val_loss
        if val_perplexity is not None:
            metrics["val/perplexity"] = val_perplexity
        if train_loss is not None:
            metrics["train/epoch_loss"] = train_loss

        self.wandb.log(metrics, step=global_step)

    def log_text(self, key: str, text: str):
        """Log text (e.g., generated samples)."""
        self.wandb.log({key: self.wandb.Html(f"<pre>{text}</pre>")})

    def finish(self):
        """Finish the W&B run."""
        self.wandb.finish()
