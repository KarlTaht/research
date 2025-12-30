"""Training loop for least action learning experiments."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from pathlib import Path
import json

from .data import ModularArithmeticDataset, SequenceArithmeticDataset
from .models import BaselineMLP, RoutedNetwork, GrokTransformer, create_model
from .losses import (
    LeastActionLoss,
    spectral_smoothness,
    compute_jacobian_norm,
    compute_hessian_trace,
)
from .metrics import (
    TrainingMetrics,
    MetricsHistory,
    compute_layer_weight_norms,
    compute_total_weight_norm,
    compute_representation_norm,
)


@dataclass
class TrainerConfig:
    """Configuration for training."""

    # Data
    p: int = 113
    operation: str = "add"
    train_frac: float = 0.3
    data_seed: int = 42

    # Model
    model_type: str = "routed"  # "baseline", "routed", "single_head"
    hidden_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4

    # Training
    epochs: int = 50000
    lr: float = 1e-3
    weight_decay: float = 0.1
    optimizer: str = "adamw"
    beta1: float = 0.9  # AdamW first moment decay
    beta2: float = 0.98  # AdamW second moment decay (0.98 more stable than 0.999)
    eps: float = 1e-8  # AdamW epsilon
    grad_clip: Optional[float] = None  # Max gradient norm (None = no clipping)
    warmup_epochs: int = 0  # Linear warmup from 0 to lr over this many epochs

    # Loss / regularization
    routing_regularizer: Optional[str] = "entropy"
    lambda_routing: float = 0.01
    lambda_spectral: float = 0.0
    spectral_k: Optional[int] = None
    spectral_interval: int = 100

    # Logging
    log_every: int = 100
    save_routing_every: int = 1000
    checkpoint_every: int = 5000

    # Experiment
    name: str = ""
    seed: int = 42

    def __post_init__(self):
        """Convert string values to proper types (handles YAML quirks)."""
        # Float fields that YAML might parse as strings
        float_fields = ['lr', 'weight_decay', 'beta1', 'beta2', 'eps',
                        'train_frac', 'lambda_routing', 'lambda_spectral']
        for field in float_fields:
            val = getattr(self, field)
            if isinstance(val, str):
                setattr(self, field, float(val))

    def to_dict(self) -> dict:
        return asdict(self)


class Trainer:
    """
    Training loop for least action learning experiments.

    Handles training, evaluation, and metric logging for both
    baseline MLPs and routed networks.
    """

    def __init__(
        self,
        config: TrainerConfig,
        output_dir: Optional[Path] = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set seeds
        torch.manual_seed(config.seed)

        # Setup device (CUDA > MPS > CPU)
        self.device = self._get_device()

        # Track if using transformer (sequence-based input)
        self.is_transformer = config.model_type == "transformer"

        # Create dataset (sequence-based for transformer, one-hot for others)
        if self.is_transformer:
            self.dataset = SequenceArithmeticDataset(
                p=config.p,
                operation=config.operation,
                train_frac=config.train_frac,
                seed=config.data_seed,
            )
        else:
            self.dataset = ModularArithmeticDataset(
                p=config.p,
                operation=config.operation,
                train_frac=config.train_frac,
                seed=config.data_seed,
            )

        # Create model
        self.model = create_model(
            model_type=config.model_type,
            input_dim=self.dataset.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=self.dataset.output_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
        ).to(self.device)

        # Create optimizer
        self.optimizer = self._create_optimizer()

        # Create loss function
        self.loss_fn = LeastActionLoss(
            routing_regularizer=config.routing_regularizer,
            lambda_routing=config.lambda_routing,
            lambda_spectral=config.lambda_spectral,
            spectral_k=config.spectral_k,
            spectral_interval=config.spectral_interval,
        )

        # Metrics history
        self.history = MetricsHistory()

        # Move data to device
        train_split = self.dataset.get_train()
        test_split = self.dataset.get_test()
        if self.is_transformer:
            # Sequence-based inputs (token IDs)
            self.train_inputs = train_split.input_ids.to(self.device)
            self.test_inputs = test_split.input_ids.to(self.device)
        else:
            # One-hot encoded inputs
            self.train_inputs = train_split.inputs.to(self.device)
            self.test_inputs = test_split.inputs.to(self.device)
        self.train_targets = train_split.targets.to(self.device)
        self.test_targets = test_split.targets.to(self.device)

        # Best model tracking
        self.best_test_acc = 0.0
        self.best_epoch = 0

        # Track steps with routing snapshots (for visualizer)
        self.routing_snapshot_steps: list[int] = []

    @staticmethod
    def _get_device() -> torch.device:
        """Get best available device: CUDA > MPS > CPU."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config.

        For AdamW, uses separate parameter groups to exclude embeddings,
        biases, and LayerNorm from weight decay (standard practice).
        """
        if self.config.optimizer == "adamw":
            # Separate parameters into decay and no-decay groups
            decay_params = []
            no_decay_params = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                # No weight decay for: biases, LayerNorm, embeddings
                if (
                    'bias' in name
                    or 'ln' in name.lower()
                    or 'layernorm' in name.lower()
                    or 'embedding' in name
                ):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

            param_groups = [
                {'params': decay_params, 'weight_decay': self.config.weight_decay},
                {'params': no_decay_params, 'weight_decay': 0.0},
            ]

            return optim.AdamW(
                param_groups,
                lr=self.config.lr,
                betas=(self.config.beta1, self.config.beta2),
                eps=self.config.eps,
            )
        elif self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def train_step(self, epoch: int) -> tuple[float, float, Optional[list[Tensor]]]:
        """
        Single training step (full batch).

        Returns:
            train_loss: Training loss
            train_acc: Training accuracy
            routing_weights: Routing weights if applicable
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(self.train_inputs)

        # Handle routed vs baseline models
        if isinstance(outputs, tuple):
            logits, metrics = outputs
            routing_weights = metrics.layer_weights
        else:
            logits = outputs
            routing_weights = None

        # Compute loss
        loss, loss_dict = self.loss_fn(
            logits=logits,
            targets=self.train_targets,
            routing_weights=routing_weights,
            inputs=self.train_inputs,
            model=self.model,
            p=self.config.p,
            step=epoch,
        )

        # Backward pass
        loss.backward()
        if self.config.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            acc = (preds == self.train_targets).float().mean().item()

        return loss_dict["task_loss"], acc, routing_weights

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """
        Evaluate on test set.

        Returns:
            test_loss: Test loss
            test_acc: Test accuracy
        """
        self.model.eval()

        outputs = self.model(self.test_inputs)
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs

        loss = nn.functional.cross_entropy(logits, self.test_targets)
        preds = logits.argmax(dim=-1)
        acc = (preds == self.test_targets).float().mean().item()

        return loss.item(), acc

    @torch.no_grad()
    def compute_spectral_smoothness(self) -> float:
        """Compute spectral smoothness of current model."""
        k = self.config.spectral_k or (self.config.p // 4)
        return spectral_smoothness(
            self.model, self.config.p, k, self.device,
            is_transformer=self.is_transformer
        )

    def train(
        self,
        callback: Optional[Callable[[int, TrainingMetrics], None]] = None,
        start_epoch: int = 0,
    ) -> MetricsHistory:
        """
        Run full training loop.

        Args:
            callback: Optional function called after each logged step
            start_epoch: Epoch to start from (for resuming)

        Returns:
            MetricsHistory with all training metrics
        """
        print(f"Training {self.config.model_type} model on mod-{self.config.p} {self.config.operation}")
        print(f"  Layers: {self.config.n_layers}, Hidden: {self.config.hidden_dim}")
        if self.config.model_type == "routed":
            print(f"  Heads: {self.config.n_heads}, Routing reg: {self.config.routing_regularizer}")
        elif self.config.model_type == "transformer":
            print(f"  Attention heads: {self.config.n_heads}, Vocab: {self.dataset.vocab_size}")
        print(f"  Device: {self.device}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        if start_epoch > 0:
            print(f"  Resuming from epoch {start_epoch}")
        print()

        for epoch in range(start_epoch, self.config.epochs):
            # Learning rate warmup
            if self.config.warmup_epochs > 0 and epoch < self.config.warmup_epochs:
                warmup_factor = (epoch + 1) / self.config.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.lr * warmup_factor
            elif epoch == self.config.warmup_epochs and self.config.warmup_epochs > 0:
                # Restore full learning rate after warmup
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.lr

            # Training step
            train_loss, train_acc, routing_weights = self.train_step(epoch)

            # Logging
            if epoch % self.config.log_every == 0:
                test_loss, test_acc = self.evaluate()

                # Compute routing metrics
                if routing_weights is not None:
                    routing_entropy = sum(
                        -(w * w.log().clamp(min=-100)).sum(dim=-1).mean().item()
                        for w in routing_weights
                    ) / len(routing_weights)
                    head_util = torch.stack(routing_weights).mean(dim=(0, 1)).tolist()
                else:
                    routing_entropy = 0.0
                    head_util = []

                # Spectral smoothness (every log_every epochs)
                smoothness = self.compute_spectral_smoothness()

                # Curvature metrics (every log_every epochs)
                # Sample subset for efficiency
                sample_size = min(256, len(self.test_inputs))
                sample_indices = torch.randperm(len(self.test_inputs))[:sample_size]
                sample_inputs = self.test_inputs[sample_indices]

                # Compute Jacobian/Hessian w.r.t. embeddings for transformers
                jac_norm = compute_jacobian_norm(
                    self.model, sample_inputs,
                    num_samples=10,
                    is_transformer=self.is_transformer,
                )
                hess_trace = compute_hessian_trace(
                    self.model, sample_inputs,
                    num_hutchinson_samples=5,
                    is_transformer=self.is_transformer,
                )

                # Weight norms
                layer_norms = compute_layer_weight_norms(self.model)
                total_norm = compute_total_weight_norm(self.model)

                # Representation norm (hidden state before unembedding)
                repr_norm = compute_representation_norm(self.model, self.test_inputs)

                # Create metrics object
                metrics = TrainingMetrics(
                    step=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    test_loss=test_loss,
                    test_acc=test_acc,
                    routing_entropy=routing_entropy,
                    head_utilization=head_util,
                    spectral_smoothness=smoothness,
                    layer_weight_norms=layer_norms,
                    total_weight_norm=total_norm,
                    representation_norm=repr_norm,
                    jacobian_norm=jac_norm,
                    hessian_trace=hess_trace,
                )

                # Log to history
                save_routing = (
                    routing_weights is not None
                    and epoch % self.config.save_routing_every == 0
                )
                self.history.log(
                    metrics,
                    routing_weights if save_routing else None,
                )
                if save_routing:
                    self.routing_snapshot_steps.append(epoch)

                # Track best model
                if test_acc > self.best_test_acc:
                    self.best_test_acc = test_acc
                    self.best_epoch = epoch
                    self.save_checkpoint("best.pt", epoch=epoch)

                # Print progress
                routing_str = f", entropy={routing_entropy:.3f}" if routing_weights else ""
                print(
                    f"Epoch {epoch:7d} | "
                    f"Train: {train_acc*100:5.1f}% | "
                    f"Test: {test_acc*100:5.1f}% | "
                    f"Loss: {train_loss:.2e}, "
                    f"rnorm={repr_norm:.1f}, smooth={smoothness:.3f}, "
                    f"jac={jac_norm:.2e}, |hess|={abs(hess_trace):.2e}"
                    f"{routing_str}"
                )

                # Callback
                if callback is not None:
                    callback(epoch, metrics)

            # Checkpointing
            if epoch % self.config.checkpoint_every == 0 and epoch > 0:
                self.save_checkpoint(f"checkpoint_{epoch}.pt", epoch=epoch)

        # Final save (epoch is config.epochs - 1 since range is exclusive)
        self.save_checkpoint("final.pt", epoch=self.config.epochs - 1)
        self.save_history()

        print()
        print(f"Training complete. Best test acc: {self.best_test_acc:.4f} at epoch {self.best_epoch}")

        return self.history

    def save_checkpoint(self, filename: str, epoch: Optional[int] = None):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "best_test_acc": self.best_test_acc,
            "best_epoch": self.best_epoch,
            "epoch": epoch,  # Actual last trained epoch
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.output_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.best_test_acc = checkpoint.get("best_test_acc", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)

    def save_history(self):
        """Save training history and routing snapshots."""
        df = self.history.get_dataframe()
        df.to_parquet(self.output_dir / "history.parquet")

        # Also save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Save routing snapshots for visualizer
        if self.history.routing_weights_history:
            snapshots = []
            for step, weights in zip(
                self.routing_snapshot_steps,
                self.history.routing_weights_history,
            ):
                snapshots.append({
                    "step": step,
                    "weights": [w.cpu() for w in weights],
                })

            torch.save(
                {"snapshots": snapshots},
                self.output_dir / "routing_snapshots.pt",
            )
            print(f"Saved {len(snapshots)} routing snapshots")


def train_experiment(
    config: TrainerConfig,
    output_dir: Optional[Path] = None,
) -> MetricsHistory:
    """
    Convenience function to train an experiment.

    Args:
        config: Training configuration
        output_dir: Output directory (defaults to outputs/{config.name})

    Returns:
        Training history
    """
    if output_dir is None:
        name = config.name or f"{config.model_type}_{config.p}_{config.operation}"
        output_dir = Path("outputs") / name

    trainer = Trainer(config, output_dir)
    return trainer.train()
