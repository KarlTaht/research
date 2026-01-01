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
from .losses import LeastActionLoss
from .metrics import (
    TrainingMetrics,
    MetricsHistory,
    compute_layer_weight_norms,
    compute_total_weight_norm,
    compute_representation_norm,
    compute_per_layer_representation_norms,
    spectral_smoothness,
    compute_jacobian_norm,
    compute_hessian_trace,
    compute_per_layer_jacobian_norms,
    compute_per_layer_hessian_traces,
    # Weight-curvature metrics (loss landscape)
    compute_gradient_norm,
    compute_weight_hessian_trace,
    compute_fisher_trace,
    # Adam optimizer dynamics
    compute_adam_metrics,
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

    # Per-layer metrics (new)
    compute_per_layer_metrics: bool = False  # Enable per-layer curvature metrics

    # Weight curvature metrics (loss landscape)
    compute_weight_curvature: bool = True       # Enable gradient_norm, weight_hessian, fisher
    weight_curvature_interval: int = 100        # Compute every N steps (expensive)
    weight_hessian_samples: int = 10            # Hutchinson samples for Hessian trace

    # Adam optimizer dynamics
    compute_optimizer_metrics: bool = True      # Enable Adam state analysis

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


class MetricsComputer:
    """
    Encapsulates metric computation logic for training loops.

    Separates metric computation from the training loop for cleaner code
    and easier testing/extension.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: TrainerConfig,
        is_transformer: bool = False,
        optimizer: Optional[optim.Optimizer] = None,
    ):
        self.model = model
        self.device = device
        self.config = config
        self.is_transformer = is_transformer
        self.optimizer = optimizer

    def compute_routing_metrics(
        self,
        routing_weights: Optional[list[Tensor]],
    ) -> tuple[float, list[float]]:
        """
        Compute routing entropy and head utilization.

        Args:
            routing_weights: List of [batch, n_heads] tensors per layer

        Returns:
            Tuple of (routing_entropy, head_utilization_list)
        """
        if routing_weights is None:
            return 0.0, []

        routing_entropy = sum(
            -(w * w.log().clamp(min=-100)).sum(dim=-1).mean().item()
            for w in routing_weights
        ) / len(routing_weights)

        head_util = torch.stack(routing_weights).mean(dim=(0, 1)).tolist()
        return routing_entropy, head_util

    def compute_spectral_smoothness(self) -> float:
        """Compute spectral smoothness of current model."""
        k = self.config.spectral_k or (self.config.p // 4)
        return spectral_smoothness(
            self.model, self.config.p, k, self.device,
            is_transformer=self.is_transformer
        )

    def compute_curvature_metrics(
        self,
        sample_inputs: Tensor,
    ) -> tuple[float, float]:
        """
        Compute Jacobian norm and Hessian trace.

        Args:
            sample_inputs: Subset of inputs for efficiency

        Returns:
            Tuple of (jacobian_norm, hessian_trace)
        """
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
        return jac_norm, hess_trace

    def compute_per_layer_curvature(
        self,
        sample_inputs: Tensor,
    ) -> tuple[list[float], list[float]]:
        """
        Compute per-layer Jacobian norms and Hessian traces.

        Args:
            sample_inputs: Subset of inputs for efficiency

        Returns:
            Tuple of (layer_jacobian_norms, layer_hessian_traces)
        """
        layer_jac_norms = compute_per_layer_jacobian_norms(
            self.model, sample_inputs,
            num_samples=5,  # Fewer samples for per-layer (expensive)
            is_transformer=self.is_transformer,
        )
        layer_hess_traces = compute_per_layer_hessian_traces(
            self.model, sample_inputs,
            num_hutchinson_samples=3,
            is_transformer=self.is_transformer,
        )
        return layer_jac_norms, layer_hess_traces

    def compute_weight_metrics(self) -> tuple[list[float], float]:
        """
        Compute layer and total weight norms.

        Returns:
            Tuple of (layer_weight_norms, total_weight_norm)
        """
        layer_norms = compute_layer_weight_norms(self.model)
        total_norm = compute_total_weight_norm(self.model)
        return layer_norms, total_norm

    def compute_representation_metrics(
        self,
        inputs: Tensor,
    ) -> tuple[float, Optional[list[float]]]:
        """
        Compute total and optionally per-layer representation norms.

        Args:
            inputs: Input tensor

        Returns:
            Tuple of (representation_norm, layer_representation_norms or None)
        """
        repr_norm = compute_representation_norm(self.model, inputs)

        layer_repr_norms = None
        if self.config.compute_per_layer_metrics:
            layer_repr_norms = compute_per_layer_representation_norms(self.model, inputs)

        return repr_norm, layer_repr_norms

    def compute_weight_curvature_metrics(
        self,
        inputs: Tensor,
        targets: Tensor,
    ) -> tuple[float, float, float]:
        """
        Compute weight-based curvature metrics (loss landscape analysis).

        These measure the loss surface geometry w.r.t. model weights:
        - gradient_norm: ||grad_w L|| - magnitude of loss gradient
        - weight_hessian_trace: Tr(grad^2_w L) - curvature of loss surface
        - fisher_trace: Tr(grad L * grad L^T) - empirical Fisher information

        Args:
            inputs: Training inputs for loss computation
            targets: Training targets for loss computation

        Returns:
            Tuple of (gradient_norm, weight_hessian_trace, fisher_trace)
        """
        # Loss function for curvature computation
        def loss_fn(logits: Tensor, targets: Tensor) -> Tensor:
            return nn.functional.cross_entropy(logits, targets)

        # Sample subset for efficiency
        sample_size = min(256, len(inputs))
        sample_indices = torch.randperm(len(inputs), device=inputs.device)[:sample_size]
        sample_inputs = inputs[sample_indices]
        sample_targets = targets[sample_indices]

        # Gradient norm (fast - single forward/backward)
        grad_norm = compute_gradient_norm(
            self.model, loss_fn, sample_inputs, sample_targets
        )

        # Weight Hessian trace (expensive - Hutchinson estimator)
        hessian_trace = compute_weight_hessian_trace(
            self.model, loss_fn, sample_inputs, sample_targets,
            num_hutchinson_samples=self.config.weight_hessian_samples,
        )

        # Fisher trace (moderate - per-sample gradients)
        fisher = compute_fisher_trace(
            self.model, loss_fn, sample_inputs, sample_targets,
            max_samples=32,
        )

        return grad_norm, hessian_trace, fisher

    def compute_optimizer_metrics(
        self,
    ) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Compute Adam optimizer dynamics metrics.

        Extracts and analyzes the internal state of Adam/AdamW:
        - effective_lr_mean/max: sqrt(v_t) statistics (adaptive LR scaling)
        - adam_ratio_mean/max: |m_t|/(sqrt(v_t)+eps) (signal-to-noise)
        - update_decay_ratio: ||grad update|| / ||weight decay||

        Returns:
            Tuple of (effective_lr_mean, effective_lr_max, adam_ratio_mean,
                     adam_ratio_max, update_decay_ratio) or all None if unavailable
        """
        if self.optimizer is None:
            return None, None, None, None, None

        adam_metrics = compute_adam_metrics(
            self.optimizer,
            weight_decay=self.config.weight_decay,
            eps=self.config.eps,
        )

        if adam_metrics is None:
            return None, None, None, None, None

        return (
            adam_metrics.effective_lr_mean,
            adam_metrics.effective_lr_max,
            adam_metrics.adam_ratio_mean,
            adam_metrics.adam_ratio_max,
            adam_metrics.update_decay_ratio,
        )

    def compute_all_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        test_loss: float,
        test_acc: float,
        routing_weights: Optional[list[Tensor]],
        test_inputs: Tensor,
        train_inputs: Optional[Tensor] = None,
        train_targets: Optional[Tensor] = None,
    ) -> TrainingMetrics:
        """
        Compute all metrics and return a TrainingMetrics object.

        Args:
            epoch: Current training epoch
            train_loss: Training loss
            train_acc: Training accuracy
            test_loss: Test loss
            test_acc: Test accuracy
            routing_weights: Routing weights from model (if applicable)
            test_inputs: Test inputs for curvature/representation metrics
            train_inputs: Training inputs for weight curvature metrics
            train_targets: Training targets for weight curvature metrics

        Returns:
            TrainingMetrics with all computed metrics
        """
        # Routing metrics
        routing_entropy, head_util = self.compute_routing_metrics(routing_weights)

        # Spectral smoothness
        smoothness = self.compute_spectral_smoothness()

        # Sample inputs for curvature computation (for efficiency)
        sample_size = min(256, len(test_inputs))
        sample_indices = torch.randperm(len(test_inputs))[:sample_size]
        sample_inputs = test_inputs[sample_indices]

        # Input-sensitivity curvature metrics (Jacobian, Hessian w.r.t. inputs)
        jac_norm, hess_trace = self.compute_curvature_metrics(sample_inputs)

        # Per-layer curvature metrics (if enabled)
        layer_jac_norms = None
        layer_hess_traces = None
        if self.config.compute_per_layer_metrics:
            layer_jac_norms, layer_hess_traces = self.compute_per_layer_curvature(sample_inputs)

        # Weight metrics
        layer_norms, total_norm = self.compute_weight_metrics()

        # Representation metrics
        repr_norm, layer_repr_norms = self.compute_representation_metrics(test_inputs)

        # Weight-curvature metrics (loss landscape - periodic due to expense)
        gradient_norm = None
        weight_hessian_trace = None
        fisher_trace = None
        if (
            self.config.compute_weight_curvature
            and train_inputs is not None
            and train_targets is not None
            and epoch % self.config.weight_curvature_interval == 0
        ):
            gradient_norm, weight_hessian_trace, fisher_trace = \
                self.compute_weight_curvature_metrics(train_inputs, train_targets)

        # Adam optimizer dynamics
        effective_lr_mean = None
        effective_lr_max = None
        adam_ratio_mean = None
        adam_ratio_max = None
        update_decay_ratio = None
        if self.config.compute_optimizer_metrics:
            (
                effective_lr_mean,
                effective_lr_max,
                adam_ratio_mean,
                adam_ratio_max,
                update_decay_ratio,
            ) = self.compute_optimizer_metrics()

        return TrainingMetrics(
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
            layer_jacobian_norms=layer_jac_norms,
            layer_hessian_traces=layer_hess_traces,
            layer_representation_norms=layer_repr_norms,
            # Weight-curvature metrics
            gradient_norm=gradient_norm,
            weight_hessian_trace=weight_hessian_trace,
            fisher_trace=fisher_trace,
            # Adam optimizer dynamics
            effective_lr_mean=effective_lr_mean,
            effective_lr_max=effective_lr_max,
            adam_ratio_mean=adam_ratio_mean,
            adam_ratio_max=adam_ratio_max,
            update_decay_ratio=update_decay_ratio,
        )


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

        # Metrics computer
        self.metrics_computer = MetricsComputer(
            model=self.model,
            device=self.device,
            config=config,
            is_transformer=self.is_transformer,
            optimizer=self.optimizer,
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

    def compute_spectral_smoothness(self) -> float:
        """
        Compute spectral smoothness of the learned function.

        Returns:
            Spectral smoothness value in [0, 1], where higher means smoother.
        """
        K = self.config.spectral_k or self.config.p // 4
        return spectral_smoothness(
            self.model,
            self.config.p,
            K,
            self.device,
            is_transformer=self.is_transformer,
        )

    # ─── Helper Methods ───────────────────────────────────────────────────────

    def _apply_warmup(self, epoch: int) -> None:
        """Apply learning rate warmup if configured."""
        if self.config.warmup_epochs > 0 and epoch < self.config.warmup_epochs:
            warmup_factor = (epoch + 1) / self.config.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.lr * warmup_factor
        elif epoch == self.config.warmup_epochs and self.config.warmup_epochs > 0:
            # Restore full learning rate after warmup
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.lr

    def _log_metrics(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        routing_weights: Optional[list[Tensor]],
    ) -> TrainingMetrics:
        """
        Compute and log all metrics for the current epoch.

        Returns:
            TrainingMetrics object with all computed metrics
        """
        test_loss, test_acc = self.evaluate()

        metrics = self.metrics_computer.compute_all_metrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            test_loss=test_loss,
            test_acc=test_acc,
            routing_weights=routing_weights,
            test_inputs=self.test_inputs,
            train_inputs=self.train_inputs,
            train_targets=self.train_targets,
        )

        # Determine if routing should be saved
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

        return metrics

    def _update_best_model(self, epoch: int, test_acc: float) -> None:
        """Update best model tracking and save checkpoint if improved."""
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.best_epoch = epoch
            self.save_checkpoint("best.pt", epoch=epoch)

    def _print_progress(
        self,
        metrics: TrainingMetrics,
        routing_weights: Optional[list[Tensor]],
    ) -> None:
        """Print training progress to stdout."""
        routing_str = f", entropy={metrics.routing_entropy:.3f}" if routing_weights else ""
        print(
            f"Epoch {metrics.step:7d} | "
            f"Train: {metrics.train_acc*100:5.1f}% | "
            f"Test: {metrics.test_acc*100:5.1f}% | "
            f"Loss: {metrics.train_loss:.2e}, "
            f"rnorm={metrics.representation_norm:.1f}, smooth={metrics.spectral_smoothness:.3f}, "
            f"jac={metrics.jacobian_norm:.2e}, |hess|={abs(metrics.hessian_trace):.2e}"
            f"{routing_str}"
        )

    def _print_training_header(self, start_epoch: int) -> None:
        """Print training configuration header."""
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

    def _print_training_summary(self) -> None:
        """Print training completion summary."""
        print()
        print(f"Training complete. Best test acc: {self.best_test_acc:.4f} at epoch {self.best_epoch}")

    # ─── Main Training Loop ───────────────────────────────────────────────────

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
        self._print_training_header(start_epoch)

        for epoch in range(start_epoch, self.config.epochs):
            self._apply_warmup(epoch)

            # Training step
            train_loss, train_acc, routing_weights = self.train_step(epoch)

            # Periodic logging
            if epoch % self.config.log_every == 0:
                metrics = self._log_metrics(epoch, train_loss, train_acc, routing_weights)
                self._update_best_model(epoch, metrics.test_acc)
                self._print_progress(metrics, routing_weights)

                if callback is not None:
                    callback(epoch, metrics)

            # Periodic checkpointing
            if epoch % self.config.checkpoint_every == 0 and epoch > 0:
                self.save_checkpoint(f"checkpoint_{epoch}.pt", epoch=epoch)

        # Final save
        self.save_checkpoint("final.pt", epoch=self.config.epochs - 1)
        self.save_history()

        self._print_training_summary()

        return self.history

    # ─── Checkpointing ────────────────────────────────────────────────────────

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
