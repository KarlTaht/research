"""Training metrics containers: TrainingMetrics and MetricsHistory."""

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingMetrics:
    """Container for all training metrics at a given step."""

    # Required fields - always tracked
    step: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    routing_entropy: float
    head_utilization: list[float]

    # Optional fields - computed on demand
    spectral_smoothness: Optional[float] = None
    layer_weight_norms: Optional[list[float]] = None
    total_weight_norm: Optional[float] = None
    decayed_weight_norm: Optional[float] = None  # Excludes embeddings, LN, output head
    representation_norm: Optional[float] = None
    jacobian_norm: Optional[float] = None
    hessian_trace: Optional[float] = None

    # Per-layer curvature metrics
    layer_jacobian_norms: Optional[list[float]] = None
    layer_hessian_traces: Optional[list[float]] = None
    layer_representation_norms: Optional[list[float]] = None

    # Weight-based curvature metrics (loss landscape analysis)
    gradient_norm: Optional[float] = None        # ||∇_w L|| - gradient w.r.t. weights
    weight_hessian_trace: Optional[float] = None # Tr(∇²_w L) - Hessian trace w.r.t. weights
    fisher_trace: Optional[float] = None         # Tr(∇L · ∇Lᵀ) - empirical Fisher trace

    # Adam optimizer dynamics
    effective_lr_mean: Optional[float] = None   # Mean of sqrt(v_t) - adaptive LR scaling
    effective_lr_max: Optional[float] = None    # Max of sqrt(v_t) - hottest learning
    adam_ratio_mean: Optional[float] = None     # Mean of |m_t| / (sqrt(v_t) + eps)
    adam_ratio_max: Optional[float] = None      # Max ratio - most confident direction
    update_decay_ratio: Optional[float] = None  # ||grad update|| / ||weight decay||

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/serialization."""
        d = {
            "step": self.step,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
            "routing_entropy": self.routing_entropy,
        }

        # Head utilization (expand to individual columns)
        for i, util in enumerate(self.head_utilization):
            d[f"head_{i}_utilization"] = util

        # Optional scalar metrics
        if self.spectral_smoothness is not None:
            d["spectral_smoothness"] = self.spectral_smoothness
        if self.total_weight_norm is not None:
            d["total_weight_norm"] = self.total_weight_norm
        if self.decayed_weight_norm is not None:
            d["decayed_weight_norm"] = self.decayed_weight_norm
        if self.representation_norm is not None:
            d["representation_norm"] = self.representation_norm
        if self.jacobian_norm is not None:
            d["jacobian_norm"] = self.jacobian_norm
        if self.hessian_trace is not None:
            d["hessian_trace"] = self.hessian_trace

        # Weight-based curvature metrics
        if self.gradient_norm is not None:
            d["gradient_norm"] = self.gradient_norm
        if self.weight_hessian_trace is not None:
            d["weight_hessian_trace"] = self.weight_hessian_trace
        if self.fisher_trace is not None:
            d["fisher_trace"] = self.fisher_trace

        # Adam optimizer dynamics
        if self.effective_lr_mean is not None:
            d["effective_lr_mean"] = self.effective_lr_mean
        if self.effective_lr_max is not None:
            d["effective_lr_max"] = self.effective_lr_max
        if self.adam_ratio_mean is not None:
            d["adam_ratio_mean"] = self.adam_ratio_mean
        if self.adam_ratio_max is not None:
            d["adam_ratio_max"] = self.adam_ratio_max
        if self.update_decay_ratio is not None:
            d["update_decay_ratio"] = self.update_decay_ratio

        # Per-layer weight norms (expand to individual columns)
        if self.layer_weight_norms is not None:
            for i, norm in enumerate(self.layer_weight_norms):
                d[f"layer_{i}_weight_norm"] = norm

        # Per-layer jacobian norms (new)
        if self.layer_jacobian_norms is not None:
            for i, norm in enumerate(self.layer_jacobian_norms):
                d[f"layer_{i}_jacobian_norm"] = norm

        # Per-layer hessian traces (new)
        if self.layer_hessian_traces is not None:
            for i, trace in enumerate(self.layer_hessian_traces):
                d[f"layer_{i}_hessian_trace"] = trace

        # Per-layer representation norms (new)
        if self.layer_representation_norms is not None:
            for i, norm in enumerate(self.layer_representation_norms):
                d[f"layer_{i}_representation_norm"] = norm

        return d


class MetricsHistory:
    """Accumulates metrics over training."""

    def __init__(self):
        self.history: list[TrainingMetrics] = []
        self.routing_weights_history: list[list[Tensor]] = []

    def log(self, metrics: TrainingMetrics, routing_weights: Optional[list[Tensor]] = None):
        """Log metrics and optionally routing weights."""
        self.history.append(metrics)
        if routing_weights is not None:
            # Detach and clone to avoid memory issues
            self.routing_weights_history.append(
                [w.detach().cpu().clone() for w in routing_weights]
            )

    def get_dataframe(self):
        """Convert history to pandas DataFrame."""
        import pandas as pd
        records = [m.to_dict() for m in self.history]
        return pd.DataFrame(records)

    def get_grokking_step(self, accuracy_threshold: float = 0.95) -> Optional[int]:
        """
        Find the step where test accuracy first exceeds threshold.

        Returns None if threshold never reached.
        """
        for m in self.history:
            if m.test_acc >= accuracy_threshold:
                return m.step
        return None

    def get_memorization_step(self, accuracy_threshold: float = 0.98) -> Optional[int]:
        """
        Find the step where train accuracy first exceeds threshold.

        Returns None if threshold never reached.
        """
        for m in self.history:
            if m.train_acc >= accuracy_threshold:
                return m.step
        return None

    def __len__(self) -> int:
        """Return number of logged metrics."""
        return len(self.history)

    def __getitem__(self, idx: int) -> TrainingMetrics:
        """Get metrics at specific index."""
        return self.history[idx]
