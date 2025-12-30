"""Least Action Learning: Efficient routing through neural networks."""

from .data import ModularArithmeticDataset, SequenceArithmeticDataset
from .models import BaselineMLP, RoutedNetwork, GrokTransformer
from .routing import RoutingGate, RoutedLayer
from .losses import (
    entropy_regularizer,
    sparsity_regularizer,
    spectral_smoothness,
    compute_jacobian_norm,
    compute_hessian_trace,
    jacobian_regularizer,
)
from .metrics import (
    RoutingMetrics,
    compute_layer_weight_norms,
    compute_total_weight_norm,
    compute_representation_norm,
)

__all__ = [
    "ModularArithmeticDataset",
    "SequenceArithmeticDataset",
    "BaselineMLP",
    "RoutedNetwork",
    "GrokTransformer",
    "RoutingGate",
    "RoutedLayer",
    "entropy_regularizer",
    "sparsity_regularizer",
    "spectral_smoothness",
    "compute_jacobian_norm",
    "compute_hessian_trace",
    "jacobian_regularizer",
    "RoutingMetrics",
    "compute_layer_weight_norms",
    "compute_total_weight_norm",
    "compute_representation_norm",
]
