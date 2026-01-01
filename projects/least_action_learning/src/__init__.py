"""Least Action Learning: Efficient routing through neural networks."""

from .data import ModularArithmeticDataset, SequenceArithmeticDataset
from .models import BaselineMLP, RoutedNetwork, GrokTransformer
from .routing import RoutingGate, RoutedLayer
from .losses import (
    entropy_regularizer,
    sparsity_regularizer,
    jacobian_regularizer,
)
from .metrics import (
    # Routing
    RoutingMetrics,
    compute_routing_entropy,
    compute_head_utilization,
    # Model properties
    compute_layer_weight_norms,
    compute_total_weight_norm,
    compute_representation_norm,
    compute_per_layer_representation_norms,
    # Curvature
    spectral_smoothness,
    compute_jacobian_norm,
    compute_hessian_trace,
    compute_per_layer_jacobian_norms,
    compute_per_layer_hessian_traces,
    # Training
    TrainingMetrics,
    MetricsHistory,
)

__all__ = [
    # Data
    "ModularArithmeticDataset",
    "SequenceArithmeticDataset",
    # Models
    "BaselineMLP",
    "RoutedNetwork",
    "GrokTransformer",
    # Routing
    "RoutingGate",
    "RoutedLayer",
    # Loss functions
    "entropy_regularizer",
    "sparsity_regularizer",
    "jacobian_regularizer",
    # Metrics - routing
    "RoutingMetrics",
    "compute_routing_entropy",
    "compute_head_utilization",
    # Metrics - model properties
    "compute_layer_weight_norms",
    "compute_total_weight_norm",
    "compute_representation_norm",
    "compute_per_layer_representation_norms",
    # Metrics - curvature
    "spectral_smoothness",
    "compute_jacobian_norm",
    "compute_hessian_trace",
    "compute_per_layer_jacobian_norms",
    "compute_per_layer_hessian_traces",
    # Metrics - training
    "TrainingMetrics",
    "MetricsHistory",
]
