"""Least Action Learning: Efficient routing through neural networks."""

from .data import ModularArithmeticDataset, SequenceArithmeticDataset
from .models import BaselineMLP, RoutedNetwork, GrokTransformer
from .routing import RoutingGate, RoutedLayer
from .losses import entropy_regularizer, sparsity_regularizer, spectral_smoothness
from .metrics import RoutingMetrics, compute_layer_weight_norms, compute_total_weight_norm

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
    "RoutingMetrics",
    "compute_layer_weight_norms",
    "compute_total_weight_norm",
]
