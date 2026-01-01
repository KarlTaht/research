"""
Metrics package for least action learning.

This package provides metrics organized by concern:
- routing: Routing-specific metrics for routed networks
- model_properties: Weight norms and representation norms
- curvature: Spectral smoothness, Jacobian norms, Hessian traces
- training: Container classes for metrics accumulation
"""

from .routing import (
    RoutingMetrics,
    compute_routing_entropy,
    compute_head_utilization,
    compute_routing_consistency,
    get_dominant_head_per_layer,
    get_routing_path_signature,
)

from .model_properties import (
    compute_layer_weight_norms,
    compute_total_weight_norm,
    compute_decayed_weight_norm,
    compute_layer_weight_norms_decayed,
    compute_representation_norm,
    compute_per_layer_representation_norms,
)

from .curvature import (
    # Input-sensitivity metrics
    make_low_freq_mask,
    spectral_smoothness,
    compute_jacobian_norm,
    compute_hessian_trace,
    compute_per_layer_jacobian_norms,
    compute_per_layer_hessian_traces,
    # Weight-curvature metrics (loss landscape)
    compute_gradient_norm,
    compute_weight_hessian_trace,
    compute_fisher_trace,
)

from .training import (
    TrainingMetrics,
    MetricsHistory,
)

from .optimizer import (
    AdamMetrics,
    compute_adam_metrics,
)

__all__ = [
    # Routing
    "RoutingMetrics",
    "compute_routing_entropy",
    "compute_head_utilization",
    "compute_routing_consistency",
    "get_dominant_head_per_layer",
    "get_routing_path_signature",
    # Model properties
    "compute_layer_weight_norms",
    "compute_total_weight_norm",
    "compute_decayed_weight_norm",
    "compute_layer_weight_norms_decayed",
    "compute_representation_norm",
    "compute_per_layer_representation_norms",
    # Input-sensitivity curvature
    "make_low_freq_mask",
    "spectral_smoothness",
    "compute_jacobian_norm",
    "compute_hessian_trace",
    "compute_per_layer_jacobian_norms",
    "compute_per_layer_hessian_traces",
    # Weight-curvature metrics (loss landscape)
    "compute_gradient_norm",
    "compute_weight_hessian_trace",
    "compute_fisher_trace",
    # Training
    "TrainingMetrics",
    "MetricsHistory",
    # Optimizer dynamics
    "AdamMetrics",
    "compute_adam_metrics",
]
