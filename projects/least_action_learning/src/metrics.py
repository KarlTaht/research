"""Metrics tracking for routing experiments."""

import torch
from torch import Tensor
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RoutingMetrics:
    """
    Container for routing-related metrics from a forward pass.

    Attributes:
        layer_weights: List of routing weight tensors per layer [n_layers][batch, n_heads]
        routing_entropy: Average entropy of routing distributions
        head_utilization: Average weight per head across all inputs
    """
    layer_weights: list[Tensor]
    routing_entropy: float
    head_utilization: Tensor  # [n_heads]

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "routing_entropy": self.routing_entropy,
            "head_utilization": self.head_utilization.tolist(),
            "n_layers": len(self.layer_weights),
        }


def compute_routing_entropy(weights: Tensor, eps: float = 1e-8) -> float:
    """
    Compute average entropy of routing weight distributions.

    Args:
        weights: Routing weights [batch, n_heads] or [n_layers, batch, n_heads]
        eps: Small constant for numerical stability

    Returns:
        Average entropy (lower = more decisive routing)
    """
    # Clamp for numerical stability
    weights = weights.clamp(min=eps, max=1 - eps)

    # Compute entropy: -sum(w * log(w))
    entropy = -(weights * weights.log()).sum(dim=-1)

    return entropy.mean().item()


def compute_head_utilization(layer_weights: list[Tensor]) -> Tensor:
    """
    Compute average utilization per head across all layers and inputs.

    Args:
        layer_weights: List of [batch, n_heads] tensors per layer

    Returns:
        Average weight per head [n_heads]
    """
    # Stack all layers: [n_layers, batch, n_heads]
    stacked = torch.stack(layer_weights, dim=0)

    # Average over layers and batch
    return stacked.mean(dim=(0, 1))


def compute_routing_consistency(
    layer_weights: list[Tensor],
    pairs: Tensor,
    similar_mask: Tensor,
) -> float:
    """
    Compute how consistent routing is for similar inputs.

    Args:
        layer_weights: List of [batch, n_heads] tensors per layer
        pairs: Input pairs [batch, 2]
        similar_mask: Boolean mask [batch, batch] where True means inputs are similar

    Returns:
        Average variance of routing weights for similar inputs (lower = more consistent)
    """
    # Stack weights: [n_layers, batch, n_heads]
    stacked = torch.stack(layer_weights, dim=0)

    # Flatten to [batch, n_layers * n_heads]
    flat = stacked.permute(1, 0, 2).reshape(stacked.shape[1], -1)

    # For each pair of similar inputs, compute distance
    total_variance = 0.0
    count = 0

    for i in range(len(pairs)):
        similar_indices = similar_mask[i].nonzero().squeeze(-1)
        if len(similar_indices) > 1:
            similar_weights = flat[similar_indices]
            variance = similar_weights.var(dim=0).mean()
            total_variance += variance.item()
            count += 1

    return total_variance / max(count, 1)


def get_dominant_head_per_layer(layer_weights: list[Tensor]) -> Tensor:
    """
    Get the dominant (highest weight) head for each input at each layer.

    Args:
        layer_weights: List of [batch, n_heads] tensors per layer

    Returns:
        Dominant head indices [n_layers, batch]
    """
    dominant = []
    for weights in layer_weights:
        dominant.append(weights.argmax(dim=-1))
    return torch.stack(dominant, dim=0)


def get_routing_path_signature(layer_weights: list[Tensor]) -> Tensor:
    """
    Create a signature of the routing path for clustering.

    Concatenates dominant head indices across layers into a path signature.

    Args:
        layer_weights: List of [batch, n_heads] tensors per layer

    Returns:
        Path signatures [batch, n_layers]
    """
    dominant = get_dominant_head_per_layer(layer_weights)
    return dominant.T  # [batch, n_layers]


def compute_layer_weight_norms(model: torch.nn.Module) -> list[float]:
    """
    Compute L2 norm of weights for each layer in the model.

    For baseline MLP: returns norm of each Linear layer's weight matrix.
    For routed networks: returns norm of each RoutedLayer's combined weights.

    Args:
        model: The neural network model

    Returns:
        List of L2 norms, one per layer
    """
    norms = []

    # Handle different model types
    if hasattr(model, 'net'):
        # BaselineMLP - extract Linear layers from Sequential
        for module in model.net:
            if isinstance(module, torch.nn.Linear):
                norm = module.weight.norm(2).item()
                norms.append(norm)
    elif hasattr(model, 'layers'):
        # RoutedNetwork - get norms from each routed layer
        # Also include embed and output head
        if hasattr(model, 'embed'):
            norms.append(model.embed.weight.norm(2).item())
        for layer in model.layers:
            # Sum norms of all heads in the layer
            layer_norm = 0.0
            for name, param in layer.named_parameters():
                if 'weight' in name:
                    layer_norm += param.norm(2).item() ** 2
            norms.append(layer_norm ** 0.5)
        if hasattr(model, 'output_head'):
            norms.append(model.output_head.weight.norm(2).item())
    else:
        # Fallback: iterate all parameters
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                norms.append(param.norm(2).item())

    return norms


def compute_total_weight_norm(model: torch.nn.Module) -> float:
    """
    Compute total L2 norm of all weights in the model.

    Args:
        model: The neural network model

    Returns:
        Total L2 norm (sqrt of sum of squared norms)
    """
    total_sq = 0.0
    for param in model.parameters():
        total_sq += param.norm(2).item() ** 2
    return total_sq ** 0.5


@dataclass
class TrainingMetrics:
    """Container for all training metrics at a given step."""
    step: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    routing_entropy: float
    head_utilization: list[float]
    spectral_smoothness: Optional[float] = None
    layer_weight_norms: Optional[list[float]] = None
    total_weight_norm: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        d = {
            "step": self.step,
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "test_loss": self.test_loss,
            "test_acc": self.test_acc,
            "routing_entropy": self.routing_entropy,
        }
        for i, util in enumerate(self.head_utilization):
            d[f"head_{i}_utilization"] = util
        if self.spectral_smoothness is not None:
            d["spectral_smoothness"] = self.spectral_smoothness
        if self.layer_weight_norms is not None:
            for i, norm in enumerate(self.layer_weight_norms):
                d[f"layer_{i}_weight_norm"] = norm
        if self.total_weight_norm is not None:
            d["total_weight_norm"] = self.total_weight_norm
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
