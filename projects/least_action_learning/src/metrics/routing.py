"""Routing-specific metrics for routed networks."""

import torch
from torch import Tensor
from dataclasses import dataclass


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
