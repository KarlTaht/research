"""Model property metrics: weight norms and representation norms."""

import torch
from torch import Tensor
from typing import Optional


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


def compute_decayed_weight_norm(model: torch.nn.Module) -> float:
    """
    Compute L2 norm of weights that receive weight decay.

    Excludes parameters typically not decayed in AdamW:
    - Embeddings (token_embedding, pos_embedding)
    - LayerNorm parameters
    - Biases
    - Output head (typically excluded in grokking experiments)

    This matches the weight decay exclusions in the trainer and provides
    a more meaningful metric for understanding weight decay effects.

    Args:
        model: The neural network model

    Returns:
        L2 norm of decayed weights only
    """
    total_sq = 0.0

    for name, param in model.named_parameters():
        # Skip if not 2D (biases are 1D)
        if param.dim() < 2:
            continue

        # Skip embeddings
        if "embedding" in name.lower() or "embed" in name.lower():
            continue

        # Skip LayerNorm
        if "ln" in name.lower() or "layernorm" in name.lower() or "layer_norm" in name.lower():
            continue

        # Skip output head
        if "output_head" in name.lower() or "output" in name.lower():
            continue

        # Skip biases (explicit check)
        if "bias" in name.lower():
            continue

        # Include this parameter
        total_sq += param.norm(2).item() ** 2

    return total_sq ** 0.5


def compute_layer_weight_norms_decayed(model: torch.nn.Module) -> dict[str, float]:
    """
    Compute weight norms for layers that receive weight decay, with descriptive names.

    Returns a dict with descriptive layer names as keys, making it easier
    to understand which layers contribute to weight norm.

    Args:
        model: The neural network model

    Returns:
        Dict mapping layer name to L2 norm (only for decayed layers)
    """
    norms = {}

    for name, param in model.named_parameters():
        # Skip if not 2D weight matrix
        if param.dim() < 2:
            continue

        # Skip embeddings
        if "embedding" in name.lower() or "embed" in name.lower():
            continue

        # Skip LayerNorm
        if "ln" in name.lower() or "layernorm" in name.lower() or "layer_norm" in name.lower():
            continue

        # Skip output head
        if "output_head" in name.lower() or "output" in name.lower():
            continue

        # Skip biases
        if "bias" in name.lower():
            continue

        # Create a readable name
        readable_name = name.replace(".weight", "").replace("blocks.", "b").replace(".", "_")
        norms[readable_name] = param.norm(2).item()

    return norms


def compute_representation_norm(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> float:
    """
    Compute the mean L2 norm of representations before unembedding.

    This measures the magnitude of the final hidden state (after all
    transformer blocks / MLP layers, before the output projection).
    This is more meaningful for analyzing grokking dynamics than
    weight norms, as it captures how the model organizes its
    internal representations.

    Args:
        model: The neural network model (must have get_representation method)
        inputs: Input tensor (one-hot for MLP, token IDs for transformer)

    Returns:
        Mean L2 norm of representations across the batch
    """
    with torch.no_grad():
        representations = model.get_representation(inputs)
        # Compute L2 norm for each sample, then take mean
        norms = representations.norm(2, dim=-1)  # [batch]
        return norms.mean().item()


def compute_per_layer_representation_norms(
    model: torch.nn.Module,
    inputs: torch.Tensor,
) -> list[float]:
    """
    Compute L2 norm of representations at each layer's output.

    For transformers: after each transformer block
    For MLPs: after each hidden layer (post-activation)
    For routed networks: after each routed layer

    Args:
        model: The neural network model
        inputs: Input tensor (one-hot for MLP, token IDs for transformer)

    Returns:
        List of mean L2 norms, one per layer
    """
    norms = []

    with torch.no_grad():
        if hasattr(model, 'get_per_layer_representations'):
            # Model provides a method for this
            representations = model.get_per_layer_representations(inputs)
            for rep in representations:
                layer_norms = rep.norm(2, dim=-1)  # [batch] or [batch, seq_len]
                norms.append(layer_norms.mean().item())
        elif hasattr(model, 'layers'):
            # RoutedNetwork or similar - manually track through layers
            if hasattr(model, 'embed'):
                x = model.embed(inputs)
                norms.append(x.norm(2, dim=-1).mean().item())
            else:
                x = inputs

            for layer in model.layers:
                x, _ = layer(x)
                norms.append(x.norm(2, dim=-1).mean().item())
        elif hasattr(model, 'net'):
            # BaselineMLP - track through Sequential
            x = inputs
            for module in model.net:
                x = module(x)
                # Log norm after each activation (or linear if no activation follows)
                if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
                    norms.append(x.norm(2, dim=-1).mean().item())
                elif isinstance(module, torch.nn.Linear) and module == model.net[-1]:
                    # Final linear layer (no activation after)
                    norms.append(x.norm(2, dim=-1).mean().item())
        else:
            # Fallback: just return final representation
            rep = model.get_representation(inputs)
            norms.append(rep.norm(2, dim=-1).mean().item())

    return norms
