"""Model architecture utilities for analysis.

Provides layer naming and grouping based on model architecture.
"""

from typing import Union

from .loader import ExperimentData


def infer_model_type(config: dict) -> str:
    """Infer model type from configuration.

    Args:
        config: Experiment config dict

    Returns:
        Model type string: "transformer", "baseline", "routed", or "single_head"
    """
    return config.get("model_type", "transformer")


def get_layer_names(
    model_type: str,
    n_layers: int,
    n_heads: int = 4,
) -> dict[int, str]:
    """Get descriptive layer names by index.

    Maps layer indices (from compute_layer_weight_norms) to descriptive names
    based on the model architecture.

    Args:
        model_type: One of "transformer", "baseline", "routed", "single_head"
        n_layers: Number of layers/blocks
        n_heads: Number of heads (for routed models)

    Returns:
        Dict mapping layer index (int) to descriptive name (str)

    Example:
        >>> names = get_layer_names("transformer", n_layers=2)
        >>> names[0]
        'tok_embed'
        >>> names[2]
        'b0_attn_Q'
    """
    if model_type == "transformer":
        return _get_transformer_layer_names(n_layers)
    elif model_type == "baseline":
        return _get_baseline_layer_names(n_layers)
    elif model_type in ("routed", "single_head"):
        return _get_routed_layer_names(n_layers)
    else:
        # Generic fallback
        return {i: f"layer_{i}" for i in range(n_layers + 1)}


def _get_transformer_layer_names(n_layers: int) -> dict[int, str]:
    """Generate descriptive layer names for transformer architecture.

    For a transformer with n_layers blocks, the parameter order is:
    - 0: token_embedding
    - 1: pos_embedding
    - For each block i:
        - 2 + i*6 + 0: block{i}_attn_Q
        - 2 + i*6 + 1: block{i}_attn_K
        - 2 + i*6 + 2: block{i}_attn_V
        - 2 + i*6 + 3: block{i}_attn_Wo
        - 2 + i*6 + 4: block{i}_ffn_up
        - 2 + i*6 + 5: block{i}_ffn_down
    - Final: output_head

    Args:
        n_layers: Number of transformer blocks

    Returns:
        Dict mapping layer index to descriptive name
    """
    names = {}

    # Input embeddings
    names[0] = "tok_embed"
    names[1] = "pos_embed"

    # Transformer blocks
    for block_idx in range(n_layers):
        base = 2 + block_idx * 6
        block_prefix = f"b{block_idx}"
        names[base + 0] = f"{block_prefix}_attn_Q"
        names[base + 1] = f"{block_prefix}_attn_K"
        names[base + 2] = f"{block_prefix}_attn_V"
        names[base + 3] = f"{block_prefix}_attn_Wo"
        names[base + 4] = f"{block_prefix}_ffn_up"
        names[base + 5] = f"{block_prefix}_ffn_down"

    # Output
    final_idx = 2 + n_layers * 6
    names[final_idx] = "output_head"

    return names


def _get_baseline_layer_names(n_layers: int) -> dict[int, str]:
    """Generate layer names for baseline MLP.

    Args:
        n_layers: Number of hidden layers

    Returns:
        Dict mapping layer index to name (linear_0, linear_1, ..., output)
    """
    names = {}
    for i in range(n_layers):
        names[i] = f"linear_{i}"
    names[n_layers] = "output"
    return names


def _get_routed_layer_names(n_layers: int) -> dict[int, str]:
    """Generate layer names for routed network.

    Args:
        n_layers: Number of routed layers

    Returns:
        Dict mapping layer index to name (embed, routed_0, ..., output)
    """
    names = {0: "embed"}
    for i in range(n_layers):
        names[i + 1] = f"routed_{i}"
    names[n_layers + 1] = "output"
    return names


def get_layer_display_name(
    layer_idx: Union[int, str],
    experiment: ExperimentData,
) -> str:
    """Get display name for a layer based on experiment config.

    Args:
        layer_idx: Layer index (int or string)
        experiment: ExperimentData with config

    Returns:
        Descriptive layer name (e.g., "b0_attn_Q") or fallback "layer_{idx}"
    """
    idx = int(layer_idx)
    model_type = experiment.model_type
    n_layers = experiment.n_layers

    layer_names = get_layer_names(model_type, n_layers)
    return layer_names.get(idx, f"layer_{idx}")


def get_layer_groups(
    model_type: str,
    n_layers: int,
) -> dict[str, tuple[list[int], list[str]]]:
    """Get layer indices grouped by component type.

    Useful for plotting weight norms by architectural component.

    Args:
        model_type: One of "transformer", "baseline", "routed"
        n_layers: Number of layers/blocks

    Returns:
        Dict mapping group name to (layer_indices, display_names) tuple

    Example:
        >>> groups = get_layer_groups("transformer", n_layers=2)
        >>> groups["embeddings"]
        ([0, 1], ["tok_embed", "pos_embed"])
        >>> groups["block0_attn"]
        ([2, 3, 4, 5], ["Q", "K", "V", "Wo"])
    """
    if model_type == "transformer":
        return _get_transformer_layer_groups(n_layers)
    elif model_type == "baseline":
        return _get_baseline_layer_groups(n_layers)
    elif model_type in ("routed", "single_head"):
        return _get_routed_layer_groups(n_layers)
    else:
        # Generic fallback
        layer_names = get_layer_names(model_type, n_layers)
        return {"all": (list(layer_names.keys()), list(layer_names.values()))}


def _get_transformer_layer_groups(n_layers: int) -> dict[str, tuple[list[int], list[str]]]:
    """Get layer groups for transformer architecture.

    Args:
        n_layers: Number of transformer blocks

    Returns:
        Dict mapping group name to (indices, names)
    """
    groups = {}

    # Embeddings: tok_embed (0), pos_embed (1)
    groups["embeddings"] = ([0, 1], ["tok_embed", "pos_embed"])

    # Per-block groups
    for block_idx in range(n_layers):
        base = 2 + block_idx * 6

        # Attention: Q, K, V, Wo
        attn_indices = [base + 0, base + 1, base + 2, base + 3]
        attn_names = ["Q", "K", "V", "Wo"]
        groups[f"block{block_idx}_attn"] = (attn_indices, attn_names)

        # FFN: up, down
        ffn_indices = [base + 4, base + 5]
        ffn_names = ["FFN_up", "FFN_down"]
        groups[f"block{block_idx}_ffn"] = (ffn_indices, ffn_names)

    # Output head
    output_idx = 2 + n_layers * 6
    groups["output"] = ([output_idx], ["output_head"])

    return groups


def _get_baseline_layer_groups(n_layers: int) -> dict[str, tuple[list[int], list[str]]]:
    """Get layer groups for baseline MLP.

    Args:
        n_layers: Number of hidden layers

    Returns:
        Dict mapping group name to (indices, names)
    """
    hidden_indices = list(range(n_layers))
    hidden_names = [f"linear_{i}" for i in range(n_layers)]

    return {
        "hidden": (hidden_indices, hidden_names),
        "output": ([n_layers], ["output"]),
    }


def _get_routed_layer_groups(n_layers: int) -> dict[str, tuple[list[int], list[str]]]:
    """Get layer groups for routed network.

    Args:
        n_layers: Number of routed layers

    Returns:
        Dict mapping group name to (indices, names)
    """
    routed_indices = list(range(1, n_layers + 1))
    routed_names = [f"routed_{i}" for i in range(n_layers)]

    return {
        "embed": ([0], ["embed"]),
        "routed": (routed_indices, routed_names),
        "output": ([n_layers + 1], ["output"]),
    }


def get_layer_choices(
    experiment: ExperimentData,
    metric_suffix: str = "weight_norm",
) -> list[str]:
    """Get list of layer choices for UI dropdown based on available data.

    Args:
        experiment: ExperimentData instance
        metric_suffix: Metric suffix to look for (e.g., "weight_norm")

    Returns:
        List of layer choices including "All" and descriptive names
        Format: ["All", "0:tok_embed", "1:pos_embed", ...]
    """
    df = experiment.history_df
    layer_cols = [
        c for c in df.columns if c.startswith("layer_") and c.endswith(f"_{metric_suffix}")
    ]

    if not layer_cols:
        return ["All"]

    choices = ["All"]
    for col in sorted(layer_cols):
        # Extract layer number from column name like "layer_0_weight_norm"
        layer_num = col.split("_")[1]
        display_name = get_layer_display_name(layer_num, experiment)
        # Use format "idx:name" so we can parse it back
        choices.append(f"{layer_num}:{display_name}")

    return choices
