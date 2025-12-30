"""Visualization utilities for routing analysis."""

import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import Normalize
from typing import Optional, Union
from pathlib import Path

from .metrics import MetricsHistory, get_dominant_head_per_layer


def plot_training_curves(
    history: MetricsHistory,
    figsize: tuple[int, int] = (12, 4),
) -> Figure:
    """
    Plot training loss and accuracy curves.

    Args:
        history: Training metrics history
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    df = history.get_dataframe()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Loss curve
    axes[0].plot(df["step"], df["train_loss"], label="Train", alpha=0.8)
    axes[0].plot(df["step"], df["test_loss"], label="Test", alpha=0.8)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    axes[0].set_yscale("log")

    # Accuracy curve
    axes[1].plot(df["step"], df["train_acc"], label="Train", alpha=0.8)
    axes[1].plot(df["step"], df["test_acc"], label="Test", alpha=0.8)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    axes[1].axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95%")

    # Routing entropy (if available)
    if "routing_entropy" in df.columns and df["routing_entropy"].max() > 0:
        axes[2].plot(df["step"], df["routing_entropy"], alpha=0.8)
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Entropy")
        axes[2].set_title("Routing Entropy")

        # Mark grokking point if it exists
        grokking_step = history.get_grokking_step()
        if grokking_step is not None:
            for ax in axes:
                ax.axvline(x=grokking_step, color="red", linestyle="--", alpha=0.5)
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "No routing data", ha="center", va="center")

    plt.tight_layout()
    return fig


def plot_routing_heatmap(
    routing_weights: list[Tensor],
    pairs: Tensor,
    p: int,
    layer_idx: int = 0,
    figsize: tuple[int, int] = (8, 6),
) -> Figure:
    """
    Plot heatmap of dominant routing head for each input pair.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer
        pairs: Input pairs [batch, 2]
        p: Prime modulus
        layer_idx: Which layer to visualize
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    weights = routing_weights[layer_idx]
    dominant = weights.argmax(dim=-1).cpu().numpy()
    n_heads = weights.shape[1]

    # Create p x p grid
    grid = np.zeros((p, p))
    pairs_np = pairs.cpu().numpy()

    for idx, (a, b) in enumerate(pairs_np):
        grid[a, b] = dominant[idx]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(grid, cmap="tab10", vmin=0, vmax=n_heads - 1)
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"Dominant Head at Layer {layer_idx} (a + b mod {p})")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_heads))
    cbar.set_label("Head Index")

    return fig


def plot_routing_weights_heatmap(
    routing_weights: list[Tensor],
    pairs: Tensor,
    p: int,
    layer_idx: int = 0,
    head_idx: int = 0,
    figsize: tuple[int, int] = (8, 6),
) -> Figure:
    """
    Plot heatmap of routing weight for a specific head.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer
        pairs: Input pairs [batch, 2]
        p: Prime modulus
        layer_idx: Which layer to visualize
        head_idx: Which head's weight to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    weights = routing_weights[layer_idx][:, head_idx].cpu().numpy()

    # Create p x p grid
    grid = np.zeros((p, p))
    pairs_np = pairs.cpu().numpy()

    for idx, (a, b) in enumerate(pairs_np):
        grid[a, b] = weights[idx]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1)
    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"Head {head_idx} Weight at Layer {layer_idx}")

    plt.colorbar(im, ax=ax, label="Routing Weight")

    return fig


def plot_head_utilization(
    history: MetricsHistory,
    figsize: tuple[int, int] = (10, 4),
) -> Figure:
    """
    Plot head utilization over training.

    Args:
        history: Training metrics history
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    df = history.get_dataframe()

    # Find head utilization columns
    head_cols = [c for c in df.columns if c.startswith("head_") and c.endswith("_utilization")]
    if not head_cols:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No head utilization data", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=figsize)

    for col in head_cols:
        head_idx = col.split("_")[1]
        ax.plot(df["step"], df[col], label=f"Head {head_idx}", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Utilization")
    ax.set_title("Head Utilization Over Training")
    ax.legend()

    # Mark grokking point
    grokking_step = history.get_grokking_step()
    if grokking_step is not None:
        ax.axvline(x=grokking_step, color="red", linestyle="--", alpha=0.5, label="Grokking")

    return fig


def plot_routing_evolution(
    history: MetricsHistory,
    p: int,
    layer_idx: int = 0,
    n_snapshots: int = 4,
    figsize: tuple[int, int] = (16, 4),
) -> Figure:
    """
    Plot routing patterns at multiple training stages.

    Args:
        history: Training metrics history (must include routing weights)
        p: Prime modulus
        layer_idx: Which layer to visualize
        n_snapshots: Number of snapshots to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if not history.routing_weights_history:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No routing weight history", ha="center", va="center")
        return fig

    n_saved = len(history.routing_weights_history)
    indices = np.linspace(0, n_saved - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=figsize)

    for i, idx in enumerate(indices):
        routing_weights = history.routing_weights_history[idx]
        weights = routing_weights[layer_idx]
        dominant = weights.argmax(dim=-1).cpu().numpy()
        n_heads = weights.shape[1]

        # Reshape to grid (assuming full dataset evaluation)
        # This requires knowing the order of inputs
        batch_size = len(dominant)
        if batch_size == p * p:
            grid = dominant.reshape(p, p)
        else:
            # Partial data - create sparse grid
            grid = np.full((p, p), -1)
            # Would need pairs tensor to fill correctly

        im = axes[i].imshow(grid, cmap="tab10", vmin=0, vmax=n_heads - 1)
        step = history.history[idx].step if idx < len(history.history) else idx
        axes[i].set_title(f"Step {step}")
        axes[i].set_xlabel("b")
        if i == 0:
            axes[i].set_ylabel("a")

    plt.suptitle(f"Routing Evolution at Layer {layer_idx}")
    plt.tight_layout()
    return fig


def plot_routing_by_target(
    routing_weights: list[Tensor],
    targets: Tensor,
    p: int,
    layer_idx: int = 0,
    figsize: tuple[int, int] = (12, 4),
) -> Figure:
    """
    Analyze routing patterns grouped by target output.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer
        targets: Target values [batch]
        p: Prime modulus
        layer_idx: Which layer to analyze
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    weights = routing_weights[layer_idx]
    n_heads = weights.shape[1]

    # Group by target
    target_groups = {}
    for target in range(p):
        mask = targets == target
        if mask.any():
            target_groups[target] = weights[mask].mean(dim=0).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Heatmap of average routing by target
    routing_matrix = np.zeros((p, n_heads))
    for target, avg_routing in target_groups.items():
        routing_matrix[target] = avg_routing

    im = axes[0].imshow(routing_matrix, cmap="viridis", aspect="auto")
    axes[0].set_xlabel("Head")
    axes[0].set_ylabel("Target Output")
    axes[0].set_title(f"Average Routing by Target (Layer {layer_idx})")
    plt.colorbar(im, ax=axes[0], label="Weight")

    # Variance of routing within each target group
    variance_by_target = []
    for target in range(p):
        mask = targets == target
        if mask.any():
            var = weights[mask].var(dim=0).mean().item()
            variance_by_target.append(var)
        else:
            variance_by_target.append(0)

    axes[1].bar(range(p), variance_by_target, alpha=0.7)
    axes[1].set_xlabel("Target Output")
    axes[1].set_ylabel("Routing Variance")
    axes[1].set_title("Routing Consistency by Target")

    plt.tight_layout()
    return fig


def plot_spectral_analysis(
    model: torch.nn.Module,
    p: int,
    device: torch.device,
    figsize: tuple[int, int] = (10, 4),
    is_transformer: bool = False,
) -> Figure:
    """
    Plot spectral analysis of model's output function.

    Args:
        model: Trained model
        p: Prime modulus
        device: Device to run on
        figsize: Figure size
        is_transformer: If True, create sequence inputs for transformer

    Returns:
        Matplotlib figure
    """
    model.eval()

    with torch.no_grad():
        # Create all inputs
        a_vals = torch.arange(p, device=device)
        b_vals = torch.arange(p, device=device)
        aa, bb = torch.meshgrid(a_vals, b_vals, indexing='ij')
        pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)
        batch_size = p * p

        if is_transformer:
            # Create sequence inputs: [a, op, b, =]
            op_token = p      # operation token
            eq_token = p + 1  # equals token
            inputs = torch.stack([
                pairs[:, 0].long(),
                torch.full((batch_size,), op_token, dtype=torch.long, device=device),
                pairs[:, 1].long(),
                torch.full((batch_size,), eq_token, dtype=torch.long, device=device),
            ], dim=1)
        else:
            # One-hot encode for MLP
            inputs = torch.zeros(batch_size, 2 * p, device=device)
            inputs[torch.arange(batch_size), pairs[:, 0]] = 1.0
            inputs[torch.arange(batch_size), p + pairs[:, 1]] = 1.0

        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        predicted = logits.argmax(dim=-1).float().view(p, p).cpu().numpy()

        # Compute FFT
        spectrum = np.fft.fft2(predicted)
        power = np.abs(spectrum) ** 2

        # Shift to center low frequencies
        power_shifted = np.fft.fftshift(power)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Output function
    im0 = axes[0].imshow(predicted, cmap="viridis")
    axes[0].set_xlabel("b")
    axes[0].set_ylabel("a")
    axes[0].set_title("Predicted Output")
    plt.colorbar(im0, ax=axes[0])

    # Power spectrum
    im1 = axes[1].imshow(np.log(power_shifted + 1), cmap="hot")
    axes[1].set_xlabel("Frequency (x)")
    axes[1].set_ylabel("Frequency (y)")
    axes[1].set_title("Log Power Spectrum")
    plt.colorbar(im1, ax=axes[1])

    # Radial power distribution
    center = p // 2
    y, x = np.ogrid[:p, :p]
    r = np.sqrt((x - center)**2 + (y - center)**2).astype(int)

    radial_power = np.bincount(r.ravel(), power_shifted.ravel())
    radial_counts = np.bincount(r.ravel())
    radial_mean = radial_power / np.maximum(radial_counts, 1)

    axes[2].plot(radial_mean[:p//2], 'b-', linewidth=2)
    axes[2].set_xlabel("Frequency Magnitude")
    axes[2].set_ylabel("Average Power")
    axes[2].set_title("Radial Power Distribution")
    axes[2].set_yscale("log")

    plt.tight_layout()
    return fig


def save_all_visualizations(
    history: MetricsHistory,
    model: torch.nn.Module,
    dataset,
    p: int,
    device: torch.device,
    output_dir: Union[str, Path],
):
    """
    Save all visualization plots to directory.

    Args:
        history: Training metrics history
        model: Trained model
        dataset: Dataset with pairs
        p: Prime modulus
        device: Device to run on
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect if model is a transformer
    is_transformer = hasattr(model, 'token_embedding')

    # Training curves
    fig = plot_training_curves(history)
    fig.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Head utilization
    fig = plot_head_utilization(history)
    fig.savefig(output_dir / "head_utilization.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Spectral analysis
    fig = plot_spectral_analysis(model, p, device, is_transformer=is_transformer)
    fig.savefig(output_dir / "spectral_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Routing heatmaps (if we have routing data)
    if history.routing_weights_history:
        # Get final routing weights
        all_data = dataset.get_all()
        # Handle both sequence (transformer) and one-hot (MLP) datasets
        if hasattr(all_data, 'input_ids'):
            inputs = all_data.input_ids.to(device)
        else:
            inputs = all_data.inputs.to(device)
        pairs = all_data.pairs

        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                _, metrics = outputs
                routing_weights = metrics.layer_weights

                n_layers = len(routing_weights)
                for layer_idx in range(n_layers):
                    fig = plot_routing_heatmap(routing_weights, pairs, p, layer_idx)
                    fig.savefig(output_dir / f"routing_layer_{layer_idx}.png", dpi=150, bbox_inches="tight")
                    plt.close(fig)

    print(f"Saved visualizations to {output_dir}")
