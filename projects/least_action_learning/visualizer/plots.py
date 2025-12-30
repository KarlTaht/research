"""Plotly visualization functions for routing analysis."""

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from torch import Tensor

# Color palette for heads (categorical)
HEAD_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]


def create_empty_figure(message: str = "No data available") -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="gray"),
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
    )
    return fig


def create_training_curves(history_df: pd.DataFrame) -> go.Figure:
    """
    Create dual-axis plot of loss and accuracy over training.

    Args:
        history_df: DataFrame with step, train_loss, test_loss, train_acc, test_acc

    Returns:
        Plotly figure
    """
    if history_df.empty:
        return create_empty_figure("No training history")

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Loss curves (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["train_loss"],
            mode="lines",
            name="Train Loss",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Step: %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["test_loss"],
            mode="lines",
            name="Test Loss",
            line=dict(color="#1f77b4", width=2, dash="dash"),
            hovertemplate="Step: %{x}<br>Test Loss: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Accuracy curves (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["train_acc"],
            mode="lines",
            name="Train Acc",
            line=dict(color="#2ca02c", width=2),
            hovertemplate="Step: %{x}<br>Train Acc: %{y:.2%}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["test_acc"],
            mode="lines",
            name="Test Acc",
            line=dict(color="#2ca02c", width=2, dash="dash"),
            hovertemplate="Step: %{x}<br>Test Acc: %{y:.2%}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Add 95% accuracy threshold line
    fig.add_hline(
        y=0.95,
        line_dash="dot",
        line_color="gray",
        annotation_text="95%",
        secondary_y=True,
    )

    fig.update_layout(
        title="Training Curves",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Loss", type="log", secondary_y=False)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], secondary_y=True)

    return fig


def create_routing_entropy_curve(history_df: pd.DataFrame) -> go.Figure:
    """
    Create plot of routing entropy over training.

    Args:
        history_df: DataFrame with step and routing_entropy columns

    Returns:
        Plotly figure
    """
    if history_df.empty or "routing_entropy" not in history_df.columns:
        return create_empty_figure("No routing entropy data")

    entropy = history_df["routing_entropy"]
    if entropy.max() == 0:
        return create_empty_figure("No routing entropy data")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=entropy,
            mode="lines",
            name="Routing Entropy",
            line=dict(color="#9467bd", width=2),
            fill="tozeroy",
            fillcolor="rgba(148, 103, 189, 0.2)",
            hovertemplate="Step: %{x}<br>Entropy: %{y:.4f}<extra></extra>",
        )
    )

    # Add max entropy reference line (log(n_heads))
    max_entropy = np.log(4)  # Assuming 4 heads; will be approximate
    fig.add_hline(
        y=max_entropy,
        line_dash="dot",
        line_color="gray",
        annotation_text="Max (uniform)",
    )

    fig.update_layout(
        title="Routing Entropy",
        xaxis_title="Step",
        yaxis_title="Entropy",
        hovermode="x unified",
    )

    return fig


def create_routing_heatmap(
    routing_weights: list[Tensor],
    p: int,
    layer_idx: int = 0,
    n_heads: int = 4,
) -> go.Figure:
    """
    Create heatmap of dominant routing head for each input pair.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer
        p: Prime modulus
        layer_idx: Which layer to visualize
        n_heads: Number of heads

    Returns:
        Plotly figure with p×p heatmap
    """
    if not routing_weights or layer_idx >= len(routing_weights):
        return create_empty_figure("No routing data for this layer")

    weights = routing_weights[layer_idx]
    if isinstance(weights, Tensor):
        weights = weights.cpu().numpy()

    # Get dominant head for each input
    dominant = np.argmax(weights, axis=-1)

    # Reshape to p×p grid (assumes weights are ordered by (a,b) pairs)
    batch_size = len(dominant)
    if batch_size == p * p:
        grid = dominant.reshape(p, p)
    else:
        # Partial data - can't create full grid
        return create_empty_figure(f"Incomplete data: {batch_size} samples, need {p*p}")

    # Create hover text
    hover_text = []
    for a in range(p):
        row = []
        for b in range(p):
            head = grid[a, b]
            weight = weights[a * p + b, head]
            row.append(f"a={a}, b={b}<br>Head: {head}<br>Weight: {weight:.3f}")
        hover_text.append(row)

    # Create discrete colorscale for heads
    colorscale = []
    for i in range(n_heads):
        val = i / (n_heads - 1) if n_heads > 1 else 0
        colorscale.append([val, HEAD_COLORS[i % len(HEAD_COLORS)]])

    fig = go.Figure(
        data=go.Heatmap(
            z=grid,
            x=list(range(p)),
            y=list(range(p)),
            colorscale=colorscale,
            zmin=0,
            zmax=n_heads - 1,
            text=hover_text,
            hoverinfo="text",
            colorbar=dict(
                title="Head",
                tickmode="array",
                tickvals=list(range(n_heads)),
                ticktext=[f"Head {i}" for i in range(n_heads)],
            ),
        )
    )

    fig.update_layout(
        title=f"Dominant Head at Layer {layer_idx}",
        xaxis_title="b",
        yaxis_title="a",
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def create_head_utilization(
    routing_weights: list[Tensor],
    n_heads: int = 4,
) -> go.Figure:
    """
    Create bar chart of head utilization across all layers.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer
        n_heads: Number of heads

    Returns:
        Plotly figure with grouped bar chart
    """
    if not routing_weights:
        return create_empty_figure("No routing data")

    # Compute average utilization per head per layer
    n_layers = len(routing_weights)
    utilization = np.zeros((n_layers, n_heads))

    for layer_idx, weights in enumerate(routing_weights):
        if isinstance(weights, Tensor):
            weights = weights.cpu().numpy()
        utilization[layer_idx] = weights.mean(axis=0)

    fig = go.Figure()

    for head_idx in range(n_heads):
        fig.add_trace(
            go.Bar(
                name=f"Head {head_idx}",
                x=[f"Layer {i}" for i in range(n_layers)],
                y=utilization[:, head_idx],
                marker_color=HEAD_COLORS[head_idx % len(HEAD_COLORS)],
                hovertemplate="Layer %{x}<br>Utilization: %{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Head Utilization by Layer",
        xaxis_title="Layer",
        yaxis_title="Average Weight",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_spectral_plot(
    output_grid: np.ndarray,
    p: int,
) -> go.Figure:
    """
    Create spectral analysis plot showing output function and power spectrum.

    Args:
        output_grid: p×p array of predicted outputs
        p: Prime modulus

    Returns:
        Plotly figure with subplots
    """
    if output_grid is None:
        return create_empty_figure("No output data")

    # Compute FFT
    spectrum = np.fft.fft2(output_grid.astype(float))
    power = np.abs(spectrum) ** 2
    power_shifted = np.fft.fftshift(power)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Predicted Outputs", "Log Power Spectrum"],
    )

    # Output function
    fig.add_trace(
        go.Heatmap(
            z=output_grid,
            colorscale="Viridis",
            colorbar=dict(title="Output", x=0.45),
            hovertemplate="a=%{y}, b=%{x}<br>Output: %{z}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Power spectrum (log scale)
    fig.add_trace(
        go.Heatmap(
            z=np.log(power_shifted + 1),
            colorscale="Hot",
            colorbar=dict(title="Log Power", x=1.0),
            hovertemplate="freq_x=%{x}, freq_y=%{y}<br>Log Power: %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Spectral Analysis",
        height=400,
    )

    fig.update_xaxes(title_text="b", row=1, col=1)
    fig.update_yaxes(title_text="a", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (x)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency (y)", row=1, col=2)

    return fig


def create_routing_evolution_comparison(
    snapshots: list[dict],
    p: int,
    layer_idx: int,
    n_snapshots: int = 4,
) -> go.Figure:
    """
    Create comparison of routing patterns at different training stages.

    Args:
        snapshots: List of routing snapshot dicts with 'step' and 'weights' keys
        p: Prime modulus
        layer_idx: Which layer to visualize
        n_snapshots: Number of snapshots to show

    Returns:
        Plotly figure with subplot per snapshot
    """
    if not snapshots:
        return create_empty_figure("No routing snapshots available")

    # Select evenly spaced snapshots
    n_available = len(snapshots)
    if n_available < n_snapshots:
        indices = list(range(n_available))
    else:
        indices = np.linspace(0, n_available - 1, n_snapshots, dtype=int).tolist()

    selected = [snapshots[i] for i in indices]

    fig = make_subplots(
        rows=1,
        cols=len(selected),
        subplot_titles=[f"Step {s.get('step', i)}" for i, s in enumerate(selected)],
    )

    for col_idx, snapshot in enumerate(selected, 1):
        weights = snapshot.get("weights", [])
        if layer_idx >= len(weights):
            continue

        layer_weights = weights[layer_idx]
        if isinstance(layer_weights, Tensor):
            layer_weights = layer_weights.cpu().numpy()

        dominant = np.argmax(layer_weights, axis=-1)
        batch_size = len(dominant)

        if batch_size == p * p:
            grid = dominant.reshape(p, p)
        else:
            continue

        n_heads = layer_weights.shape[-1]
        colorscale = []
        for i in range(n_heads):
            val = i / (n_heads - 1) if n_heads > 1 else 0
            colorscale.append([val, HEAD_COLORS[i % len(HEAD_COLORS)]])

        fig.add_trace(
            go.Heatmap(
                z=grid,
                colorscale=colorscale,
                zmin=0,
                zmax=n_heads - 1,
                showscale=(col_idx == len(selected)),  # Only show colorbar on last
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title=f"Routing Evolution at Layer {layer_idx}",
        height=300,
    )

    return fig
