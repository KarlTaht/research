"""Plotly visualization functions for routing analysis."""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch import Tensor

from .styles import (
    CATEGORICAL_COLORS,
    HEAD_COLORS,
    get_metric_display_name,
    should_use_log_scale,
)
from .helpers import create_empty_figure, get_valid_data

if TYPE_CHECKING:
    from .data import ExperimentRun


def create_training_curves(history_df: pd.DataFrame) -> go.Figure:
    """
    Create stacked plots of accuracy (top) and loss (bottom) over training.

    Args:
        history_df: DataFrame with step, train_loss, test_loss, train_acc, test_acc

    Returns:
        Plotly figure with two subplots
    """
    if history_df.empty:
        return create_empty_figure("No training history")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Accuracy", "Loss"),
    )

    # Accuracy curves (top plot, linear scale)
    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["train_acc"],
            mode="lines",
            name="Train Acc",
            line=dict(color="#2ca02c", width=2),
            hovertemplate="Step: %{x}<br>Train Acc: %{y:.2%}<extra></extra>",
        ),
        row=1,
        col=1,
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
        row=1,
        col=1,
    )

    # Add threshold lines for accuracy
    fig.add_hline(
        y=0.95,
        line_dash="dot",
        line_color="gray",
        annotation_text="95%",
        annotation_position="right",
        row=1,
        col=1,
    )

    # Loss curves (bottom plot, log scale)
    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["train_loss"],
            mode="lines",
            name="Train Loss",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Step: %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
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
        row=2,
        col=1,
    )

    fig.update_layout(
        height=500,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Accuracy plot: linear scale, 0-1 range
    fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=1, col=1)

    # Loss plot: log scale
    fig.update_yaxes(title_text="Loss", type="log", row=2, col=1)

    # Only show x-axis label on bottom plot
    fig.update_xaxes(title_text="Step", row=2, col=1)

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


def create_grokking_summary_table(analysis_df: pd.DataFrame) -> go.Figure:
    """
    Create interactive Plotly table showing grokking quality for all experiments.

    Args:
        analysis_df: DataFrame from analyze_all_experiments()

    Returns:
        Plotly figure with table including ID, Group, and Split columns
    """
    if analysis_df.empty:
        return create_empty_figure("No experiments to analyze")

    # Format columns for display
    df = analysis_df.copy()

    # Add sequential ID column
    df.insert(0, "ID", range(1, len(df) + 1))

    # Ensure group column exists (for backward compatibility)
    if "group" not in df.columns:
        df["group"] = "no group"

    # Format train_frac as train/test ratio (e.g., "50/50" or "30/70")
    if "train_frac" in df.columns:
        df["Split"] = df["train_frac"].apply(
            lambda x: f"{int(x*100)}/{int((1-x)*100)}" if pd.notna(x) else "50/50"
        )
    else:
        df["Split"] = "50/50"

    df["Train≥98%"] = df["pct_train_above_98"].apply(lambda x: f"{x:.1f}%")
    df["Test≥95%"] = df["pct_test_above_95"].apply(lambda x: f"{x:.1f}%")
    df["Final Test"] = (df["final_test_acc"] * 100).round(1).astype(str) + "%"
    df["Variance"] = df["test_variance"].apply(
        lambda x: f"{x:.2e}" if pd.notna(x) and x > 0 else "-"
    )
    df["Grok Step"] = df["grok_step"].apply(lambda x: str(int(x)) if pd.notna(x) else "-")
    df["lr_fmt"] = df["lr"].apply(lambda x: f"{x:.0e}" if x > 0 else "-")
    df["wd_fmt"] = df["weight_decay"].apply(lambda x: f"{x:.1f}" if x > 0 else "-")

    # Row colors based on grokking status
    # Green for grokked (test≥95% for most of post-grok period)
    # Yellow for partial/in-progress
    # Red for not grokked
    def get_row_color(grok_step, pct_test):
        has_grokked = pd.notna(grok_step)
        if has_grokked and pct_test >= 90:
            return "#d4edda"  # Light green - successfully grokked
        elif has_grokked or pct_test >= 50:
            return "#fff3cd"  # Yellow - partial/in-progress
        else:
            return "#f8d7da"  # Light red - not grokked

    row_colors = [[get_row_color(g, p) for g, p in zip(df["grok_step"], df["pct_test_above_95"])]]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "ID",
                        "Group",
                        "Name",
                        "p",
                        "Split",
                        "lr",
                        "wd",
                        "Grok Step",
                        "Train≥98%",
                        "Test≥95%",
                        "Final Test",
                        "Variance",
                    ],
                    fill_color="#343a40",
                    font=dict(color="white", size=12),
                    align="left",
                    height=30,
                ),
                cells=dict(
                    values=[
                        df["ID"],
                        df["group"],
                        df["name"],
                        df["p"],
                        df["Split"],
                        df["lr_fmt"],
                        df["wd_fmt"],
                        df["Grok Step"],
                        df["Train≥98%"],
                        df["Test≥95%"],
                        df["Final Test"],
                        df["Variance"],
                    ],
                    fill_color=row_colors * 12,  # Repeat for all columns
                    align="left",
                    height=25,
                    font=dict(size=11),
                ),
            )
        ]
    )

    n_rows = len(df)
    height = max(200, min(600, 60 + n_rows * 28))
    fig.update_layout(
        title="Grokking Quality Summary (sorted by % test≥95%, then % train≥98%)",
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def create_accuracy_comparison(
    experiments: list["ExperimentRun"],
) -> go.Figure:
    """
    Create overlay plot of train and test accuracy from multiple experiments.

    Args:
        experiments: List of ExperimentRun objects to compare

    Returns:
        Plotly figure with overlaid train (solid) and test (dashed) curves
    """
    if not experiments:
        return create_empty_figure("Select experiments to compare")

    fig = go.Figure()
    has_data = False

    for i, exp in enumerate(experiments):
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        df = exp.history_df

        # Plot train accuracy (solid line)
        y_data, valid_mask = get_valid_data(df, "train_acc")
        if valid_mask.any():
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, "train_acc"],
                    mode="lines",
                    name=f"{exp.name} (train)",
                    line=dict(color=color, width=2),
                    legendgroup=exp.name,
                    hovertemplate=f"{exp.name}<br>Step: %{{x}}<br>Train: %{{y:.2%}}<extra></extra>",
                )
            )

        # Plot test accuracy (dashed line)
        y_data, valid_mask = get_valid_data(df, "test_acc")
        if valid_mask.any():
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, "test_acc"],
                    mode="lines",
                    name=f"{exp.name} (test)",
                    line=dict(color=color, width=2, dash="dash"),
                    legendgroup=exp.name,
                    hovertemplate=f"{exp.name}<br>Step: %{{x}}<br>Test: %{{y:.2%}}<extra></extra>",
                )
            )

    if not has_data:
        return create_empty_figure("No accuracy data")

    # Add threshold lines
    fig.add_hline(
        y=0.95,
        line_dash="dot",
        line_color="gray",
        annotation_text="95%",
        annotation_position="right",
    )
    fig.add_hline(
        y=0.98,
        line_dash="dot",
        line_color="lightgray",
        annotation_text="98%",
        annotation_position="right",
    )

    fig.update_layout(
        title="Accuracy (solid=train, dashed=test)",
        xaxis_title="Step",
        yaxis_title="Accuracy",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=400,
    )

    fig.update_yaxes(range=[0, 1.05])

    return fig


def create_multi_experiment_comparison(
    experiments: list["ExperimentRun"],
    metric: str = "test_acc",
) -> go.Figure:
    """
    Create overlay plot of training curves from multiple experiments.

    Args:
        experiments: List of ExperimentRun objects to compare
        metric: Column to plot (test_acc, train_acc, test_loss, total_weight_norm, etc.)

    Returns:
        Plotly figure with overlaid curves
    """
    if not experiments:
        return create_empty_figure("Select experiments to compare")

    fig = go.Figure()
    has_data = False

    for i, exp in enumerate(experiments):
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        df = exp.history_df

        if metric not in df.columns:
            continue

        # Filter out NaN/inf values for cleaner plots
        y_data, valid_mask = get_valid_data(df, metric)

        if not valid_mask.any():
            continue

        has_data = True
        fig.add_trace(
            go.Scatter(
                x=df.loc[valid_mask, "step"],
                y=df.loc[valid_mask, metric],
                mode="lines",
                name=exp.name,
                line=dict(color=color, width=2),
                hovertemplate=f"{exp.name}<br>Step: %{{x}}<br>{metric}: %{{y:.4g}}<extra></extra>",
            )
        )

    if not has_data:
        return create_empty_figure(f"No data for {metric}")

    # Add threshold lines for accuracy metrics
    if metric in ("test_acc", "train_acc"):
        fig.add_hline(
            y=0.95,
            line_dash="dot",
            line_color="gray",
            annotation_text="95%",
            annotation_position="right",
        )
        if metric == "train_acc":
            fig.add_hline(
                y=0.98,
                line_dash="dot",
                line_color="lightgray",
                annotation_text="98%",
                annotation_position="right",
            )

    # Format title using centralized metric names
    metric_display = get_metric_display_name(metric)

    fig.update_layout(
        title=metric_display,
        xaxis_title="Step",
        yaxis_title=metric_display,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        height=350,
    )

    # Set appropriate y-axis settings
    if metric in ("test_acc", "train_acc"):
        fig.update_yaxes(range=[0, 1.05])
    elif should_use_log_scale(metric):
        fig.update_yaxes(type="log")

    return fig


# =============================================================================
# Fourier Analysis Plots
# =============================================================================


def create_embedding_spectrum_plot(
    freq_power: np.ndarray,
    key_frequencies: list[int],
    p: int,
) -> go.Figure:
    """Create bar chart of embedding Fourier power by frequency.

    Args:
        freq_power: [p] total power at each frequency
        key_frequencies: List of key frequency indices to highlight
        p: Prime modulus

    Returns:
        Plotly figure with bar chart
    """
    if freq_power is None or len(freq_power) == 0:
        return create_empty_figure("No embedding spectrum data")

    # Only show unique frequencies (k and p-k are conjugates)
    max_k = (p + 1) // 2
    x_vals = list(range(max_k))
    y_vals = freq_power[:max_k].tolist()

    # Color bars based on whether they're key frequencies
    colors = ["#2ca02c" if k in key_frequencies else "#1f77b4" for k in range(max_k)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=colors,
            hovertemplate="Frequency k=%{x}<br>Power: %{y:.4g}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Embedding Fourier Spectrum (p={p})",
        xaxis_title="Frequency k",
        yaxis_title="Power (sum over d_model)",
        height=350,
        showlegend=False,
    )

    # Add annotation for key frequencies
    if key_frequencies:
        key_str = ", ".join(str(k) for k in sorted(key_frequencies)[:5])
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Key: {key_str}",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#2ca02c",
            borderwidth=1,
        )

    return fig


def create_logit_spectrum_heatmap(
    power_2d: np.ndarray,
    p: int,
) -> go.Figure:
    """Create 2D heatmap of logit Fourier power spectrum.

    Args:
        power_2d: [p, p] 2D power spectrum
        p: Prime modulus

    Returns:
        Plotly figure with heatmap
    """
    if power_2d is None or power_2d.size == 0:
        return create_empty_figure("No logit spectrum data")

    # Only show relevant portion (DC to p//2 in each dimension)
    max_k = (p + 1) // 2
    power_crop = power_2d[:max_k, :max_k]

    # Log scale for better visualization
    power_log = np.log10(power_crop + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=power_log,
            x=list(range(max_k)),
            y=list(range(max_k)),
            colorscale="Hot",
            colorbar=dict(title="log₁₀(P+1)"),
            hovertemplate="k_a=%{y}, k_b=%{x}<br>log₁₀(Power+1): %{z:.2f}<extra></extra>",
        )
    )

    # Add diagonal line annotation (where a+b signal lives)
    fig.add_trace(
        go.Scatter(
            x=list(range(max_k)),
            y=list(range(max_k)),
            mode="lines",
            line=dict(color="cyan", width=2, dash="dash"),
            name="(a+b) diagonal",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=f"2D Logit Fourier Spectrum (p={p})",
        xaxis_title="Frequency k_b",
        yaxis_title="Frequency k_a",
        height=400,
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )

    return fig


def create_logit_diagonal_plot(
    diag_power: np.ndarray,
    key_frequencies: list[int],
    p: int,
) -> go.Figure:
    """Create line plot of power along the (a+b) diagonal frequencies.

    Args:
        diag_power: [p] power at each diagonal frequency
        key_frequencies: Key frequency indices to highlight
        p: Prime modulus

    Returns:
        Plotly figure with line plot and highlighted key frequencies
    """
    if diag_power is None or len(diag_power) == 0:
        return create_empty_figure("No diagonal power data")

    max_k = (p + 1) // 2
    x_vals = list(range(max_k))
    y_vals = diag_power[:max_k].tolist()

    fig = go.Figure()

    # Main line
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name="Diagonal power",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6),
            hovertemplate="k=%{x}<br>Power: %{y:.4g}<extra></extra>",
        )
    )

    # Highlight key frequencies with larger markers
    key_x = [k for k in key_frequencies if k < max_k]
    key_y = [diag_power[k] for k in key_x]

    if key_x:
        fig.add_trace(
            go.Scatter(
                x=key_x,
                y=key_y,
                mode="markers",
                name="Key frequencies",
                marker=dict(color="#2ca02c", size=12, symbol="star"),
                hovertemplate="Key k=%{x}<br>Power: %{y:.4g}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Power Along (a+b) Diagonal",
        xaxis_title="Frequency k",
        yaxis_title="Power",
        height=350,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
    )

    return fig


def create_key_frequency_table(
    key_frequencies: list[int],
    freq_power: np.ndarray,
    fve_per_freq: np.ndarray,
    p: int,
) -> go.Figure:
    """Create table showing key frequencies with power and FVE.

    Args:
        key_frequencies: List of key frequency indices
        freq_power: [p] power at each frequency
        fve_per_freq: [p] FVE for each frequency
        p: Prime modulus

    Returns:
        Plotly figure with table
    """
    if not key_frequencies:
        return create_empty_figure("No key frequencies detected")

    # Sort by power
    sorted_freqs = sorted(key_frequencies, key=lambda k: freq_power[k], reverse=True)

    # Build table data
    k_vals = sorted_freqs
    omega_vals = [f"2π·{k}/{p}" for k in sorted_freqs]
    power_vals = [f"{freq_power[k]:.2e}" for k in sorted_freqs]
    fve_vals = [f"{fve_per_freq[k]*100:.1f}%" for k in sorted_freqs]
    cumulative_fve = np.cumsum([fve_per_freq[k] for k in sorted_freqs])
    cum_fve_vals = [f"{v*100:.1f}%" for v in cumulative_fve]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["k", "ω_k", "Power", "FVE", "Cumulative FVE"],
                    fill_color="#343a40",
                    font=dict(color="white", size=12),
                    align="center",
                ),
                cells=dict(
                    values=[k_vals, omega_vals, power_vals, fve_vals, cum_fve_vals],
                    fill_color=[["#d4edda"] * len(k_vals)] * 5,
                    align="center",
                    height=25,
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Key Frequencies (p={p})",
        height=min(300, 80 + len(key_frequencies) * 30),
        margin=dict(l=10, r=10, t=40, b=10),
    )

    return fig


def create_ablation_comparison(
    full_loss: float,
    full_accuracy: float,
    restricted_loss: float,
    restricted_accuracy: float,
    excluded_loss: float,
    excluded_accuracy: float,
) -> go.Figure:
    """Create grouped bar comparing full/restricted/excluded loss and accuracy.

    Args:
        full_loss: Loss with all frequencies
        full_accuracy: Accuracy with all frequencies
        restricted_loss: Loss with only key frequencies
        restricted_accuracy: Accuracy with only key frequencies
        excluded_loss: Loss without key frequencies
        excluded_accuracy: Accuracy without key frequencies

    Returns:
        Plotly figure with grouped bars
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cross-Entropy Loss", "Accuracy"),
    )

    categories = ["Full", "Restricted", "Excluded"]
    colors = ["#1f77b4", "#2ca02c", "#d62728"]

    # Loss bars (left subplot)
    losses = [full_loss, restricted_loss, excluded_loss]
    fig.add_trace(
        go.Bar(
            x=categories,
            y=losses,
            marker_color=colors,
            text=[f"{v:.3f}" for v in losses],
            textposition="outside",
            hovertemplate="%{x}<br>Loss: %{y:.4f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Accuracy bars (right subplot)
    accuracies = [full_accuracy, restricted_accuracy, excluded_accuracy]
    fig.add_trace(
        go.Bar(
            x=categories,
            y=[a * 100 for a in accuracies],
            marker_color=colors,
            text=[f"{v*100:.1f}%" for v in accuracies],
            textposition="outside",
            hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Frequency Ablation Results",
        height=350,
    )

    # Log scale for loss
    fig.update_yaxes(type="log", title_text="Loss", row=1, col=1)
    fig.update_yaxes(range=[0, 105], title_text="Accuracy (%)", row=1, col=2)

    return fig


def create_fve_breakdown_plot(
    fve_per_freq: np.ndarray,
    key_frequencies: list[int],
    p: int,
) -> go.Figure:
    """Create bar chart showing FVE per frequency.

    Args:
        fve_per_freq: [p] FVE for each frequency
        key_frequencies: Key frequency indices to highlight
        p: Prime modulus

    Returns:
        Plotly figure with bar chart and total FVE annotation
    """
    if fve_per_freq is None or len(fve_per_freq) == 0:
        return create_empty_figure("No FVE data")

    max_k = (p + 1) // 2
    x_vals = list(range(max_k))
    y_vals = (fve_per_freq[:max_k] * 100).tolist()  # Convert to percentage

    # Color bars based on whether they're key frequencies
    colors = ["#2ca02c" if k in key_frequencies else "#1f77b4" for k in range(max_k)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=x_vals,
            y=y_vals,
            marker_color=colors,
            hovertemplate="k=%{x}<br>FVE: %{y:.2f}%<extra></extra>",
        )
    )

    # Calculate total FVE for key frequencies
    total_fve = sum(fve_per_freq[k] for k in key_frequencies if k < len(fve_per_freq))

    fig.update_layout(
        title=f"Fraction of Variance Explained (Total Key: {total_fve*100:.1f}%)",
        xaxis_title="Frequency k",
        yaxis_title="FVE (%)",
        height=350,
        showlegend=False,
    )

    return fig
