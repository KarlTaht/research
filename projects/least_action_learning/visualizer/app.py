"""Gradio application for grokking experiment visualization with single-page layout."""

from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import (
    ExperimentRun,
    analyze_all_experiments,
    discover_experiments,
    get_experiment_display_name,
    load_experiment,
    load_sweep_groups,
)
from .plots import (
    create_grokking_summary_table,
    create_embedding_spectrum_plot,
    create_logit_spectrum_heatmap,
    create_logit_diagonal_plot,
    create_key_frequency_table,
    create_ablation_comparison,
    create_fve_breakdown_plot,
)
from .helpers import create_empty_figure, filter_by_epoch_range, get_valid_data
from .styles import CATEGORICAL_COLORS

# Import architecture utilities from analysis layer
# Handle both running from project root and from within project directory
try:
    from ..src.analysis.architecture import (
        get_layer_groups,
        get_layer_display_name,
        get_layer_choices,
    )
    from ..src.analysis.phases import analyze_grokking
    from ..src.analysis.frequency import analyze_checkpoint_frequency
    from ..src.analysis.loader import ExperimentLoader
except ImportError:
    from src.analysis.architecture import (
        get_layer_groups,
        get_layer_display_name,
        get_layer_choices,
    )
    from src.analysis.phases import analyze_grokking
    from src.analysis.frequency import analyze_checkpoint_frequency
    from src.analysis.loader import ExperimentLoader


# Cache for loaded experiments
_experiment_cache: dict[str, ExperimentRun] = {}


def get_experiment(exp_path: str) -> Optional[ExperimentRun]:
    """Load experiment with caching."""
    if not exp_path:
        return None

    if exp_path not in _experiment_cache:
        try:
            _experiment_cache[exp_path] = load_experiment(Path(exp_path))
        except Exception as e:
            print(f"Failed to load experiment: {e}")
            return None

    return _experiment_cache[exp_path]


def get_experiment_choices() -> list[tuple[str, str]]:
    """Get experiment choices as (display_name, path) pairs with ID prefixes.

    IDs are assigned based on the same sorted order as the summary table
    (sorted by test percentage, then train percentage, then variance).
    """
    # Load all experiments to get the sorted analysis order
    experiments = load_all_experiments()

    if not experiments:
        return []

    # Analyze to get sorted order (same as summary table)
    analysis_df = analyze_all_experiments(experiments)

    if analysis_df.empty:
        # Fallback to unsorted if analysis fails
        return [(get_experiment_display_name(Path(path)), path) for path in experiments.keys()]

    # Build choices with IDs based on sorted order
    choices = []
    for idx, row in analysis_df.iterrows():
        exp_name = row["name"]
        # Find the path for this experiment
        for path, exp in experiments.items():
            if exp.name == exp_name:
                exp_id = idx + 1  # 1-based ID
                display = f"{exp_id}. {get_experiment_display_name(Path(path))}"
                choices.append((display, path))
                break

    return choices


def refresh_experiments() -> dict:
    """Refresh the experiment dropdown choices."""
    _experiment_cache.clear()
    choices = get_experiment_choices()
    return gr.update(choices=choices, value=None)


def load_all_experiments() -> dict[str, ExperimentRun]:
    """Load all experiments into cache and return the cache."""
    experiments = discover_experiments()
    for exp_path in experiments:
        path_str = str(exp_path)
        if path_str not in _experiment_cache:
            try:
                _experiment_cache[path_str] = load_experiment(exp_path)
            except Exception as e:
                print(f"Failed to load experiment {exp_path}: {e}")
    return _experiment_cache


# ─── Helper Functions ─────────────────────────────────────────────────────────


def create_single_metric_plot(
    df: pd.DataFrame,
    metric: str,
    title: str,
    log_scale: bool = False,
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create a single-metric evolution plot."""
    if metric not in df.columns:
        return create_empty_figure(f"No {metric} data")

    df = filter_by_epoch_range(df, min_epoch, max_epoch)

    y_data = df[metric]
    valid_mask = y_data.notna() & np.isfinite(y_data)

    if not valid_mask.any():
        return create_empty_figure(f"No valid {metric} data")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.loc[valid_mask, "step"],
            y=df.loc[valid_mask, metric],
            mode="lines",
            line=dict(color="#1f77b4", width=2),
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.2)",
            hovertemplate=f"Step: %{{x}}<br>{title}: %{{y:.4g}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title=title,
        hovermode="x unified",
        height=300,
    )

    if log_scale:
        fig.update_yaxes(type="log")

    return fig


def create_loss_curves_plot(
    df: pd.DataFrame,
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create train/test loss curves plot.

    Train loss is shown with a dotted line, test loss with a solid line.

    Args:
        df: History DataFrame with train_loss and test_loss columns
        min_epoch: Start of epoch range
        max_epoch: End of epoch range

    Returns:
        Plotly figure with loss curves
    """
    df = filter_by_epoch_range(df, min_epoch, max_epoch)

    if df.empty:
        return create_empty_figure("No loss data")

    fig = go.Figure()
    has_data = False

    # Test loss (solid line) - shown first so it's behind train
    if "test_loss" in df.columns:
        y_data = df["test_loss"]
        valid_mask = y_data.notna() & np.isfinite(y_data)
        if valid_mask.any():
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, "test_loss"],
                    mode="lines",
                    name="Test Loss",
                    line=dict(color="#d62728", width=2),  # Red solid
                    hovertemplate="Step: %{x}<br>Test Loss: %{y:.4g}<extra></extra>",
                )
            )

    # Train loss (dotted line)
    if "train_loss" in df.columns:
        y_data = df["train_loss"]
        valid_mask = y_data.notna() & np.isfinite(y_data)
        if valid_mask.any():
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, "train_loss"],
                    mode="lines",
                    name="Train Loss",
                    line=dict(color="#1f77b4", width=2, dash="dot"),  # Blue dotted
                    hovertemplate="Step: %{x}<br>Train Loss: %{y:.4g}<extra></extra>",
                )
            )

    if not has_data:
        return create_empty_figure("No loss data")

    fig.update_layout(
        title="Loss Curves",
        xaxis_title="Step",
        yaxis_title="Loss",
        yaxis_type="log",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )

    return fig


def create_accuracy_curves_plot(
    df: pd.DataFrame,
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create train/test accuracy curves plot.

    Train accuracy is shown with a dotted line, test accuracy with a solid line.

    Args:
        df: History DataFrame with train_acc and test_acc columns
        min_epoch: Start of epoch range
        max_epoch: End of epoch range

    Returns:
        Plotly figure with accuracy curves
    """
    df = filter_by_epoch_range(df, min_epoch, max_epoch)

    if df.empty:
        return create_empty_figure("No accuracy data")

    fig = go.Figure()
    has_data = False

    # Test accuracy (solid line) - shown first so it's behind train
    if "test_acc" in df.columns:
        y_data = df["test_acc"]
        valid_mask = y_data.notna() & np.isfinite(y_data)
        if valid_mask.any():
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, "test_acc"],
                    mode="lines",
                    name="Test Acc",
                    line=dict(color="#d62728", width=2),  # Red solid
                    hovertemplate="Step: %{x}<br>Test Acc: %{y:.2%}<extra></extra>",
                )
            )

    # Train accuracy (dotted line)
    if "train_acc" in df.columns:
        y_data = df["train_acc"]
        valid_mask = y_data.notna() & np.isfinite(y_data)
        if valid_mask.any():
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, "train_acc"],
                    mode="lines",
                    name="Train Acc",
                    line=dict(color="#1f77b4", width=2, dash="dot"),  # Blue dotted
                    hovertemplate="Step: %{x}<br>Train Acc: %{y:.2%}<extra></extra>",
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
        title="Accuracy Curves",
        xaxis_title="Step",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )

    return fig


def create_learning_rate_plot(
    exp: "ExperimentRun",
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create learning rate schedule plot.

    Computes LR from config (base LR + warmup schedule).

    Args:
        exp: Experiment run with config
        min_epoch: Start of epoch range
        max_epoch: End of epoch range

    Returns:
        Plotly figure with learning rate curve
    """
    config = exp.config
    df = exp.history_df

    if df.empty or "step" not in df.columns:
        return create_empty_figure("No training data")

    base_lr = config.get("lr", 1e-3)
    warmup_epochs = config.get("warmup_epochs", 0)

    # Filter by epoch range
    df = filter_by_epoch_range(df, min_epoch, max_epoch)

    if df.empty:
        return create_empty_figure("No data in epoch range")

    # Compute LR at each step
    steps = df["step"].values
    lrs = []
    for step in steps:
        if warmup_epochs > 0 and step < warmup_epochs:
            # Linear warmup from 0 to base_lr
            lr = base_lr * (step + 1) / warmup_epochs
        else:
            lr = base_lr
        lrs.append(lr)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=steps,
            y=lrs,
            mode="lines",
            name="Learning Rate",
            line=dict(color="#2ca02c", width=2),  # Green
            fill="tozeroy",
            fillcolor="rgba(44, 160, 44, 0.2)",
            hovertemplate="Step: %{x}<br>LR: %{y:.2e}<extra></extra>",
        )
    )

    # Add warmup annotation if applicable
    if warmup_epochs > 0 and warmup_epochs >= min_epoch:
        fig.add_vline(
            x=warmup_epochs,
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Warmup ends ({warmup_epochs})",
            annotation_position="top right",
        )

    fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        hovermode="x unified",
        height=300,
    )

    # Use scientific notation for y-axis
    fig.update_yaxes(tickformat=".1e")

    return fig


def create_weight_group_plot(
    df: pd.DataFrame,
    layer_indices: list[int],
    layer_names: list[str],
    title: str,
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create plot of weight norms for a specific group of layers.

    Args:
        df: History DataFrame with layer_{i}_weight_norm columns
        layer_indices: List of layer indices to include (e.g., [2, 3, 4, 5])
        layer_names: Display names for each layer (e.g., ["Q", "K", "V", "Wo"])
        title: Plot title
        min_epoch: Start of epoch range
        max_epoch: End of epoch range

    Returns:
        Plotly figure with overlaid traces for each layer
    """
    df = filter_by_epoch_range(df, min_epoch, max_epoch)

    fig = go.Figure()
    has_data = False

    for i, (layer_idx, name) in enumerate(zip(layer_indices, layer_names)):
        col = f"layer_{layer_idx}_weight_norm"
        y_data, valid_mask = get_valid_data(df, col)

        if valid_mask.any():
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, col],
                    mode="lines",
                    name=name,
                    line=dict(color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)], width=2),
                    hovertemplate=f"{name}<br>Step: %{{x}}<br>Norm: %{{y:.4g}}<extra></extra>",
                )
            )

    if not has_data:
        return create_empty_figure(f"No data for {title}")

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Weight Norm",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=280,
    )

    return fig


def create_per_layer_metric_plot(
    exp: ExperimentRun,
    metric_suffix: str,
    title: str,
    selected_layer: str = "All",
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create plot of per-layer metrics over training."""
    df = exp.history_df.copy()

    df = filter_by_epoch_range(df, min_epoch, max_epoch)

    layer_cols = [
        c for c in df.columns if c.startswith("layer_") and c.endswith(f"_{metric_suffix}")
    ]

    if not layer_cols:
        return create_empty_figure(f"No per-layer {metric_suffix} data")

    fig = go.Figure()

    # Parse selected layer (format: "idx:name" or "All")
    selected_idx = None
    if selected_layer != "All" and ":" in selected_layer:
        selected_idx = selected_layer.split(":")[0]
    elif selected_layer != "All":
        selected_idx = selected_layer

    for i, col in enumerate(sorted(layer_cols)):
        layer_num = col.split("_")[1]

        # Skip if specific layer selected and this isn't it
        if selected_idx is not None and layer_num != selected_idx:
            continue

        # Get descriptive name for legend
        display_name = get_layer_display_name(layer_num, exp)

        y_data, valid_mask = get_valid_data(df, col)

        if valid_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, col],
                    mode="lines",
                    name=display_name,
                    line=dict(color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)], width=2),
                )
            )

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=350,
    )

    return fig


# ─── Multi-Experiment Plot Functions ──────────────────────────────────────────


def create_multi_loss_curves_plot(
    experiments: list[ExperimentRun],
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create loss curves for multiple experiments with unique colors."""
    if not experiments:
        return create_empty_figure("Select experiments")

    fig = go.Figure()
    has_data = False

    for i, exp in enumerate(experiments):
        df = filter_by_epoch_range(exp.history_df, min_epoch, max_epoch)
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        name = exp.name

        # Test loss (solid)
        if "test_loss" in df.columns:
            y_data, valid_mask = get_valid_data(df, "test_loss")
            if valid_mask.any():
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[valid_mask, "step"],
                        y=df.loc[valid_mask, "test_loss"],
                        mode="lines",
                        name=f"{name} (test)",
                        line=dict(color=color, width=2),
                        legendgroup=name,
                    )
                )

        # Train loss (dotted)
        if "train_loss" in df.columns:
            y_data, valid_mask = get_valid_data(df, "train_loss")
            if valid_mask.any():
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[valid_mask, "step"],
                        y=df.loc[valid_mask, "train_loss"],
                        mode="lines",
                        name=f"{name} (train)",
                        line=dict(color=color, width=2, dash="dot"),
                        legendgroup=name,
                        showlegend=False,
                    )
                )

    if not has_data:
        return create_empty_figure("No loss data")

    fig.update_layout(
        title="Loss Curves",
        xaxis_title="Step",
        yaxis_title="Loss",
        yaxis_type="log",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    return fig


def create_multi_accuracy_curves_plot(
    experiments: list[ExperimentRun],
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create accuracy curves for multiple experiments with unique colors."""
    if not experiments:
        return create_empty_figure("Select experiments")

    fig = go.Figure()
    has_data = False

    for i, exp in enumerate(experiments):
        df = filter_by_epoch_range(exp.history_df, min_epoch, max_epoch)
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        name = exp.name

        # Test accuracy (solid)
        if "test_acc" in df.columns:
            y_data, valid_mask = get_valid_data(df, "test_acc")
            if valid_mask.any():
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[valid_mask, "step"],
                        y=df.loc[valid_mask, "test_acc"],
                        mode="lines",
                        name=f"{name} (test)",
                        line=dict(color=color, width=2),
                        legendgroup=name,
                    )
                )

        # Train accuracy (dotted)
        if "train_acc" in df.columns:
            y_data, valid_mask = get_valid_data(df, "train_acc")
            if valid_mask.any():
                has_data = True
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[valid_mask, "step"],
                        y=df.loc[valid_mask, "train_acc"],
                        mode="lines",
                        name=f"{name} (train)",
                        line=dict(color=color, width=2, dash="dot"),
                        legendgroup=name,
                        showlegend=False,
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
        title="Accuracy Curves",
        xaxis_title="Step",
        yaxis_title="Accuracy",
        yaxis_range=[0, 1.05],
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    return fig


def create_multi_learning_rate_plot(
    experiments: list[ExperimentRun],
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create learning rate schedule plot for multiple experiments."""
    if not experiments:
        return create_empty_figure("Select experiments")

    fig = go.Figure()
    has_data = False

    for i, exp in enumerate(experiments):
        config = exp.config
        df = filter_by_epoch_range(exp.history_df, min_epoch, max_epoch)
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        name = exp.name

        if df.empty or "step" not in df.columns:
            continue

        base_lr = config.get("lr", 1e-3)
        warmup_epochs = config.get("warmup_epochs", 0)

        # Compute LR at each step
        steps = df["step"].values
        lrs = []
        for step in steps:
            if warmup_epochs > 0 and step < warmup_epochs:
                lr = base_lr * (step + 1) / warmup_epochs
            else:
                lr = base_lr
            lrs.append(lr)

        has_data = True
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=lrs,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
            )
        )

    if not has_data:
        return create_empty_figure("No training data")

    fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    fig.update_yaxes(tickformat=".1e")
    return fig


def create_multi_single_metric_plot(
    experiments: list[ExperimentRun],
    metric: str,
    title: str,
    log_scale: bool = False,
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create a single-metric plot for multiple experiments."""
    if not experiments:
        return create_empty_figure("Select experiments")

    fig = go.Figure()
    has_data = False

    for i, exp in enumerate(experiments):
        df = filter_by_epoch_range(exp.history_df, min_epoch, max_epoch)
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        name = exp.name

        if metric not in df.columns:
            continue

        y_data, valid_mask = get_valid_data(df, metric)
        if valid_mask.any():
            has_data = True
            fig.add_trace(
                go.Scatter(
                    x=df.loc[valid_mask, "step"],
                    y=df.loc[valid_mask, metric],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                )
            )

    if not has_data:
        return create_empty_figure(f"No {metric} data")

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )

    if log_scale:
        fig.update_yaxes(type="log")

    return fig


def create_multi_weight_group_plot(
    experiments: list[ExperimentRun],
    layer_indices: list[int],
    layer_names: list[str],
    title: str,
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create weight norm plot for multiple experiments (single layer group)."""
    if not experiments:
        return create_empty_figure("Select experiments")

    fig = go.Figure()
    has_data = False

    # For multi-experiment, we simplify to total norm per experiment for this group
    for exp_idx, exp in enumerate(experiments):
        df = filter_by_epoch_range(exp.history_df, min_epoch, max_epoch)
        color = CATEGORICAL_COLORS[exp_idx % len(CATEGORICAL_COLORS)]
        name = exp.name

        # Sum weight norms for this group
        group_cols = [f"layer_{idx}_weight_norm" for idx in layer_indices]
        existing_cols = [c for c in group_cols if c in df.columns]

        if not existing_cols:
            continue

        # Plot each layer within the group with different dash patterns
        for layer_idx, layer_name in zip(layer_indices, layer_names):
            col = f"layer_{layer_idx}_weight_norm"
            if col not in df.columns:
                continue

            y_data, valid_mask = get_valid_data(df, col)
            if valid_mask.any():
                has_data = True
                # Use different dash patterns for layers within same experiment
                dash_patterns = ["solid", "dash", "dot", "dashdot"]
                dash_idx = layer_indices.index(layer_idx) % len(dash_patterns)

                fig.add_trace(
                    go.Scatter(
                        x=df.loc[valid_mask, "step"],
                        y=df.loc[valid_mask, col],
                        mode="lines",
                        name=f"{name}: {layer_name}",
                        line=dict(
                            color=color,
                            width=2,
                            dash=dash_patterns[dash_idx] if len(experiments) > 1 else "solid",
                        ),
                        legendgroup=name,
                    )
                )

    if not has_data:
        return create_empty_figure(f"No data for {title}")

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Weight Norm",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=280,
    )
    return fig


def create_multi_effective_lr_plot(
    experiments: list[ExperimentRun],
    min_epoch: int = 0,
    max_epoch: int = 100000,
) -> go.Figure:
    """Create effective LR plot for multiple experiments."""
    if not experiments:
        return create_empty_figure("Select experiments")

    fig = go.Figure()
    has_data = False

    for i, exp in enumerate(experiments):
        df = filter_by_epoch_range(exp.history_df, min_epoch, max_epoch)
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        name = exp.name

        for metric, line_style in [("effective_lr_mean", "solid"), ("effective_lr_max", "dash")]:
            y_data, valid_mask = get_valid_data(df, metric)
            if valid_mask.any():
                has_data = True
                metric_label = "mean" if "mean" in metric else "max"
                fig.add_trace(
                    go.Scatter(
                        x=df.loc[valid_mask, "step"],
                        y=df.loc[valid_mask, metric],
                        mode="lines",
                        name=f"{name} ({metric_label})",
                        line=dict(color=color, width=2, dash=line_style),
                        legendgroup=name,
                        showlegend=(metric == "effective_lr_mean"),
                    )
                )

    if not has_data:
        return create_empty_figure("No effective LR data")

    fig.update_layout(
        title="Effective LR (√v_t)",
        xaxis_title="Step",
        yaxis_title="√v_t",
        yaxis_type="log",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
    )
    return fig


# ─── Update Functions ─────────────────────────────────────────────────────────


def update_grokking_summary(group_filter: str = "All") -> go.Figure:
    """Update grokking quality summary table.

    Args:
        group_filter: Group to filter by ("All" shows all experiments)

    Returns:
        Plotly figure with summary table
    """
    experiments = load_all_experiments()

    if not experiments:
        return create_empty_figure("No experiments found")

    # Filter by group if specified
    if group_filter != "All":
        filtered = {}
        for path, exp in experiments.items():
            if group_filter == exp.group or (
                group_filter == "no group" and exp.group == "no group"
            ):
                filtered[path] = exp
        experiments = filtered

    if not experiments:
        return create_empty_figure(f"No experiments in group '{group_filter}'")

    analysis_df = analyze_all_experiments(experiments)
    table_fig = create_grokking_summary_table(analysis_df)

    return table_fig


def update_experiment_controls(exp_paths: list[str] | str | None) -> Tuple:
    """Update control widgets when experiment changes.

    Uses intelligent max epoch default: grok_step + max(10% of total, 500 epochs).
    Falls back to full range if experiment hasn't grokked.
    Based on first selected experiment for multi-experiment mode.
    """
    # Normalize to list
    if not exp_paths:
        return (
            gr.update(maximum=10000, value=0),
            gr.update(maximum=10000, value=10000),
            gr.update(choices=["All"], value="All"),
        )

    if isinstance(exp_paths, str):
        exp_paths = [exp_paths]

    # Load first experiment for controls (use first for defaults)
    first_exp = None
    all_max_steps = []
    for path in exp_paths:
        exp = get_experiment(path)
        if exp:
            if first_exp is None:
                first_exp = exp
            max_step = int(exp.history_df["step"].max()) if not exp.history_df.empty else 10000
            all_max_steps.append(max_step)

    if first_exp is None:
        return (
            gr.update(maximum=10000, value=0),
            gr.update(maximum=10000, value=10000),
            gr.update(choices=["All"], value="All"),
        )

    # Use max across all experiments for slider range
    max_step = max(all_max_steps) if all_max_steps else 10000

    # Compute intelligent default max_epoch based on first experiment's grok_step
    default_max_epoch = max_step  # fallback to full range
    try:
        analysis = analyze_grokking(first_exp)
        if analysis.grok_step is not None:
            # Default to grok_step + max(10% of total, 500 epochs)
            buffer = max(int(0.1 * max_step), 500)
            default_max_epoch = min(analysis.grok_step + buffer, max_step)
    except Exception:
        pass  # Fall back to full range if analysis fails

    # Get layer choices with descriptive names (e.g., "0:tok_embed", "2:b0_attn_Q")
    layer_choices = get_layer_choices(first_exp, metric_suffix="weight_norm")

    return (
        gr.update(maximum=max_step, value=0),
        gr.update(maximum=max_step, value=default_max_epoch),
        gr.update(choices=layer_choices, value="All"),
    )


def update_training_dynamics_plots(
    exp_paths: list[str] | str | None,
    min_epoch: int,
    max_epoch: int,
) -> Tuple:
    """Update training dynamics plots (loss, accuracy, learning rate).

    Returns 3 plots:
    - Loss curves (train dotted, test solid)
    - Accuracy curves (train dotted, test solid)
    - Learning rate schedule

    Supports multiple experiments for overlay comparison.
    """
    # Normalize to list
    if not exp_paths:
        empty = create_empty_figure("Select experiment(s)")
        return empty, empty, empty

    if isinstance(exp_paths, str):
        exp_paths = [exp_paths]

    # Load all experiments
    experiments = []
    for path in exp_paths:
        exp = get_experiment(path)
        if exp:
            experiments.append(exp)

    if not experiments:
        empty = create_empty_figure("Failed to load experiment(s)")
        return empty, empty, empty

    # Use multi-experiment functions for overlay
    loss_fig = create_multi_loss_curves_plot(experiments, min_epoch=min_epoch, max_epoch=max_epoch)
    acc_fig = create_multi_accuracy_curves_plot(
        experiments, min_epoch=min_epoch, max_epoch=max_epoch
    )
    lr_fig = create_multi_learning_rate_plot(experiments, min_epoch=min_epoch, max_epoch=max_epoch)

    return loss_fig, acc_fig, lr_fig


def update_weight_plots(
    exp_paths: list[str] | str | None,
    min_epoch: int,
    max_epoch: int,
    selected_layer: str,
    show_embeddings: bool = False,
) -> Tuple:
    """Update weight analysis plots with expanded layout.

    Returns 6 plots:
    - Embeddings (tok_embed, pos_embed) - hidden by default unless show_embeddings=True
    - Block 0 Attention (Q, K, V, Wo)
    - Block 0 FFN (up, down)
    - Block 1 Attention (Q, K, V, Wo)
    - Block 1 FFN (up, down)
    - Output head

    Note: UI currently supports 2 blocks. To support more blocks,
    update the UI layout in create_app() as well.

    Supports multiple experiments for overlay comparison.
    """
    # UI currently hardcoded for 2 blocks (4 block plots + embed + output = 6)
    n_ui_blocks = 2
    n_plots = 2 + n_ui_blocks * 2  # embed + blocks*2 + output

    # Normalize to list
    if not exp_paths:
        empty = create_empty_figure("Select experiment(s)")
        return (empty,) * n_plots

    if isinstance(exp_paths, str):
        exp_paths = [exp_paths]

    # Load all experiments
    experiments = []
    for path in exp_paths:
        exp = get_experiment(path)
        if exp:
            experiments.append(exp)

    if not experiments:
        empty = create_empty_figure("Failed to load experiment(s)")
        return (empty,) * n_plots

    # Use first experiment for structure (assumes consistent architecture)
    first_exp = experiments[0]
    n_layers = first_exp.config.get("n_layers", first_exp.n_layers)
    model_type = first_exp.config.get("model_type", "transformer")
    groups = get_layer_groups(model_type, n_layers)

    # Embeddings plot - only create when visible (section is hidden entirely when unchecked)
    if show_embeddings:
        emb_indices, emb_names = groups["embeddings"]
        embed_fig = create_multi_weight_group_plot(
            experiments,
            emb_indices,
            emb_names,
            "Embeddings",
            min_epoch=min_epoch,
            max_epoch=max_epoch,
        )
    else:
        # Section is hidden, no need to render anything
        embed_fig = None

    # Block plots via loop (attention then FFN for each block)
    block_figs = []
    component_names = {"attn": "Attention", "ffn": "FFN"}
    for block_idx in range(n_ui_blocks):
        for component in ["attn", "ffn"]:
            key = f"block{block_idx}_{component}"
            if key in groups:
                indices, names = groups[key]
                fig = create_multi_weight_group_plot(
                    experiments,
                    indices,
                    names,
                    f"Block {block_idx}: {component_names[component]}",
                    min_epoch=min_epoch,
                    max_epoch=max_epoch,
                )
            else:
                fig = create_empty_figure(f"No Block {block_idx} {component} data")
            block_figs.append(fig)

    # Output head plot (always last)
    out_indices, out_names = groups["output"]
    output_fig = create_multi_weight_group_plot(
        experiments, out_indices, out_names, "Output Head", min_epoch=min_epoch, max_epoch=max_epoch
    )

    return (embed_fig, *block_figs, output_fig)


def update_curvature_plots(
    exp_paths: list[str] | str | None,
    min_epoch: int,
    max_epoch: int,
) -> Tuple:
    """Update curvature analysis plots (both input-sensitivity and weight-curvature).

    Supports multiple experiments for overlay comparison.
    """
    # Normalize to list
    if not exp_paths:
        empty = create_empty_figure("Select experiment(s)")
        return empty, empty, empty, empty, empty

    if isinstance(exp_paths, str):
        exp_paths = [exp_paths]

    # Load all experiments
    experiments = []
    for path in exp_paths:
        exp = get_experiment(path)
        if exp:
            experiments.append(exp)

    if not experiments:
        empty = create_empty_figure("Failed to load experiment(s)")
        return empty, empty, empty, empty, empty

    # Input-sensitivity metrics (∇_x)
    jacobian_fig = create_multi_single_metric_plot(
        experiments,
        "jacobian_norm",
        "Jacobian Norm (∇ₓf)",
        log_scale=True,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    input_hessian_fig = create_multi_single_metric_plot(
        experiments,
        "hessian_trace",
        "Input Hessian Trace (∇²ₓf)",
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    # Weight-curvature metrics (∇_w)
    gradient_norm_fig = create_multi_single_metric_plot(
        experiments,
        "gradient_norm",
        "Gradient Norm (∇ᵥL)",
        log_scale=True,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    weight_hessian_fig = create_multi_single_metric_plot(
        experiments,
        "weight_hessian_trace",
        "Weight Hessian (∇²ᵥL)",
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    fisher_fig = create_multi_single_metric_plot(
        experiments,
        "fisher_trace",
        "Fisher Trace (∇L·∇Lᵀ)",
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    return jacobian_fig, input_hessian_fig, gradient_norm_fig, weight_hessian_fig, fisher_fig


# Keep old function for backward compatibility with tests
def update_gradient_plots(
    exp_path: str,
    min_epoch: int,
    max_epoch: int,
) -> Tuple:
    """Update gradient analysis plots (legacy - returns 2 plots)."""
    results = update_curvature_plots(exp_path, min_epoch, max_epoch)
    return results[0], results[1]  # Just jacobian and input hessian


def update_adam_plots(
    exp_paths: list[str] | str | None,
    min_epoch: int,
    max_epoch: int,
) -> Tuple:
    """Update Adam optimizer dynamics plots.

    Returns 3 plots:
    - Effective LR (mean and max sqrt(v_t))
    - Adam ratio (signal-to-noise: |m|/(sqrt(v)+eps))
    - Update/Decay ratio (learning vs forgetting)

    Supports multiple experiments for overlay comparison.
    """
    # Normalize to list
    if not exp_paths:
        empty = create_empty_figure("Select experiment(s)")
        return empty, empty, empty

    if isinstance(exp_paths, str):
        exp_paths = [exp_paths]

    # Load all experiments
    experiments = []
    for path in exp_paths:
        exp = get_experiment(path)
        if exp:
            experiments.append(exp)

    if not experiments:
        empty = create_empty_figure("Failed to load experiment(s)")
        return empty, empty, empty

    # Effective LR plot
    effective_lr_fig = create_multi_effective_lr_plot(experiments, min_epoch, max_epoch)

    # Adam ratio plot
    adam_ratio_fig = create_multi_single_metric_plot(
        experiments,
        "adam_ratio_mean",
        "Adam Ratio (|m|/(√v+ε))",
        log_scale=True,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    # Update/Decay ratio plot
    update_decay_fig = create_multi_single_metric_plot(
        experiments,
        "update_decay_ratio",
        "Update/Decay Ratio",
        log_scale=True,
        min_epoch=min_epoch,
        max_epoch=max_epoch,
    )

    return effective_lr_fig, adam_ratio_fig, update_decay_fig


def update_all_plots(
    exp_paths: list[str] | str | None,
    min_epoch: int,
    max_epoch: int,
    selected_layer: str,
    show_embeddings: bool = False,
) -> Tuple:
    """Update all analysis plots when experiment or controls change.

    Returns 17 plots total:
    - 3 training dynamics plots (loss_curves, accuracy_curves, learning_rate)
    - 6 weight plots (embed, b0_attn, b0_ffn, b1_attn, b1_ffn, output)
    - 5 curvature plots (jacobian, input_hessian, gradient, weight_hessian, fisher)
    - 3 Adam optimizer plots (effective_lr, adam_ratio, update_decay)

    Supports multiple experiments for overlay comparison.
    """
    dynamics_plots = update_training_dynamics_plots(exp_paths, min_epoch, max_epoch)
    weight_plots = update_weight_plots(
        exp_paths, min_epoch, max_epoch, selected_layer, show_embeddings
    )
    curvature_plots = update_curvature_plots(exp_paths, min_epoch, max_epoch)
    adam_plots = update_adam_plots(exp_paths, min_epoch, max_epoch)

    return dynamics_plots + weight_plots + curvature_plots + adam_plots


def get_available_groups() -> list[str]:
    """Get list of available experiment groups."""
    groups = load_sweep_groups()
    unique_groups = sorted(set(groups.values()))
    return ["All"] + unique_groups + ["no group"]


# ─── Main App ─────────────────────────────────────────────────────────────────


def create_app() -> gr.Blocks:
    """Create the Gradio application interface with single-page layout."""

    with gr.Blocks(title="Grokking Experiment Visualizer") as app:
        gr.Markdown("# Grokking Experiment Visualizer")
        gr.Markdown(
            "Analyze training dynamics, weight norms, and gradient metrics "
            "from grokking experiments."
        )

        # ─── Section 1: Grokking Quality Summary ───
        gr.Markdown("## Grokking Quality Summary")
        gr.Markdown(
            "*Overview of all experiments ranked by grokking quality. "
            "Green = high test accuracy, Red = low.*"
        )
        summary_table = gr.Plot(label="Experiment Summary")

        gr.Markdown("---")

        # ─── Section 2: Visual Controls ───
        gr.Markdown("## Visual Controls")
        with gr.Row():
            with gr.Column(scale=2):
                group_dropdown = gr.Dropdown(
                    choices=get_available_groups(),
                    value="All",
                    label="Group",
                    info="Filter by sweep group",
                )
                exp_dropdown = gr.Dropdown(
                    choices=get_experiment_choices(),
                    label="Experiments",
                    info="Select up to 3 experiments to compare",
                    multiselect=True,
                    max_choices=3,
                )
                refresh_btn = gr.Button("Refresh Experiments", variant="secondary", size="sm")

            with gr.Column(scale=2):
                min_epoch_slider = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    step=100,
                    value=0,
                    label="Min Epoch",
                    info="Start of epoch range",
                )
                max_epoch_slider = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    step=100,
                    value=10000,
                    label="Max Epoch",
                    info="End of epoch range (auto-set to grok step + buffer)",
                )

            with gr.Column(scale=1):
                layer_dropdown = gr.Dropdown(
                    choices=["All"],
                    value="All",
                    label="Layer",
                    info="Filter per-layer plots",
                )

        gr.Markdown("---")

        # ─── Section 3: Training Dynamics ───
        gr.Markdown("## Training Dynamics")
        gr.Markdown("*Loss and accuracy curves over training. " "Train (dotted), test (solid).*")
        with gr.Row():
            loss_curves_plot = gr.Plot(label="Loss Curves")
            accuracy_curves_plot = gr.Plot(label="Accuracy Curves")
            learning_rate_plot = gr.Plot(label="Learning Rate Schedule")

        gr.Markdown("---")

        # ─── Tabbed Analysis Sections ───
        with gr.Tabs():
            # Tab 1: Weight Analysis
            with gr.Tab("Weight Analysis"):
                gr.Markdown(
                    "*Track how model weights evolve during training. "
                    "Weight norm growth often precedes grokking.*"
                )

                # Checkbox to show/hide embeddings section
                show_embeddings = gr.Checkbox(
                    label="Show Embeddings",
                    value=False,
                    info="Include embedding layers (tok_embed, pos_embed) in the analysis",
                )

                # Row 1: Embeddings (hidden by default, wrapped in Column for visibility toggle)
                embed_section = gr.Column(visible=False)
                with embed_section:
                    gr.Markdown("### Embeddings")
                    embed_plot = gr.Plot(label="Embeddings (tok_embed, pos_embed)")

                # Row 2: Block 0
                gr.Markdown("### Block 0")
                with gr.Row():
                    b0_attn_plot = gr.Plot(label="Block 0: Attention (Q, K, V, Wo)")
                    b0_ffn_plot = gr.Plot(label="Block 0: FFN (up, down)")

                # Row 3: Block 1
                gr.Markdown("### Block 1")
                with gr.Row():
                    b1_attn_plot = gr.Plot(label="Block 1: Attention (Q, K, V, Wo)")
                    b1_ffn_plot = gr.Plot(label="Block 1: FFN (up, down)")

                # Row 4: Output
                gr.Markdown("### Output")
                output_plot = gr.Plot(label="Output Head")

            # Tab 2: Curvature Analysis
            with gr.Tab("Curvature Analysis"):
                gr.Markdown(
                    "*Input sensitivity (∇ₓ) measures output change w.r.t. inputs. "
                    "Weight curvature (∇ᵥ) measures loss landscape for generalization analysis.*"
                )

                # Row 1: Input-sensitivity metrics
                gr.Markdown("### Input Sensitivity")
                with gr.Row():
                    jacobian_plot = gr.Plot(label="Jacobian Norm (∇ₓf)")
                    input_hessian_plot = gr.Plot(label="Input Hessian Trace (∇²ₓf)")

                # Row 2: Weight-curvature metrics
                gr.Markdown("### Weight Curvature (Loss Landscape)")
                with gr.Row():
                    gradient_norm_plot = gr.Plot(label="Gradient Norm (∇ᵥL)")
                    weight_hessian_plot = gr.Plot(label="Weight Hessian (∇²ᵥL)")
                    fisher_plot = gr.Plot(label="Fisher Trace (∇L·∇Lᵀ)")

            # Tab 3: Adam Optimizer Dynamics
            with gr.Tab("Adam Optimizer"):
                gr.Markdown(
                    "*Internal state of Adam optimizer. "
                    "Effective LR shows adaptive learning rate scaling (√v_t). "
                    "Adam ratio shows signal-to-noise (|m|/√v). "
                    "Update/Decay ratio shows learning vs forgetting balance.*"
                )
                with gr.Row():
                    effective_lr_plot = gr.Plot(label="Effective LR (√v)")
                    adam_ratio_plot = gr.Plot(label="Adam Ratio (|m|/√v)")
                    update_decay_plot = gr.Plot(label="Update/Decay Ratio")

            # Tab 4: Fourier Analysis
            with gr.Tab("Fourier Analysis"):
                gr.Markdown(
                    "*Mechanistic interpretability via Fourier analysis. "
                    "Based on Nanda et al. 'Progress measures for grokking' (ICLR 2023). "
                    "Shows how the model learns key frequencies for modular arithmetic.*"
                )

                # Controls row
                with gr.Row():
                    fourier_checkpoint_dropdown = gr.Dropdown(
                        label="Checkpoint",
                        choices=["best"],
                        value="best",
                        info="Select checkpoint to analyze",
                    )
                    fourier_threshold_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.50,
                        step=0.05,
                        value=0.15,
                        label="Key Frequency Threshold",
                        info="Fraction of max power to consider key",
                    )
                    fourier_analyze_btn = gr.Button("Analyze", variant="primary")

                # Status display
                fourier_status = gr.Textbox(
                    label="Analysis Status",
                    interactive=False,
                    value="Select an experiment and click Analyze",
                )

                # Embedding spectrum
                gr.Markdown("### Embedding Fourier Spectrum")
                with gr.Row():
                    fourier_embedding_plot = gr.Plot(label="Embedding Power by Frequency")

                # Logit spectrum
                gr.Markdown("### Logit Fourier Spectrum")
                with gr.Row():
                    fourier_logit_2d_plot = gr.Plot(label="2D Power Spectrum")
                    fourier_logit_diagonal_plot = gr.Plot(label="Power Along (a+b) Diagonal")

                # Key frequencies & ablation
                gr.Markdown("### Key Frequency Analysis")
                with gr.Row():
                    fourier_key_freq_table = gr.Plot(label="Key Frequencies")
                    fourier_ablation_plot = gr.Plot(label="Ablation Results")

                # FVE summary
                gr.Markdown("### Variance Explained")
                with gr.Row():
                    fourier_fve_plot = gr.Plot(label="Fraction of Variance Explained")

        # ─── Event Handlers ───

        # Initialize summary table on load
        app.load(
            fn=update_grokking_summary,
            outputs=[summary_table],
        )

        # Refresh button updates summary and experiment list
        def refresh_all():
            load_sweep_groups(force_reload=True)  # Reload sweep groups
            dropdown_update = refresh_experiments()
            group_update = gr.update(choices=get_available_groups(), value="All")
            table_fig = update_grokking_summary()
            return group_update, dropdown_update, table_fig

        refresh_btn.click(
            fn=refresh_all,
            outputs=[group_dropdown, exp_dropdown, summary_table],
        )

        # Group filter updates experiment dropdown and summary table
        def filter_experiments_by_group(selected_group):
            all_choices = get_experiment_choices()
            if selected_group == "All":
                dropdown_update = gr.update(choices=all_choices, value=None)
            else:
                # Filter by group
                filtered = []
                for display, path in all_choices:
                    exp = get_experiment(path)
                    if exp and (
                        selected_group == exp.group
                        or (selected_group == "no group" and exp.group == "no group")
                    ):
                        filtered.append((display, path))
                dropdown_update = gr.update(
                    choices=filtered if filtered else all_choices, value=None
                )

            # Update summary table with same filter
            table_fig = update_grokking_summary(selected_group)
            return dropdown_update, table_fig

        group_dropdown.change(
            fn=filter_experiments_by_group,
            inputs=[group_dropdown],
            outputs=[exp_dropdown, summary_table],
        )

        # Experiment selection updates controls and all plots
        def on_experiment_change(exp_paths, show_emb):
            controls = update_experiment_controls(exp_paths)
            # Use intelligent default epoch range from controls
            if exp_paths:
                # Extract the computed default max_epoch from controls
                default_max = controls[1].get("value", 10000)
                plots = update_all_plots(exp_paths, 0, default_max, "All", show_emb)
                return controls + plots
            empty = create_empty_figure("Select experiment(s)")
            # 17 empty plots: 3 dynamics + 6 weight + 5 curvature + 3 adam
            return controls + (empty,) * 17

        exp_dropdown.change(
            fn=on_experiment_change,
            inputs=[exp_dropdown, show_embeddings],
            outputs=[
                min_epoch_slider,
                max_epoch_slider,
                layer_dropdown,
                # 3 training dynamics plots
                loss_curves_plot,
                accuracy_curves_plot,
                learning_rate_plot,
                # 6 weight plots
                embed_plot,
                b0_attn_plot,
                b0_ffn_plot,
                b1_attn_plot,
                b1_ffn_plot,
                output_plot,
                # 5 curvature plots
                jacobian_plot,
                input_hessian_plot,
                gradient_norm_plot,
                weight_hessian_plot,
                fisher_plot,
                # 3 Adam optimizer plots
                effective_lr_plot,
                adam_ratio_plot,
                update_decay_plot,
            ],
        )

        # Training dynamics plot outputs
        all_dynamics_plots = [loss_curves_plot, accuracy_curves_plot, learning_rate_plot]

        # All weight plot outputs
        all_weight_plots = [
            embed_plot,
            b0_attn_plot,
            b0_ffn_plot,
            b1_attn_plot,
            b1_ffn_plot,
            output_plot,
        ]

        # All plot outputs for epoch slider updates (17 total)
        all_plot_outputs = (
            all_dynamics_plots
            + all_weight_plots
            + [
                jacobian_plot,
                input_hessian_plot,
                gradient_norm_plot,
                weight_hessian_plot,
                fisher_plot,
                # 3 Adam optimizer plots
                effective_lr_plot,
                adam_ratio_plot,
                update_decay_plot,
            ]
        )

        # Epoch sliders update all plots
        min_epoch_slider.release(
            fn=update_all_plots,
            inputs=[
                exp_dropdown,
                min_epoch_slider,
                max_epoch_slider,
                layer_dropdown,
                show_embeddings,
            ],
            outputs=all_plot_outputs,
        )

        max_epoch_slider.release(
            fn=update_all_plots,
            inputs=[
                exp_dropdown,
                min_epoch_slider,
                max_epoch_slider,
                layer_dropdown,
                show_embeddings,
            ],
            outputs=all_plot_outputs,
        )

        # Show embeddings checkbox toggles section visibility and updates plots
        def toggle_embeddings_section(show: bool):
            return gr.update(visible=show)

        show_embeddings.change(
            fn=toggle_embeddings_section,
            inputs=[show_embeddings],
            outputs=[embed_section],
        )

        show_embeddings.change(
            fn=update_all_plots,
            inputs=[
                exp_dropdown,
                min_epoch_slider,
                max_epoch_slider,
                layer_dropdown,
                show_embeddings,
            ],
            outputs=all_plot_outputs,
        )

        # Layer dropdown no longer affects weight plots (they're now pre-split by group)
        # Keep dropdown for future per-layer curvature metrics if needed

        # ─── Fourier Analysis Event Handlers ───

        def update_fourier_checkpoints(exp_paths: list[str] | str | None):
            """Update checkpoint dropdown when experiment changes."""
            if not exp_paths:
                return gr.update(choices=["best"], value="best")

            if isinstance(exp_paths, str):
                exp_paths = [exp_paths]

            # Use first experiment for checkpoints
            exp = get_experiment(exp_paths[0])
            if not exp or not exp.output_dir:
                return gr.update(choices=["best"], value="best")

            # List available checkpoints
            loader = ExperimentLoader(exp.output_dir.parent)
            try:
                epochs = loader.list_checkpoints(exp)
                choices = ["best"] + [str(e) for e in sorted(epochs)]
            except Exception:
                choices = ["best"]

            return gr.update(choices=choices, value="best")

        def run_fourier_analysis(
            exp_paths: list[str] | str | None,
            checkpoint: str,
            threshold: float,
        ):
            """Run Fourier analysis and update all plots."""
            empty = create_empty_figure("Run analysis to see results")

            if not exp_paths:
                return (
                    "Select an experiment first",
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                )

            if isinstance(exp_paths, str):
                exp_paths = [exp_paths]

            exp = get_experiment(exp_paths[0])
            if not exp:
                return (
                    "Failed to load experiment",
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                )

            # Check model type
            model_type = exp.config.get("model_type", "transformer")
            if model_type != "transformer":
                return (
                    f"Fourier analysis only supports transformer models (got {model_type})",
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                )

            try:
                # Parse checkpoint epoch
                epoch = None if checkpoint == "best" else int(checkpoint)

                # Run analysis
                import torch

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                results = analyze_checkpoint_frequency(
                    exp,
                    epoch=epoch,
                    threshold=threshold,
                    device=device,
                )

                # Extract results
                emb_spectrum = results["embedding_spectrum"]
                logit_spectrum = results["logit_spectrum"]
                ablation = results["ablation_results"]
                p = exp.p

                # Create plots
                emb_fig = create_embedding_spectrum_plot(
                    emb_spectrum.freq_power,
                    emb_spectrum.key_frequencies,
                    p,
                )

                logit_2d_fig = create_logit_spectrum_heatmap(
                    logit_spectrum.power_2d,
                    p,
                )

                logit_diag_fig = create_logit_diagonal_plot(
                    logit_spectrum.diag_power,
                    logit_spectrum.key_frequencies,
                    p,
                )

                key_freq_fig = create_key_frequency_table(
                    logit_spectrum.key_frequencies,
                    logit_spectrum.diag_power,
                    logit_spectrum.fve_per_freq,
                    p,
                )

                ablation_fig = create_ablation_comparison(
                    ablation.full_loss,
                    ablation.full_accuracy,
                    ablation.restricted_loss,
                    ablation.restricted_accuracy,
                    ablation.excluded_loss,
                    ablation.excluded_accuracy,
                )

                fve_fig = create_fve_breakdown_plot(
                    logit_spectrum.fve_per_freq,
                    logit_spectrum.key_frequencies,
                    p,
                )

                # Build status message
                key_freqs_str = ", ".join(str(k) for k in logit_spectrum.key_frequencies[:5])
                status = (
                    f"Analysis complete for {exp.name} (p={p})\n"
                    f"Key frequencies: {key_freqs_str}\n"
                    f"Total FVE: {logit_spectrum.fve_total*100:.1f}%\n"
                    f"Full accuracy: {ablation.full_accuracy*100:.1f}%, "
                    f"Restricted: {ablation.restricted_accuracy*100:.1f}%, "
                    f"Excluded: {ablation.excluded_accuracy*100:.1f}%"
                )

                return (
                    status,
                    emb_fig,
                    logit_2d_fig,
                    logit_diag_fig,
                    key_freq_fig,
                    ablation_fig,
                    fve_fig,
                )

            except Exception as e:
                import traceback

                error_msg = f"Analysis failed: {str(e)}\n{traceback.format_exc()}"
                return (
                    error_msg,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                    empty,
                )

        # Update checkpoint dropdown when experiment changes
        exp_dropdown.change(
            fn=update_fourier_checkpoints,
            inputs=[exp_dropdown],
            outputs=[fourier_checkpoint_dropdown],
        )

        # Run analysis when button clicked
        fourier_analyze_btn.click(
            fn=run_fourier_analysis,
            inputs=[exp_dropdown, fourier_checkpoint_dropdown, fourier_threshold_slider],
            outputs=[
                fourier_status,
                fourier_embedding_plot,
                fourier_logit_2d_plot,
                fourier_logit_diagonal_plot,
                fourier_key_freq_table,
                fourier_ablation_plot,
                fourier_fve_plot,
            ],
        )

    return app


def launch(share: bool = False, port: int = 7860):
    """
    Launch the Gradio application.

    Args:
        share: Whether to create a public shareable link
        port: Port to run on (default: 7860)
    """
    app = create_app()
    app.launch(share=share, server_port=port, theme=gr.themes.Soft())
