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
from .plots import create_grokking_summary_table
from .helpers import create_empty_figure, filter_by_epoch_range, get_valid_data
from .styles import CATEGORICAL_COLORS

# Import architecture utilities from analysis layer
# Handle both running from project root and from within project directory
try:
    from ..src.analysis.architecture import (
        get_layer_names,
        get_layer_groups,
        get_layer_display_name,
        get_layer_choices,
    )
except ImportError:
    from src.analysis.architecture import (
        get_layer_names,
        get_layer_groups,
        get_layer_display_name,
        get_layer_choices,
    )


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
        return [
            (get_experiment_display_name(Path(path)), path)
            for path in experiments.keys()
        ]

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
    fig.add_hline(y=0.95, line_dash="dot", line_color="gray",
                  annotation_text="95%", annotation_position="right")
    fig.add_hline(y=0.98, line_dash="dot", line_color="lightgray",
                  annotation_text="98%", annotation_position="right")

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

    layer_cols = [c for c in df.columns if c.startswith("layer_") and c.endswith(f"_{metric_suffix}")]

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
            if group_filter == exp.group or (group_filter == "no group" and exp.group == "no group"):
                filtered[path] = exp
        experiments = filtered

    if not experiments:
        return create_empty_figure(f"No experiments in group '{group_filter}'")

    analysis_df = analyze_all_experiments(experiments)
    table_fig = create_grokking_summary_table(analysis_df)

    return table_fig


def update_experiment_controls(exp_path: str) -> Tuple:
    """Update control widgets when experiment changes."""
    if not exp_path:
        return (
            gr.update(maximum=10000, value=0),
            gr.update(maximum=10000, value=10000),
            gr.update(choices=["All"], value="All"),
        )

    exp = get_experiment(exp_path)
    if exp is None:
        return (
            gr.update(maximum=10000, value=0),
            gr.update(maximum=10000, value=10000),
            gr.update(choices=["All"], value="All"),
        )

    # Get max step from experiment
    max_step = int(exp.history_df["step"].max()) if not exp.history_df.empty else 10000

    # Get layer choices with descriptive names (e.g., "0:tok_embed", "2:b0_attn_Q")
    layer_choices = get_layer_choices(exp, metric_suffix="weight_norm")

    return (
        gr.update(maximum=max_step, value=0),
        gr.update(maximum=max_step, value=max_step),
        gr.update(choices=layer_choices, value="All"),
    )


def update_training_dynamics_plots(
    exp_path: str,
    min_epoch: int,
    max_epoch: int,
) -> Tuple:
    """Update training dynamics plots (loss, accuracy, learning rate).

    Returns 3 plots:
    - Loss curves (train dotted, test solid)
    - Accuracy curves (train dotted, test solid)
    - Learning rate schedule
    """
    if not exp_path:
        empty = create_empty_figure("Select an experiment")
        return empty, empty, empty

    exp = get_experiment(exp_path)
    if exp is None:
        empty = create_empty_figure("Failed to load experiment")
        return empty, empty, empty

    df = exp.history_df

    # Loss curves
    loss_fig = create_loss_curves_plot(df, min_epoch=min_epoch, max_epoch=max_epoch)

    # Accuracy curves
    acc_fig = create_accuracy_curves_plot(df, min_epoch=min_epoch, max_epoch=max_epoch)

    # Learning rate
    lr_fig = create_learning_rate_plot(exp, min_epoch=min_epoch, max_epoch=max_epoch)

    return loss_fig, acc_fig, lr_fig


def update_weight_plots(
    exp_path: str,
    min_epoch: int,
    max_epoch: int,
    selected_layer: str,
) -> Tuple:
    """Update weight analysis plots with expanded layout.

    Returns 6 plots:
    - Embeddings (tok_embed, pos_embed)
    - Block 0 Attention (Q, K, V, Wo)
    - Block 0 FFN (up, down)
    - Block 1 Attention (Q, K, V, Wo)
    - Block 1 FFN (up, down)
    - Output head

    Note: UI currently supports 2 blocks. To support more blocks,
    update the UI layout in create_app() as well.
    """
    # UI currently hardcoded for 2 blocks (4 block plots + embed + output = 6)
    n_ui_blocks = 2
    n_plots = 2 + n_ui_blocks * 2  # embed + blocks*2 + output

    if not exp_path:
        empty = create_empty_figure("Select an experiment")
        return (empty,) * n_plots

    exp = get_experiment(exp_path)
    if exp is None:
        empty = create_empty_figure("Failed to load experiment")
        return (empty,) * n_plots

    df = exp.history_df
    n_layers = exp.config.get("n_layers", exp.n_layers)
    model_type = exp.config.get("model_type", "transformer")
    groups = get_layer_groups(model_type, n_layers)

    # Embeddings plot (always first)
    emb_indices, emb_names = groups["embeddings"]
    embed_fig = create_weight_group_plot(
        df, emb_indices, emb_names, "Embeddings",
        min_epoch=min_epoch, max_epoch=max_epoch
    )

    # Block plots via loop (attention then FFN for each block)
    block_figs = []
    component_names = {"attn": "Attention", "ffn": "FFN"}
    for block_idx in range(n_ui_blocks):
        for component in ["attn", "ffn"]:
            key = f"block{block_idx}_{component}"
            if key in groups:
                indices, names = groups[key]
                fig = create_weight_group_plot(
                    df, indices, names,
                    f"Block {block_idx}: {component_names[component]}",
                    min_epoch=min_epoch, max_epoch=max_epoch
                )
            else:
                fig = create_empty_figure(f"No Block {block_idx} {component} data")
            block_figs.append(fig)

    # Output head plot (always last)
    out_indices, out_names = groups["output"]
    output_fig = create_weight_group_plot(
        df, out_indices, out_names, "Output Head",
        min_epoch=min_epoch, max_epoch=max_epoch
    )

    return (embed_fig, *block_figs, output_fig)


def update_curvature_plots(
    exp_path: str,
    min_epoch: int,
    max_epoch: int,
) -> Tuple:
    """Update curvature analysis plots (both input-sensitivity and weight-curvature)."""
    if not exp_path:
        empty = create_empty_figure("Select an experiment")
        return empty, empty, empty, empty, empty

    exp = get_experiment(exp_path)
    if exp is None:
        empty = create_empty_figure("Failed to load experiment")
        return empty, empty, empty, empty, empty

    df = exp.history_df

    # Input-sensitivity metrics (∇_x)
    jacobian_fig = create_single_metric_plot(
        df, "jacobian_norm", "Jacobian Norm (∇ₓf)", log_scale=True,
        min_epoch=min_epoch, max_epoch=max_epoch
    )

    input_hessian_fig = create_single_metric_plot(
        df, "hessian_trace", "Input Hessian Trace (∇²ₓf)",
        min_epoch=min_epoch, max_epoch=max_epoch
    )

    # Weight-curvature metrics (∇_w)
    gradient_norm_fig = create_single_metric_plot(
        df, "gradient_norm", "Gradient Norm (∇ᵥL)", log_scale=True,
        min_epoch=min_epoch, max_epoch=max_epoch
    )

    weight_hessian_fig = create_single_metric_plot(
        df, "weight_hessian_trace", "Weight Hessian (∇²ᵥL)",
        min_epoch=min_epoch, max_epoch=max_epoch
    )

    fisher_fig = create_single_metric_plot(
        df, "fisher_trace", "Fisher Trace (∇L·∇Lᵀ)",
        min_epoch=min_epoch, max_epoch=max_epoch
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
    exp_path: str,
    min_epoch: int,
    max_epoch: int,
) -> Tuple:
    """Update Adam optimizer dynamics plots.

    Returns 3 plots:
    - Effective LR (mean and max sqrt(v_t))
    - Adam ratio (signal-to-noise: |m|/(sqrt(v)+eps))
    - Update/Decay ratio (learning vs forgetting)
    """
    if not exp_path:
        empty = create_empty_figure("Select an experiment")
        return empty, empty, empty

    exp = get_experiment(exp_path)
    if exp is None:
        empty = create_empty_figure("Failed to load experiment")
        return empty, empty, empty

    df = exp.history_df
    filtered_df = filter_by_epoch_range(df, min_epoch, max_epoch)

    # Effective LR plot (show both mean and max on same plot)
    effective_lr_fig = go.Figure()
    has_eff_lr_data = False

    for metric, name, color in [
        ("effective_lr_mean", "Mean", CATEGORICAL_COLORS[0]),
        ("effective_lr_max", "Max", CATEGORICAL_COLORS[1]),
    ]:
        y_data, valid_mask = get_valid_data(filtered_df, metric)
        if valid_mask.any():
            has_eff_lr_data = True
            effective_lr_fig.add_trace(
                go.Scatter(
                    x=filtered_df.loc[valid_mask, "step"],
                    y=filtered_df.loc[valid_mask, metric],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                    hovertemplate=f"{name}<br>Step: %{{x}}<br>√v: %{{y:.4g}}<extra></extra>",
                )
            )

    if not has_eff_lr_data:
        effective_lr_fig = create_empty_figure("No effective LR data")
    else:
        effective_lr_fig.update_layout(
            title="Effective LR (√v_t)",
            xaxis_title="Step",
            yaxis_title="√v_t",
            yaxis_type="log",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300,
        )

    # Adam ratio plot
    adam_ratio_fig = create_single_metric_plot(
        df, "adam_ratio_mean", "Adam Ratio (|m|/(√v+ε))", log_scale=True,
        min_epoch=min_epoch, max_epoch=max_epoch
    )

    # Update/Decay ratio plot
    update_decay_fig = create_single_metric_plot(
        df, "update_decay_ratio", "Update/Decay Ratio", log_scale=True,
        min_epoch=min_epoch, max_epoch=max_epoch
    )

    return effective_lr_fig, adam_ratio_fig, update_decay_fig


def update_all_plots(
    exp_path: str,
    min_epoch: int,
    max_epoch: int,
    selected_layer: str,
) -> Tuple:
    """Update all analysis plots when experiment or controls change.

    Returns 17 plots total:
    - 3 training dynamics plots (loss_curves, accuracy_curves, learning_rate)
    - 6 weight plots (embed, b0_attn, b0_ffn, b1_attn, b1_ffn, output)
    - 5 curvature plots (jacobian, input_hessian, gradient, weight_hessian, fisher)
    - 3 Adam optimizer plots (effective_lr, adam_ratio, update_decay)
    """
    dynamics_plots = update_training_dynamics_plots(exp_path, min_epoch, max_epoch)
    weight_plots = update_weight_plots(
        exp_path, min_epoch, max_epoch, selected_layer
    )
    curvature_plots = update_curvature_plots(exp_path, min_epoch, max_epoch)
    adam_plots = update_adam_plots(exp_path, min_epoch, max_epoch)

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
                    label="Experiment",
                    info="Select an experiment to analyze",
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
                    info="End of epoch range",
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
        gr.Markdown(
            "*Loss and accuracy curves over training. "
            "Train (dotted), test (solid).*"
        )
        with gr.Row():
            loss_curves_plot = gr.Plot(label="Loss Curves")
            accuracy_curves_plot = gr.Plot(label="Accuracy Curves")
            learning_rate_plot = gr.Plot(label="Learning Rate Schedule")

        gr.Markdown("---")

        # ─── Section 4: Weight Analysis ───
        gr.Markdown("## Weight Analysis")
        gr.Markdown(
            "*Track how model weights evolve during training. "
            "Weight norm growth often precedes grokking.*"
        )

        # Row 1: Embeddings
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

        gr.Markdown("---")

        # ─── Section 5: Curvature Analysis ───
        gr.Markdown("## Curvature Analysis")
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

        gr.Markdown("---")

        # ─── Section 6: Adam Optimizer Dynamics ───
        gr.Markdown("## Adam Optimizer Dynamics")
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
                    if exp and (selected_group == exp.group or
                               (selected_group == "no group" and exp.group == "no group")):
                        filtered.append((display, path))
                dropdown_update = gr.update(choices=filtered if filtered else all_choices, value=None)

            # Update summary table with same filter
            table_fig = update_grokking_summary(selected_group)
            return dropdown_update, table_fig

        group_dropdown.change(
            fn=filter_experiments_by_group,
            inputs=[group_dropdown],
            outputs=[exp_dropdown, summary_table],
        )

        # Experiment selection updates controls and all plots
        def on_experiment_change(exp_path):
            controls = update_experiment_controls(exp_path)
            # Use default epoch range and layer for initial plots
            if exp_path:
                exp = get_experiment(exp_path)
                if exp:
                    max_step = int(exp.history_df["step"].max()) if not exp.history_df.empty else 10000
                    plots = update_all_plots(exp_path, 0, max_step, "All")
                    return controls + plots
            empty = create_empty_figure("Select an experiment")
            # 17 empty plots: 3 dynamics + 6 weight + 5 curvature + 3 adam
            return controls + (empty,) * 17

        exp_dropdown.change(
            fn=on_experiment_change,
            inputs=[exp_dropdown],
            outputs=[
                min_epoch_slider, max_epoch_slider, layer_dropdown,
                # 3 training dynamics plots
                loss_curves_plot, accuracy_curves_plot, learning_rate_plot,
                # 6 weight plots
                embed_plot, b0_attn_plot, b0_ffn_plot,
                b1_attn_plot, b1_ffn_plot, output_plot,
                # 5 curvature plots
                jacobian_plot, input_hessian_plot,
                gradient_norm_plot, weight_hessian_plot, fisher_plot,
                # 3 Adam optimizer plots
                effective_lr_plot, adam_ratio_plot, update_decay_plot,
            ],
        )

        # Training dynamics plot outputs
        all_dynamics_plots = [loss_curves_plot, accuracy_curves_plot, learning_rate_plot]

        # All weight plot outputs
        all_weight_plots = [
            embed_plot, b0_attn_plot, b0_ffn_plot,
            b1_attn_plot, b1_ffn_plot, output_plot,
        ]

        # All plot outputs for epoch slider updates (17 total)
        all_plot_outputs = all_dynamics_plots + all_weight_plots + [
            jacobian_plot, input_hessian_plot,
            gradient_norm_plot, weight_hessian_plot, fisher_plot,
            # 3 Adam optimizer plots
            effective_lr_plot, adam_ratio_plot, update_decay_plot,
        ]

        # Epoch sliders update all plots
        min_epoch_slider.release(
            fn=update_all_plots,
            inputs=[exp_dropdown, min_epoch_slider, max_epoch_slider, layer_dropdown],
            outputs=all_plot_outputs,
        )

        max_epoch_slider.release(
            fn=update_all_plots,
            inputs=[exp_dropdown, min_epoch_slider, max_epoch_slider, layer_dropdown],
            outputs=all_plot_outputs,
        )

        # Layer dropdown no longer affects weight plots (they're now pre-split by group)
        # Keep dropdown for future per-layer curvature metrics if needed

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
