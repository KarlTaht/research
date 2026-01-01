"""Centralized styling constants and layout helpers for visualization."""

import plotly.graph_objects as go

# Color palette for categorical data (heads, layers, experiments)
CATEGORICAL_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

# Alias for backward compatibility
HEAD_COLORS = CATEGORICAL_COLORS[:8]

# Metric display names for plot titles and legends
METRIC_DISPLAY_NAMES = {
    # Basic training metrics
    "test_acc": "Test Accuracy",
    "train_acc": "Train Accuracy",
    "test_loss": "Test Loss",
    "train_loss": "Train Loss",
    # Model properties
    "total_weight_norm": "Weight Norm",
    "representation_norm": "Representation Norm",
    # Input sensitivity (grad w.r.t. inputs)
    "jacobian_norm": "Jacobian Norm",
    "hessian_trace": "Hessian Trace",
    "spectral_smoothness": "Spectral Smoothness",
    # Weight curvature (grad w.r.t. weights - loss landscape)
    "gradient_norm": "Gradient Norm (||grad_w L||)",
    "weight_hessian_trace": "Weight Hessian Trace",
    "fisher_trace": "Fisher Trace",
    # Adam optimizer dynamics
    "effective_lr_mean": "Effective LR (mean sqrt(v))",
    "effective_lr_max": "Effective LR (max sqrt(v))",
    "adam_ratio_mean": "Adam Ratio (mean)",
    "adam_ratio_max": "Adam Ratio (max)",
    "update_decay_ratio": "Update/Decay Ratio",
}

# Metrics that should use logarithmic scale
LOG_SCALE_METRICS = {
    "test_loss",
    "train_loss",
    "jacobian_norm",
    "gradient_norm",
    "weight_hessian_trace",
    "fisher_trace",
    "effective_lr_mean",
    "effective_lr_max",
    "adam_ratio_mean",
    "adam_ratio_max",
    "update_decay_ratio",
}


def apply_standard_layout(
    fig: go.Figure,
    title: str,
    height: int = 300,
    show_legend: bool = True,
    y_title: str = None,
) -> go.Figure:
    """Apply consistent layout styling to a figure.

    Args:
        fig: Plotly figure to modify
        title: Plot title
        height: Figure height in pixels
        show_legend: Whether to show the legend
        y_title: Y-axis title (defaults to title if not specified)

    Returns:
        The modified figure (for chaining)
    """
    legend_config = (
        dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        if show_legend
        else dict(visible=False)
    )

    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title=y_title or title,
        hovermode="x unified",
        height=height,
        legend=legend_config,
    )
    return fig


def get_metric_display_name(metric: str) -> str:
    """Get the display name for a metric.

    Args:
        metric: Metric column name (e.g., "test_acc")

    Returns:
        Human-readable display name
    """
    return METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())


def should_use_log_scale(metric: str) -> bool:
    """Check if a metric should use logarithmic scale.

    Args:
        metric: Metric column name

    Returns:
        True if log scale is recommended
    """
    return metric in LOG_SCALE_METRICS
