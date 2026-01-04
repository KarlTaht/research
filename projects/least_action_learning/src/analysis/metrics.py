"""Post-hoc metric computation from saved artifacts.

Computes aggregate and derived metrics from history DataFrames.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .phases import detect_phases


def compute_aggregates_over_history(
    history_df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> dict[str, dict[str, float]]:
    """Compute aggregate statistics over training history.

    Args:
        history_df: DataFrame with training metrics
        metrics: List of metric column names to analyze
                (None = auto-detect numeric columns)

    Returns:
        Dict mapping metric name to dict with:
        - mean, std, min, max, final_value
        - variance_after_grok (if grokked)
        - pct_above_threshold (for accuracy metrics)

    Example:
        >>> aggs = compute_aggregates_over_history(exp.history_df, ["test_acc", "test_loss"])
        >>> print(aggs["test_acc"]["final_value"])
        0.98
    """
    if metrics is None:
        # Auto-detect numeric columns (excluding step)
        metrics = [
            col
            for col in history_df.columns
            if col != "step" and pd.api.types.is_numeric_dtype(history_df[col])
        ]

    result = {}

    # Detect phases for grok-aware metrics
    phases = detect_phases(history_df)
    grok_step = phases.grokking_start

    for metric in metrics:
        if metric not in history_df.columns:
            continue

        values = history_df[metric].dropna()
        if len(values) == 0:
            continue

        agg = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "final_value": float(values.iloc[-1]),
        }

        # Add variance after grokking
        if grok_step is not None:
            post_grok = history_df.loc[history_df["step"] >= grok_step, metric].dropna()
            if len(post_grok) > 1:
                agg["variance_after_grok"] = float(post_grok.var())

        # Add percentage above threshold for accuracy metrics
        if "acc" in metric.lower():
            agg["pct_above_95"] = float(100.0 * (values >= 0.95).sum() / len(values))
            agg["pct_above_98"] = float(100.0 * (values >= 0.98).sum() / len(values))

        # Add percentage below threshold for loss metrics
        if "loss" in metric.lower():
            agg["pct_below_0.1"] = float(100.0 * (values < 0.1).sum() / len(values))
            agg["pct_below_0.01"] = float(100.0 * (values < 0.01).sum() / len(values))

        result[metric] = agg

    return result


def compute_derived_metrics(history_df: pd.DataFrame) -> pd.DataFrame:
    """Add derived metric columns to history DataFrame.

    Computes the following derived metrics:
    - generalization_gap: train_acc - test_acc
    - loss_ratio: test_loss / train_loss
    - weight_growth_rate: derivative of total_weight_norm (if available)
    - smoothing_rate: derivative of spectral_smoothness (if available)
    - accuracy_delta: change in test_acc from previous step

    Args:
        history_df: DataFrame with training metrics

    Returns:
        New DataFrame with added derived columns
    """
    df = history_df.copy()

    # Generalization gap
    if "train_acc" in df.columns and "test_acc" in df.columns:
        df["generalization_gap"] = df["train_acc"] - df["test_acc"]

    # Loss ratio (test/train) - measures overfitting
    if "train_loss" in df.columns and "test_loss" in df.columns:
        # Avoid division by zero
        df["loss_ratio"] = df["test_loss"] / df["train_loss"].replace(0, np.nan)

    # Weight growth rate (derivative)
    if "total_weight_norm" in df.columns:
        df["weight_growth_rate"] = df["total_weight_norm"].diff() / df["step"].diff()

    # Smoothing rate (derivative of spectral_smoothness)
    if "spectral_smoothness" in df.columns:
        df["smoothing_rate"] = df["spectral_smoothness"].diff() / df["step"].diff()

    # Accuracy delta (change in test_acc)
    if "test_acc" in df.columns:
        df["accuracy_delta"] = df["test_acc"].diff()

    # Gradient-to-weight ratio
    if "gradient_norm" in df.columns and "total_weight_norm" in df.columns:
        df["grad_weight_ratio"] = df["gradient_norm"] / df["total_weight_norm"].replace(0, np.nan)

    return df


def compute_metric_statistics(
    history_df: pd.DataFrame,
    metric: str,
    window_size: int = 100,
) -> pd.DataFrame:
    """Compute rolling statistics for a metric.

    Args:
        history_df: DataFrame with training metrics
        metric: Column name to analyze
        window_size: Rolling window size

    Returns:
        DataFrame with step and rolling statistics columns
    """
    if metric not in history_df.columns:
        return pd.DataFrame(columns=["step", f"{metric}_mean", f"{metric}_std"])

    df = history_df[["step", metric]].copy()
    df[f"{metric}_mean"] = df[metric].rolling(window_size, min_periods=1).mean()
    df[f"{metric}_std"] = df[metric].rolling(window_size, min_periods=1).std()
    df[f"{metric}_min"] = df[metric].rolling(window_size, min_periods=1).min()
    df[f"{metric}_max"] = df[metric].rolling(window_size, min_periods=1).max()

    return df


def find_threshold_crossings(
    history_df: pd.DataFrame,
    metric: str,
    threshold: float,
    direction: str = "above",
) -> list[int]:
    """Find steps where metric crosses threshold.

    Args:
        history_df: DataFrame with training metrics
        metric: Column name to analyze
        threshold: Threshold value
        direction: "above" for rising crossings, "below" for falling

    Returns:
        List of step values where crossings occur
    """
    if metric not in history_df.columns:
        return []

    values = history_df[metric].values
    steps = history_df["step"].values

    crossings = []

    for i in range(1, len(values)):
        if pd.isna(values[i]) or pd.isna(values[i - 1]):
            continue

        if direction == "above":
            if values[i - 1] < threshold <= values[i]:
                crossings.append(int(steps[i]))
        else:  # below
            if values[i - 1] > threshold >= values[i]:
                crossings.append(int(steps[i]))

    return crossings


def compute_time_to_threshold(
    history_df: pd.DataFrame,
    metric: str,
    threshold: float,
    direction: str = "above",
) -> Optional[int]:
    """Compute steps to reach threshold.

    Args:
        history_df: DataFrame with training metrics
        metric: Column name to analyze
        threshold: Target threshold value
        direction: "above" or "below"

    Returns:
        Number of steps to reach threshold, or None if never reached
    """
    crossings = find_threshold_crossings(history_df, metric, threshold, direction)
    return crossings[0] if crossings else None


def compute_correlation_matrix(
    history_df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute correlation matrix between metrics.

    Args:
        history_df: DataFrame with training metrics
        metrics: List of metric columns (None = auto-detect)

    Returns:
        Correlation matrix as DataFrame
    """
    if metrics is None:
        metrics = [
            col
            for col in history_df.columns
            if col != "step" and pd.api.types.is_numeric_dtype(history_df[col])
        ]

    return history_df[metrics].corr()
