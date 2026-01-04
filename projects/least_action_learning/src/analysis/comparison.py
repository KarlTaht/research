"""Multi-experiment comparison utilities.

Tools for aligning, comparing, and analyzing multiple experiments.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .loader import ExperimentData


def align_by_epoch(
    experiments: list[ExperimentData],
    metrics: list[str],
    resample_epochs: Optional[int] = None,
) -> pd.DataFrame:
    """Align metrics across experiments by epoch.

    Creates a wide DataFrame with columns for each experiment's metrics.

    Args:
        experiments: List of ExperimentData objects
        metrics: List of metric column names to include
        resample_epochs: If provided, resample to this many evenly-spaced epochs

    Returns:
        Wide DataFrame with columns like:
        - step
        - {exp_name}_{metric} for each experiment and metric

    Example:
        >>> aligned = align_by_epoch([exp1, exp2], ["test_acc", "test_loss"])
        >>> aligned.columns
        Index(['step', 'exp1_test_acc', 'exp1_test_loss', 'exp2_test_acc', ...])
    """
    if not experiments:
        return pd.DataFrame()

    # Find common step range
    all_steps = set()
    for exp in experiments:
        all_steps.update(exp.history_df["step"].values)
    all_steps = sorted(all_steps)

    if resample_epochs is not None:
        # Resample to evenly-spaced epochs
        min_step, max_step = min(all_steps), max(all_steps)
        all_steps = np.linspace(min_step, max_step, resample_epochs, dtype=int).tolist()

    # Build aligned dataframe
    result = pd.DataFrame({"step": all_steps})

    for exp in experiments:
        df = exp.history_df.set_index("step")
        for metric in metrics:
            if metric not in df.columns:
                continue
            col_name = f"{exp.name}_{metric}"
            # Interpolate to common steps
            result[col_name] = np.interp(
                result["step"],
                exp.history_df["step"],
                exp.history_df[metric],
            )

    return result


def compute_metric_deltas(
    baseline: ExperimentData,
    experiments: list[ExperimentData],
    metrics: list[str],
) -> pd.DataFrame:
    """Compute metric differences relative to baseline.

    Args:
        baseline: Reference experiment
        experiments: List of experiments to compare
        metrics: Metrics to compute deltas for

    Returns:
        DataFrame with columns:
        - step
        - {exp_name}_delta_{metric} for each experiment and metric
    """
    # Align all experiments including baseline
    all_exps = [baseline] + experiments
    aligned = align_by_epoch(all_exps, metrics)

    result = aligned[["step"]].copy()

    for exp in experiments:
        for metric in metrics:
            baseline_col = f"{baseline.name}_{metric}"
            exp_col = f"{exp.name}_{metric}"
            if baseline_col in aligned.columns and exp_col in aligned.columns:
                delta_col = f"{exp.name}_delta_{metric}"
                result[delta_col] = aligned[exp_col] - aligned[baseline_col]

    return result


def group_by_hyperparameter(
    experiments: list[ExperimentData],
    group_by: str,
    metric: str,
    aggregation: str = "final",
) -> pd.DataFrame:
    """Group experiments by hyperparameter for sweep analysis.

    Args:
        experiments: List of ExperimentData objects
        group_by: Config key to group by (e.g., 'weight_decay', 'lr')
        metric: Metric to aggregate
        aggregation: One of "final", "min", "max", "mean"

    Returns:
        DataFrame with group_by value and aggregated metric

    Example:
        >>> results = group_by_hyperparameter(exps, "weight_decay", "test_acc", "max")
        >>> results
           weight_decay  test_acc
        0          0.5      0.95
        1          1.0      0.98
        2          2.0      0.97
    """
    rows = []

    for exp in experiments:
        if group_by not in exp.config:
            continue
        if metric not in exp.history_df.columns:
            continue

        hp_value = exp.config[group_by]
        values = exp.history_df[metric].dropna()

        if len(values) == 0:
            continue

        if aggregation == "final":
            agg_value = values.iloc[-1]
        elif aggregation == "min":
            agg_value = values.min()
        elif aggregation == "max":
            agg_value = values.max()
        elif aggregation == "mean":
            agg_value = values.mean()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        rows.append({group_by: hp_value, metric: agg_value, "experiment": exp.name})

    df = pd.DataFrame(rows)

    # Group by hyperparameter and average across experiments with same value
    if not df.empty:
        summary = df.groupby(group_by)[metric].agg(["mean", "std", "min", "max"]).reset_index()
        summary.columns = [group_by, f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"]
        return summary

    return df


def create_sweep_summary(
    experiments: list[ExperimentData],
    vary_params: list[str],
    metrics: list[str],
    aggregation: str = "final",
) -> pd.DataFrame:
    """Create summary table for hyperparameter sweep.

    Args:
        experiments: List of ExperimentData objects
        vary_params: Config keys that vary in the sweep (e.g., ['lr', 'weight_decay'])
        metrics: Metrics to include in summary
        aggregation: How to aggregate metrics ("final", "min", "max", "mean")

    Returns:
        DataFrame with one row per experiment, columns for params and metrics
    """
    rows = []

    for exp in experiments:
        row = {"experiment": exp.name}

        # Add hyperparameters
        for param in vary_params:
            row[param] = exp.config.get(param)

        # Add aggregated metrics
        for metric in metrics:
            if metric not in exp.history_df.columns:
                row[metric] = None
                continue

            values = exp.history_df[metric].dropna()
            if len(values) == 0:
                row[metric] = None
                continue

            if aggregation == "final":
                row[metric] = values.iloc[-1]
            elif aggregation == "min":
                row[metric] = values.min()
            elif aggregation == "max":
                row[metric] = values.max()
            elif aggregation == "mean":
                row[metric] = values.mean()

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by varying params
    if vary_params and not df.empty:
        df = df.sort_values(vary_params)

    return df.reset_index(drop=True)


def find_best_hyperparameters(
    experiments: list[ExperimentData],
    metric: str,
    minimize: bool = True,
    params: Optional[list[str]] = None,
) -> dict:
    """Find hyperparameters that optimize a metric.

    Args:
        experiments: List of ExperimentData objects
        metric: Metric to optimize
        minimize: If True, find minimum; if False, find maximum
        params: Config keys to report (None = all varying params)

    Returns:
        Dict with best hyperparameters and metric value
    """
    best_exp = None
    best_value = float("inf") if minimize else float("-inf")

    for exp in experiments:
        if metric not in exp.history_df.columns:
            continue

        values = exp.history_df[metric].dropna()
        if len(values) == 0:
            continue

        opt_value = values.min() if minimize else values.max()

        if minimize and opt_value < best_value:
            best_value = opt_value
            best_exp = exp
        elif not minimize and opt_value > best_value:
            best_value = opt_value
            best_exp = exp

    if best_exp is None:
        return {}

    result = {"experiment": best_exp.name, metric: float(best_value)}

    # Add hyperparameters
    if params is None:
        # Include common hyperparameters
        params = ["lr", "weight_decay", "p", "n_layers", "hidden_dim"]

    for param in params:
        if param in best_exp.config:
            result[param] = best_exp.config[param]

    return result


def compute_metric_rankings(
    experiments: list[ExperimentData],
    metrics: list[str],
    ascending: Optional[dict[str, bool]] = None,
) -> pd.DataFrame:
    """Rank experiments by multiple metrics.

    Args:
        experiments: List of ExperimentData objects
        metrics: List of metric columns to rank by
        ascending: Dict mapping metric to sort direction
                  (True = lower is better, False = higher is better)
                  Defaults to True for "loss", False for "acc"

    Returns:
        DataFrame with experiment names and rank columns
    """
    if ascending is None:
        ascending = {}
        for m in metrics:
            if "loss" in m.lower():
                ascending[m] = True
            else:
                ascending[m] = False

    rows = []
    for exp in experiments:
        row = {"experiment": exp.name}
        for metric in metrics:
            if metric in exp.history_df.columns:
                values = exp.history_df[metric].dropna()
                if len(values) > 0:
                    row[metric] = values.iloc[-1]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add rank columns
    for metric in metrics:
        if metric in df.columns:
            asc = ascending.get(metric, True)
            df[f"{metric}_rank"] = df[metric].rank(ascending=asc, method="min")

    return df
