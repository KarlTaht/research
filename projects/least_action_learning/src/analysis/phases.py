"""Grokking phase detection and analysis.

Detects memorization, grokking, and plateau phases in training curves.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import pandas as pd

from .loader import ExperimentData
from .store import get_experiment_group


@dataclass
class GrokkingPhases:
    """Detected phases in training.

    Attributes:
        memorization_start: Step when train_acc first crosses threshold
        memorization_end: Step when train_acc peaks before grokking
        grokking_start: Step when test_acc starts rising significantly
        grokking_end: Step when test_acc stabilizes at high level
        plateau_start: Step when both metrics are stable

        grokking_speed: Epochs from memorization to grokking
        stability_after_grok: Variance in test_acc after grokking
    """

    memorization_start: Optional[int] = None
    memorization_end: Optional[int] = None
    grokking_start: Optional[int] = None
    grokking_end: Optional[int] = None
    plateau_start: Optional[int] = None

    grokking_speed: Optional[float] = None
    stability_after_grok: Optional[float] = None


@dataclass
class GrokkingAnalysis:
    """Grokking quality metrics for one experiment.

    Attributes:
        name: Experiment name
        group: Sweep group name
        p: Prime modulus
        train_frac: Train/test split fraction
        lr: Learning rate
        weight_decay: Weight decay value

        grok_step: First step where test_acc >= threshold
        memorization_step: First step where train_acc >= threshold
        pct_train_above_98: Percentage of post-grok steps with train_acc >= 98%
        pct_test_above_95: Percentage of post-grok steps with test_acc >= 95%

        final_train_acc: Final training accuracy
        final_test_acc: Final test accuracy
        test_variance: Variance of test_acc after grok_step
    """

    name: str
    group: str
    p: int
    train_frac: float
    lr: float
    weight_decay: float
    grok_step: Optional[int]
    memorization_step: Optional[int]
    pct_train_above_98: float
    pct_test_above_95: float
    final_train_acc: float
    final_test_acc: float
    test_variance: Optional[float]


def detect_phases(
    history_df: pd.DataFrame,
    train_threshold: float = 0.98,
    test_threshold: float = 0.95,
    stability_window: int = 1000,
) -> GrokkingPhases:
    """Detect training phases from history.

    Args:
        history_df: DataFrame with step, train_acc, test_acc columns
        train_threshold: Threshold for memorization detection
        test_threshold: Threshold for grokking detection
        stability_window: Window size for stability detection

    Returns:
        GrokkingPhases with detected phase boundaries
    """
    phases = GrokkingPhases()

    if history_df.empty:
        return phases

    # Detect memorization (train_acc crosses threshold)
    train_above = history_df["train_acc"] >= train_threshold
    if train_above.any():
        phases.memorization_start = int(history_df.loc[train_above, "step"].min())

    # Detect grokking (test_acc crosses threshold)
    test_above = history_df["test_acc"] >= test_threshold
    if test_above.any():
        phases.grokking_start = int(history_df.loc[test_above, "step"].min())

    # Calculate grokking speed
    if phases.memorization_start is not None and phases.grokking_start is not None:
        phases.grokking_speed = float(phases.grokking_start - phases.memorization_start)

    # Calculate stability after grokking
    if phases.grokking_start is not None:
        post_grok = history_df[history_df["step"] >= phases.grokking_start]
        if len(post_grok) > 1:
            phases.stability_after_grok = float(post_grok["test_acc"].var())

    # Detect grokking end (when test_acc stabilizes)
    if phases.grokking_start is not None:
        post_grok = history_df[history_df["step"] >= phases.grokking_start]
        if len(post_grok) >= stability_window:
            # Find where variance drops below threshold
            rolling_var = post_grok["test_acc"].rolling(stability_window).var()
            stable_mask = rolling_var < 0.001  # Low variance = stable
            if stable_mask.any():
                phases.grokking_end = int(post_grok.loc[stable_mask, "step"].min())
                phases.plateau_start = phases.grokking_end

    return phases


def analyze_grokking(
    experiment: ExperimentData,
    train_thresh: float = 0.98,
    test_thresh: float = 0.95,
) -> GrokkingAnalysis:
    """Compute grokking quality metrics for one experiment.

    Args:
        experiment: ExperimentData instance
        train_thresh: Threshold for "memorized" (default 98%)
        test_thresh: Threshold for "grokked" (default 95%)

    Returns:
        GrokkingAnalysis with percentage-based metrics
    """
    df = experiment.history_df
    n_steps = len(df)

    # Get experiment group
    group = get_experiment_group(experiment.name)

    # Find threshold crossings
    train_above = df["train_acc"] >= train_thresh
    test_above = df["test_acc"] >= test_thresh

    # Find first step where threshold is crossed
    mem_step = df.loc[train_above, "step"].min() if train_above.any() else None
    grok_step = df.loc[test_above, "step"].min() if test_above.any() else None

    # Calculate percentage of steps above thresholds AFTER grokking starts
    # This measures stability post-grokking, not penalizing for time-to-grok
    if grok_step is not None and not pd.isna(grok_step):
        post_grok_df = df.loc[df["step"] >= grok_step]
        n_post_grok = len(post_grok_df)

        if n_post_grok > 0:
            pct_train_above = (
                100.0 * (post_grok_df["train_acc"] >= train_thresh).sum() / n_post_grok
            )
            pct_test_above = 100.0 * (post_grok_df["test_acc"] >= test_thresh).sum() / n_post_grok
            test_variance = float(post_grok_df["test_acc"].var()) if n_post_grok > 1 else 0.0
        else:
            pct_train_above = 0.0
            pct_test_above = 0.0
            test_variance = None
    else:
        # Never grokked - use full history
        pct_train_above = 100.0 * train_above.sum() / n_steps if n_steps > 0 else 0.0
        pct_test_above = 100.0 * test_above.sum() / n_steps if n_steps > 0 else 0.0
        test_variance = None

    return GrokkingAnalysis(
        name=experiment.name,
        group=group,
        p=experiment.p,
        train_frac=experiment.train_frac,
        lr=experiment.lr,
        weight_decay=experiment.weight_decay,
        grok_step=int(grok_step) if grok_step is not None and not pd.isna(grok_step) else None,
        memorization_step=int(mem_step) if mem_step is not None and not pd.isna(mem_step) else None,
        pct_train_above_98=float(pct_train_above),
        pct_test_above_95=float(pct_test_above),
        final_train_acc=float(df["train_acc"].iloc[-1]),
        final_test_acc=float(df["test_acc"].iloc[-1]),
        test_variance=test_variance,
    )


def analyze_all_experiments(
    experiments: dict[str, ExperimentData],
    train_thresh: float = 0.98,
    test_thresh: float = 0.95,
) -> pd.DataFrame:
    """Analyze all experiments and return summary DataFrame.

    Args:
        experiments: Dict mapping name/path to ExperimentData objects
        train_thresh: Threshold for "memorized" (default 98%)
        test_thresh: Threshold for "grokked" (default 95%)

    Returns:
        DataFrame sorted by test percentage (descending), then train percentage
    """
    if not experiments:
        return pd.DataFrame()

    analyses = [analyze_grokking(exp, train_thresh, test_thresh) for exp in experiments.values()]

    df = pd.DataFrame([asdict(a) for a in analyses])

    # Sort by: test percentage (desc), then train percentage (desc), then variance (asc)
    df["_sort_var"] = df["test_variance"].fillna(float("inf"))
    df = df.sort_values(
        ["pct_test_above_95", "pct_train_above_98", "_sort_var"],
        ascending=[False, False, True],
    )
    df = df.drop(columns=["_sort_var"])

    return df.reset_index(drop=True)


def get_phase_metrics(
    history_df: pd.DataFrame,
    phase: str,
    phases: Optional[GrokkingPhases] = None,
) -> pd.DataFrame:
    """Extract metrics for a specific training phase.

    Args:
        history_df: Full history DataFrame
        phase: One of "memorization", "grokking", "plateau"
        phases: Pre-computed phases (computed if None)

    Returns:
        DataFrame filtered to the specified phase
    """
    if phases is None:
        phases = detect_phases(history_df)

    if phase == "memorization":
        start = phases.memorization_start
        end = phases.grokking_start or phases.memorization_end
    elif phase == "grokking":
        start = phases.grokking_start
        end = phases.grokking_end or phases.plateau_start
    elif phase == "plateau":
        start = phases.plateau_start
        end = None
    else:
        raise ValueError(f"Unknown phase: {phase}")

    if start is None:
        return pd.DataFrame()

    mask = history_df["step"] >= start
    if end is not None:
        mask &= history_df["step"] <= end

    return history_df[mask].copy()
