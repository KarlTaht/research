"""Data loading utilities for experiment visualization."""

import json
import yaml
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


@dataclass
class ExperimentRun:
    """Container for a loaded experiment's data."""

    name: str
    config: dict
    history_df: pd.DataFrame
    routing_snapshots: Optional[list[dict]]  # Routing weights at key steps
    model_path: Optional[Path]
    p: int  # Prime modulus from config
    n_layers: int
    n_heads: int
    group: str = "no group"  # Experiment group from sweep config

    @property
    def has_routing(self) -> bool:
        """Check if this experiment has routing data."""
        return self.routing_snapshots is not None and len(self.routing_snapshots) > 0

    @property
    def max_step(self) -> int:
        """Get the maximum training step."""
        return int(self.history_df["step"].max())


def get_default_output_dir() -> Path:
    """Get the default output directory for experiments."""
    # Look relative to this file's location
    project_root = Path(__file__).parent.parent
    return project_root / "outputs"


def get_configs_dir() -> Path:
    """Get the configs directory containing sweep files."""
    project_root = Path(__file__).parent.parent
    return project_root / "configs"


# Cache for sweep groups mapping
_sweep_groups_cache: Optional[dict[str, str]] = None


def load_sweep_groups(force_reload: bool = False) -> dict[str, str]:
    """
    Parse sweep configs to build experiment name -> group mapping.

    Scans all *_sweep.yaml and *_sweep.yml files in the configs directory
    and builds a mapping from experiment names to sweep group names.

    Args:
        force_reload: If True, clear cache and reload from disk

    Returns:
        Dict mapping experiment name to group name.
        E.g., {"p17_lr3e-4_wd0.5": "transformer_sweep"}
    """
    global _sweep_groups_cache

    if _sweep_groups_cache is not None and not force_reload:
        return _sweep_groups_cache

    groups: dict[str, str] = {}
    configs_dir = get_configs_dir()

    if not configs_dir.exists():
        _sweep_groups_cache = groups
        return groups

    # Find all sweep config files
    sweep_files = list(configs_dir.glob("*_sweep.yaml")) + list(configs_dir.glob("*_sweep.yml"))

    for config_file in sweep_files:
        sweep_name = config_file.stem  # e.g., "transformer_sweep"

        try:
            with open(config_file) as f:
                sweep_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Failed to load sweep config {config_file}: {e}")
            continue

        if sweep_config is None:
            continue

        # Extract experiment names from the experiments list
        experiments = sweep_config.get("experiments", [])
        for exp_params in experiments:
            if isinstance(exp_params, dict):
                exp_name = exp_params.get("name")
                if exp_name:
                    groups[exp_name] = sweep_name

    _sweep_groups_cache = groups
    return groups


def get_experiment_group(exp_name: str) -> str:
    """
    Get the group name for an experiment.

    First checks if explicitly listed in a sweep config. If not, tries to
    infer group membership by matching naming patterns with known sweep
    experiments (e.g., experiments with same prefix/suffix pattern).

    Args:
        exp_name: Experiment directory name

    Returns:
        Group name from sweep config, or "no group" if not in any sweep
    """
    groups = load_sweep_groups()

    # Check if explicitly listed in a sweep config
    if exp_name in groups:
        return groups[exp_name]

    # Try to infer group by pattern matching
    # Extract the suffix pattern (e.g., "_train30" from "p113_lr3e-4_wd0.50_train30")
    # and see if other experiments with same suffix are in a known group
    for known_name, group in groups.items():
        # Check for common suffix patterns (like _train30, _train50, etc.)
        # by comparing the part after the last "wd" parameter
        if "_wd" in exp_name and "_wd" in known_name:
            # Extract suffix after weight decay value (e.g., "_train30")
            exp_suffix = _extract_suffix_after_wd(exp_name)
            known_suffix = _extract_suffix_after_wd(known_name)
            if exp_suffix and exp_suffix == known_suffix:
                # Also check that the prefix pattern matches (p, lr values)
                exp_prefix = _extract_prefix_before_wd(exp_name)
                known_prefix = _extract_prefix_before_wd(known_name)
                if exp_prefix == known_prefix:
                    return group

    return "no group"


def _extract_suffix_after_wd(name: str) -> str:
    """Extract suffix after weight decay value (e.g., '_train30' from 'p113_lr3e-4_wd0.50_train30')."""
    import re
    # Match _wd followed by a number (with optional decimal), then capture the rest
    match = re.search(r"_wd[\d.]+(.*)$", name)
    return match.group(1) if match else ""


def _extract_prefix_before_wd(name: str) -> str:
    """Extract prefix before weight decay (e.g., 'p113_lr3e-4' from 'p113_lr3e-4_wd0.50_train30')."""
    import re
    # Match everything before _wd
    match = re.search(r"^(.+?)_wd", name)
    return match.group(1) if match else ""


def discover_experiments(output_dir: Optional[Path] = None) -> list[Path]:
    """
    Find all experiment directories containing history.parquet.

    Args:
        output_dir: Directory to search (default: project outputs/)

    Returns:
        List of experiment directory paths, sorted by modification time (newest first)
    """
    if output_dir is None:
        output_dir = get_default_output_dir()

    if not output_dir.exists():
        return []

    experiments = []
    for path in output_dir.iterdir():
        if path.is_dir() and (path / "history.parquet").exists():
            experiments.append(path)

    # Sort by modification time, newest first
    experiments.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return experiments


def load_experiment(exp_dir: Path) -> ExperimentRun:
    """
    Load an experiment's data for visualization.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        ExperimentRun containing all loaded data

    Raises:
        FileNotFoundError: If required files are missing
    """
    # Load history
    history_path = exp_dir / "history.parquet"
    if not history_path.exists():
        raise FileNotFoundError(f"No history.parquet found in {exp_dir}")
    history_df = pd.read_parquet(history_path)

    # Load config
    config_path = exp_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Load routing snapshots if available
    routing_path = exp_dir / "routing_snapshots.pt"
    routing_snapshots = None
    if routing_path.exists():
        routing_data = torch.load(routing_path, map_location="cpu", weights_only=False)
        routing_snapshots = routing_data.get("snapshots", [])

    # Find model checkpoint
    model_path = None
    for name in ["best.pt", "final.pt", "checkpoint.pt"]:
        if (exp_dir / name).exists():
            model_path = exp_dir / name
            break

    # Extract key config values
    p = config.get("p", 113)
    n_layers = config.get("n_layers", 4)
    n_heads = config.get("n_heads", 4)

    # Get experiment group from sweep configs
    group = get_experiment_group(exp_dir.name)

    return ExperimentRun(
        name=exp_dir.name,
        config=config,
        history_df=history_df,
        routing_snapshots=routing_snapshots,
        model_path=model_path,
        p=p,
        n_layers=n_layers,
        n_heads=n_heads,
        group=group,
    )


def get_experiment_choices(output_dir: Optional[Path] = None) -> list[str]:
    """
    Get list of experiment paths as strings for Gradio dropdown.

    Args:
        output_dir: Directory to search

    Returns:
        List of experiment directory paths as strings
    """
    experiments = discover_experiments(output_dir)
    return [str(exp) for exp in experiments]


def get_experiment_display_name(exp_path: Path) -> str:
    """
    Create a human-readable display name for an experiment.

    Args:
        exp_path: Path to experiment directory

    Returns:
        Display name like "routed_entropy_113_add (2024-01-15)"
    """
    import datetime

    name = exp_path.name
    mtime = exp_path.stat().st_mtime
    date_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    return f"{name} ({date_str})"


def get_routing_at_step(
    experiment: ExperimentRun,
    step_percent: float,
) -> Optional[dict]:
    """
    Get routing snapshot closest to the given training percentage.

    Args:
        experiment: Loaded experiment
        step_percent: Training progress as percentage (0-100)

    Returns:
        Routing snapshot dict or None if no snapshots available
    """
    if not experiment.has_routing:
        return None

    snapshots = experiment.routing_snapshots
    if not snapshots:
        return None

    # Find snapshot closest to target step
    target_step = int(experiment.max_step * step_percent / 100)

    closest = None
    closest_dist = float("inf")

    for snapshot in snapshots:
        step = snapshot.get("step", 0)
        dist = abs(step - target_step)
        if dist < closest_dist:
            closest_dist = dist
            closest = snapshot

    return closest


@dataclass
class GrokkingAnalysis:
    """Grokking quality metrics for one experiment."""

    name: str
    group: str  # Experiment group from sweep config
    p: int
    lr: float
    weight_decay: float
    grok_step: Optional[int]  # First step where test_acc >= threshold
    memorization_step: Optional[int]  # First step where train_acc >= threshold
    pct_train_above_98: float  # Percentage of steps with train_acc >= 98%
    pct_test_above_95: float  # Percentage of steps with test_acc >= 95%
    final_train_acc: float
    final_test_acc: float
    test_variance: Optional[float]  # Variance of test_acc after grok_step


def analyze_grokking(
    experiment: ExperimentRun,
    train_thresh: float = 0.98,
    test_thresh: float = 0.95,
) -> GrokkingAnalysis:
    """
    Compute grokking quality metrics for one experiment.

    Args:
        experiment: Loaded experiment data
        train_thresh: Threshold for "memorized" (default 98%)
        test_thresh: Threshold for "grokked" (default 95%)

    Returns:
        GrokkingAnalysis with percentage-based metrics
    """
    df = experiment.history_df
    n_steps = len(df)

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
            pct_train_above = 100.0 * (post_grok_df["train_acc"] >= train_thresh).sum() / n_post_grok
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
        group=experiment.group,
        p=experiment.p,
        lr=experiment.config.get("lr", 0),
        weight_decay=experiment.config.get("weight_decay", 0),
        grok_step=int(grok_step) if grok_step is not None and not pd.isna(grok_step) else None,
        memorization_step=int(mem_step) if mem_step is not None and not pd.isna(mem_step) else None,
        pct_train_above_98=float(pct_train_above),
        pct_test_above_95=float(pct_test_above),
        final_train_acc=float(df["train_acc"].iloc[-1]),
        final_test_acc=float(df["test_acc"].iloc[-1]),
        test_variance=test_variance,
    )


def analyze_all_experiments(
    experiments: dict[str, ExperimentRun],
    train_thresh: float = 0.98,
    test_thresh: float = 0.95,
) -> pd.DataFrame:
    """
    Analyze all experiments and return summary DataFrame.

    Args:
        experiments: Dict mapping path strings to ExperimentRun objects
        train_thresh: Threshold for "memorized" (default 98%)
        test_thresh: Threshold for "grokked" (default 95%)

    Returns:
        DataFrame sorted by test percentage (descending), then train percentage
    """
    if not experiments:
        return pd.DataFrame()

    analyses = [
        analyze_grokking(exp, train_thresh, test_thresh) for exp in experiments.values()
    ]

    df = pd.DataFrame([asdict(a) for a in analyses])

    # Sort by: test percentage (desc), then train percentage (desc), then variance (asc)
    df["_sort_var"] = df["test_variance"].fillna(float("inf"))
    df = df.sort_values(
        ["pct_test_above_95", "pct_train_above_98", "_sort_var"],
        ascending=[False, False, True],
    )
    df = df.drop(columns=["_sort_var"])

    return df.reset_index(drop=True)
