"""Data loading utilities for experiment visualization."""

import json
from dataclasses import dataclass
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

    return ExperimentRun(
        name=exp_dir.name,
        config=config,
        history_df=history_df,
        routing_snapshots=routing_snapshots,
        model_path=model_path,
        p=p,
        n_layers=n_layers,
        n_heads=n_heads,
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
