"""Data loading utilities for experiment visualization.

This module provides a thin wrapper over src.analysis for backward compatibility.
New code should import directly from src.analysis.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# Re-export from analysis layer
# Handle both running from project root and from within project directory
try:
    from ..src.analysis.loader import (
        ExperimentData,
        ExperimentLoader,
        get_default_output_dir,
        get_routing_at_step,
    )
    from ..src.analysis.store import (
        ExperimentStore,
        get_configs_dir,
        get_experiment_group,
        load_sweep_groups,
    )
    from ..src.analysis.phases import (
        GrokkingAnalysis,
        analyze_grokking,
        analyze_all_experiments,
    )
except ImportError:
    from src.analysis.loader import (
        ExperimentData,
        ExperimentLoader,
        get_default_output_dir,
        get_routing_at_step,
    )
    from src.analysis.store import (
        ExperimentStore,
        get_configs_dir,
        get_experiment_group,
        load_sweep_groups,
    )
    from src.analysis.phases import (
        GrokkingAnalysis,
        analyze_grokking,
        analyze_all_experiments,
    )

# Backward compatibility alias
ExperimentRun = ExperimentData

# Module-level loader instance for convenience
_loader = ExperimentLoader()
_store = ExperimentStore()


def discover_experiments(output_dir: Optional[Path] = None) -> list[Path]:
    """Find all experiment directories containing history.parquet.

    Args:
        output_dir: Directory to search (default: project outputs/)

    Returns:
        List of experiment directory paths, sorted by modification time (newest first)
    """
    store = ExperimentStore(output_dir) if output_dir else _store
    return store.discover_experiments()


def load_experiment(exp_dir: Path) -> ExperimentRun:
    """Load an experiment's data for visualization.

    Args:
        exp_dir: Path to experiment directory

    Returns:
        ExperimentRun (alias for ExperimentData) containing all loaded data

    Raises:
        FileNotFoundError: If required files are missing
    """
    exp = _loader.load_from_path(exp_dir)

    # Add group attribute for backward compatibility
    # ExperimentData doesn't have group by default, but ExperimentRun did
    # We add it as a dynamic attribute
    exp.group = get_experiment_group(exp.name)

    return exp


def get_experiment_choices(output_dir: Optional[Path] = None) -> list[str]:
    """Get list of experiment paths as strings for Gradio dropdown.

    Args:
        output_dir: Directory to search

    Returns:
        List of experiment directory paths as strings
    """
    experiments = discover_experiments(output_dir)
    return [str(exp) for exp in experiments]


def get_experiment_display_name(exp_path: Path) -> str:
    """Create a human-readable display name for an experiment.

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


# Re-export for backward compatibility
__all__ = [
    # Classes
    "ExperimentRun",
    "ExperimentData",
    "ExperimentLoader",
    "ExperimentStore",
    "GrokkingAnalysis",
    # Functions
    "discover_experiments",
    "load_experiment",
    "get_experiment_choices",
    "get_experiment_display_name",
    "get_routing_at_step",
    "get_experiment_group",
    "load_sweep_groups",
    "get_default_output_dir",
    "get_configs_dir",
    "analyze_grokking",
    "analyze_all_experiments",
]
