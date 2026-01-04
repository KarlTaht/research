"""Experiment store with SQL query interface.

Provides DuckDB-powered queries over all experiment parquet files.
"""

import re
from pathlib import Path
from typing import Optional, Union

import duckdb
import pandas as pd
import yaml

from .loader import get_default_output_dir


def get_configs_dir() -> Path:
    """Get the configs directory containing sweep files."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "configs"


# Cache for sweep groups mapping
_sweep_groups_cache: Optional[dict[str, str]] = None


def load_sweep_groups(force_reload: bool = False) -> dict[str, str]:
    """Parse sweep configs to build experiment name -> group mapping.

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
    """Get the group name for an experiment.

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
    for known_name, group in groups.items():
        # Check for common suffix patterns (like _train30, _train50, etc.)
        if "_wd" in exp_name and "_wd" in known_name:
            exp_suffix = _extract_suffix_after_wd(exp_name)
            known_suffix = _extract_suffix_after_wd(known_name)
            if exp_suffix and exp_suffix == known_suffix:
                exp_prefix = _extract_prefix_before_wd(exp_name)
                known_prefix = _extract_prefix_before_wd(known_name)
                if exp_prefix == known_prefix:
                    return group

    return "no group"


def _extract_suffix_after_wd(name: str) -> str:
    """Extract suffix after weight decay value."""
    match = re.search(r"_wd[\d.]+(.*)$", name)
    return match.group(1) if match else ""


def _extract_prefix_before_wd(name: str) -> str:
    """Extract prefix before weight decay."""
    match = re.search(r"^(.+?)_wd", name)
    return match.group(1) if match else ""


class ExperimentStore:
    """SQL query interface over all experiment parquet files.

    Uses DuckDB for fast analytical queries over parquet files.

    Example:
        store = ExperimentStore()

        # SQL query
        best = store.query('''
            SELECT experiment_name, MIN(test_loss) as best_loss
            FROM experiments WHERE step > 10000
            GROUP BY experiment_name ORDER BY best_loss
        ''')

        # Filter by config
        transformer_exps = store.filter_by_config(model_type="transformer")

        # Get best experiments
        top5 = store.get_best_by_metric("test_acc", minimize=False, top_n=5)
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize store with output directory.

        Args:
            output_dir: Directory containing experiment subdirectories.
                       Defaults to project outputs/ directory.
        """
        if output_dir is None:
            self.output_dir = get_default_output_dir()
        else:
            self.output_dir = Path(output_dir)

    def list_experiments(self) -> list[str]:
        """List all available experiment names.

        Returns:
            List of experiment directory names that have history.parquet
        """
        if not self.output_dir.exists():
            return []

        experiments = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and (path / "history.parquet").exists():
                experiments.append(path.name)

        return sorted(experiments)

    def discover_experiments(self) -> list[Path]:
        """Find all experiment directories containing history.parquet.

        Returns:
            List of experiment directory paths, sorted by modification time (newest first)
        """
        if not self.output_dir.exists():
            return []

        experiments = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and (path / "history.parquet").exists():
                experiments.append(path)

        # Sort by modification time, newest first
        experiments.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return experiments

    def query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query across all experiments.

        The query has access to an 'experiments' table containing all
        history.parquet files with an added 'experiment_name' column.

        Args:
            sql: SQL query string. Use 'experiments' as table name.

        Returns:
            Query results as DataFrame

        Example:
            >>> store.query('''
            ...     SELECT experiment_name, MIN(test_loss) as best_loss
            ...     FROM experiments
            ...     WHERE step > 10000
            ...     GROUP BY experiment_name
            ...     ORDER BY best_loss
            ... ''')
        """
        # Build pattern for all parquet files
        parquet_pattern = str(self.output_dir / "*" / "history.parquet")

        # Connect to DuckDB (in-memory)
        con = duckdb.connect(":memory:")

        try:
            # Create view with experiment_name extracted from path
            con.execute(f"""
                CREATE VIEW experiments AS
                SELECT
                    *,
                    regexp_extract(filename, '.*/([^/]+)/history\\.parquet$', 1) as experiment_name
                FROM read_parquet('{parquet_pattern}', filename=true)
            """)

            result = con.execute(sql).df()
            return result
        except Exception as e:
            print(f"Query error: {e}")
            print(f"SQL: {sql}")
            raise
        finally:
            con.close()

    def get_experiments_summary(self) -> pd.DataFrame:
        """Get summary of all experiments.

        Returns:
            DataFrame with one row per experiment, including:
            - experiment_name
            - max_step
            - final_train_acc, final_test_acc
            - min_test_loss
        """
        return self.query("""
            SELECT
                experiment_name,
                MAX(step) as max_step,
                (SELECT train_acc FROM experiments e2
                 WHERE e2.experiment_name = e1.experiment_name
                 ORDER BY step DESC LIMIT 1) as final_train_acc,
                (SELECT test_acc FROM experiments e2
                 WHERE e2.experiment_name = e1.experiment_name
                 ORDER BY step DESC LIMIT 1) as final_test_acc,
                MIN(test_loss) as min_test_loss,
                MAX(test_acc) as max_test_acc
            FROM experiments e1
            GROUP BY experiment_name
            ORDER BY max_test_acc DESC
        """)

    def filter_by_config(
        self,
        experiments: Optional[list[str]] = None,
        **config_filters,
    ) -> list[str]:
        """Filter experiments by config parameters.

        This requires loading config.json files since parquet only has metrics.

        Args:
            experiments: List of experiment names to filter (None = all)
            **config_filters: Config key-value pairs to match
                             (e.g., model_type="transformer", p=17)

        Returns:
            List of matching experiment names
        """
        import json

        if experiments is None:
            experiments = self.list_experiments()

        matching = []
        for exp_name in experiments:
            config_path = self.output_dir / exp_name / "config.json"
            if not config_path.exists():
                continue

            with open(config_path) as f:
                config = json.load(f)

            # Check all filters match
            match = True
            for key, value in config_filters.items():
                if config.get(key) != value:
                    match = False
                    break

            if match:
                matching.append(exp_name)

        return matching

    def get_sweep_groups(self) -> dict[str, list[str]]:
        """Get experiments grouped by sweep config.

        Returns:
            Dict mapping group name to list of experiment names
        """
        groups_map = load_sweep_groups()
        experiments = self.list_experiments()

        # Build reverse mapping: group -> experiments
        result: dict[str, list[str]] = {}
        for exp_name in experiments:
            group = get_experiment_group(exp_name)
            if group not in result:
                result[group] = []
            result[group].append(exp_name)

        return result

    def get_available_groups(self) -> list[str]:
        """Get list of available experiment groups.

        Returns:
            List of group names, including "All" and "no group"
        """
        groups = load_sweep_groups()
        unique_groups = sorted(set(groups.values()))
        return ["All"] + unique_groups + ["no group"]

    def get_best_by_metric(
        self,
        metric: str,
        minimize: bool = True,
        top_n: int = 10,
        after_step: int = 0,
    ) -> pd.DataFrame:
        """Get top N experiments by a specific metric.

        Args:
            metric: Metric column name to rank by
            minimize: If True, lower is better; if False, higher is better
            top_n: Number of top experiments to return
            after_step: Only consider steps after this value

        Returns:
            DataFrame with top experiments

        Example:
            >>> store.get_best_by_metric('test_loss', minimize=True, top_n=5)
            >>> store.get_best_by_metric('test_acc', minimize=False, top_n=5)
        """
        agg_func = "MIN" if minimize else "MAX"
        order = "ASC" if minimize else "DESC"

        return self.query(f"""
            SELECT
                experiment_name,
                {agg_func}({metric}) as best_{metric}
            FROM experiments
            WHERE step > {after_step}
            GROUP BY experiment_name
            ORDER BY best_{metric} {order}
            LIMIT {top_n}
        """)

    def compare_experiments(
        self,
        experiment_names: list[str],
        metrics: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Compare specific experiments side-by-side.

        Args:
            experiment_names: List of experiment names to compare
            metrics: Specific metrics to include (None = common metrics)

        Returns:
            DataFrame with metrics for all specified experiments
        """
        names_str = ", ".join([f"'{name}'" for name in experiment_names])

        if metrics is None:
            metrics = ["train_loss", "test_loss", "train_acc", "test_acc"]

        metrics_str = ", ".join(["step", "experiment_name"] + metrics)

        return self.query(f"""
            SELECT {metrics_str}
            FROM experiments
            WHERE experiment_name IN ({names_str})
            ORDER BY experiment_name, step
        """)
