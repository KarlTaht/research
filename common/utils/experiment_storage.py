"""Efficient experiment storage using Parquet + DuckDB.

This module provides utilities for saving and querying ML experiment results
using a hybrid Parquet (storage) + DuckDB (querying) approach.

Key Features:
- Efficient columnar storage with Parquet compression
- Fast analytical queries with DuckDB
- Seamless pandas integration
- Zero data duplication - query Parquet files directly
- 100% local, works offline

Example Usage:
    >>> import pandas as pd
    >>> from common.utils.experiment_storage import save_experiment, query_experiments
    >>>
    >>> # Save experiment results
    >>> results = pd.DataFrame({
    ...     'epoch': [1, 2, 3],
    ...     'perplexity': [25.3, 18.2, 15.1],
    ...     'bleu_score': [0.32, 0.41, 0.45]
    ... })
    >>> save_experiment('llm_tiny_exp001', results, metadata={'model': 'tiny-llm'})
    >>>
    >>> # Query all experiments
    >>> best = query_experiments("SELECT * FROM experiments WHERE perplexity < 20")
    >>> print(best)
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import pandas as pd
import duckdb
import json
from datetime import datetime


def get_experiments_dir() -> Path:
    """Get the default experiments directory."""
    from common.data.hf_utils import get_default_assets_dir

    experiments_dir = get_default_assets_dir() / "outputs" / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    return experiments_dir


def save_experiment(
    experiment_name: str,
    results: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save experiment results to Parquet format.

    Args:
        experiment_name: Unique name for the experiment
        results: DataFrame containing experiment results (metrics, hyperparams, etc.)
        metadata: Optional metadata dict to save alongside results
        output_dir: Custom output directory (default: assets/outputs/experiments/)

    Returns:
        Path to saved Parquet file

    Example:
        >>> results = pd.DataFrame({
        ...     'epoch': [1, 2, 3],
        ...     'train_loss': [2.5, 1.8, 1.2],
        ...     'val_perplexity': [20.1, 15.3, 12.8]
        ... })
        >>> metadata = {
        ...     'model': 'llm-tiny',
        ...     'dataset': 'tiny-textbooks',
        ...     'learning_rate': 0.001
        ... }
        >>> path = save_experiment('exp_001', results, metadata)
    """
    if output_dir is None:
        output_dir = get_experiments_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Add timestamp and experiment name to results
    results = results.copy()
    results["experiment_name"] = experiment_name
    results["saved_at"] = datetime.now().isoformat()

    # Add metadata columns if provided
    if metadata:
        for key, value in metadata.items():
            # Convert to JSON string if value is complex type
            if isinstance(value, (dict, list)):
                results[f"meta_{key}"] = json.dumps(value)
            else:
                results[f"meta_{key}"] = value

    # Save to Parquet
    parquet_path = output_dir / f"{experiment_name}.parquet"
    results.to_parquet(
        parquet_path, compression="snappy", index=False, engine="pyarrow"
    )

    print(f"✓ Experiment saved: {parquet_path}")
    print(f"  Rows: {len(results)}")
    print(f"  Columns: {list(results.columns)}")

    return parquet_path


def load_experiment(
    experiment_name: str, output_dir: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Load a single experiment's results from Parquet.

    Args:
        experiment_name: Name of the experiment to load
        output_dir: Custom directory (default: assets/outputs/experiments/)

    Returns:
        DataFrame with experiment results

    Example:
        >>> df = load_experiment('exp_001')
        >>> print(df[['epoch', 'val_perplexity']])
    """
    if output_dir is None:
        output_dir = get_experiments_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    parquet_path = output_dir / f"{experiment_name}.parquet"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Experiment not found: {experiment_name}")

    return pd.read_parquet(parquet_path)


def query_experiments(
    sql_query: str,
    output_dir: Optional[Union[str, Path]] = None,
    use_experiments_table: bool = True,
) -> pd.DataFrame:
    """
    Query experiment results using SQL with DuckDB.

    Args:
        sql_query: SQL query string. Use 'experiments' table name.
        output_dir: Custom directory (default: assets/outputs/experiments/)
        use_experiments_table: If True, query uses 'experiments' virtual table

    Returns:
        DataFrame with query results

    Examples:
        >>> # Find best performing models
        >>> best = query_experiments('''
        ...     SELECT experiment_name, MIN(val_perplexity) as best_perplexity
        ...     FROM experiments
        ...     WHERE epoch >= 5
        ...     GROUP BY experiment_name
        ...     ORDER BY best_perplexity
        ...     LIMIT 5
        ... ''')
        >>>
        >>> # Compare learning rates
        >>> comparison = query_experiments('''
        ...     SELECT meta_learning_rate, AVG(val_perplexity) as avg_perplexity
        ...     FROM experiments
        ...     GROUP BY meta_learning_rate
        ... ''')
    """
    if output_dir is None:
        output_dir = get_experiments_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    # Pattern to read all Parquet files
    parquet_pattern = str(output_dir / "*.parquet")

    # Connect to DuckDB (in-memory)
    con = duckdb.connect(":memory:")

    if use_experiments_table:
        # Create virtual 'experiments' table from all Parquet files
        con.execute(
            f"CREATE VIEW experiments AS SELECT * FROM read_parquet('{parquet_pattern}')"
        )

    # Execute query and return as DataFrame
    try:
        result = con.execute(sql_query).df()
        return result
    except Exception as e:
        print(f"Query error: {e}")
        print(f"SQL: {sql_query}")
        raise
    finally:
        con.close()


def list_experiments(output_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    List all available experiments.

    Args:
        output_dir: Custom directory (default: assets/outputs/experiments/)

    Returns:
        List of experiment names

    Example:
        >>> experiments = list_experiments()
        >>> print(f"Found {len(experiments)} experiments")
    """
    if output_dir is None:
        output_dir = get_experiments_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    parquet_files = list(output_dir.glob("*.parquet"))
    return [f.stem for f in parquet_files]


def get_experiment_summary(output_dir: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Get summary statistics of all experiments.

    Args:
        output_dir: Custom directory (default: assets/outputs/experiments/)

    Returns:
        DataFrame with experiment summaries

    Example:
        >>> summary = get_experiment_summary()
        >>> print(summary[['experiment_name', 'row_count', 'saved_at']])
    """
    if output_dir is None:
        output_dir = get_experiments_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    pattern = str(output_dir / "*.parquet")
    con = duckdb.connect(":memory:")

    try:
        summary = con.execute(
            f"""
            SELECT
                experiment_name,
                COUNT(*) as row_count,
                MAX(saved_at) as latest_update
            FROM read_parquet('{pattern}')
            GROUP BY experiment_name
            ORDER BY latest_update DESC
        """
        ).df()
        return summary
    finally:
        con.close()


def delete_experiment(
    experiment_name: str, output_dir: Optional[Union[str, Path]] = None
) -> bool:
    """
    Delete an experiment's Parquet file.

    Args:
        experiment_name: Name of experiment to delete
        output_dir: Custom directory (default: assets/outputs/experiments/)

    Returns:
        True if deleted, False if not found

    Example:
        >>> delete_experiment('old_exp_001')
    """
    if output_dir is None:
        output_dir = get_experiments_dir()
    else:
        output_dir = Path(output_dir).expanduser().resolve()

    parquet_path = output_dir / f"{experiment_name}.parquet"

    if parquet_path.exists():
        parquet_path.unlink()
        print(f"✓ Deleted experiment: {experiment_name}")
        return True
    else:
        print(f"✗ Experiment not found: {experiment_name}")
        return False


# Convenience functions for common queries


def get_best_experiments(
    metric: str = "val_perplexity",
    minimize: bool = True,
    top_n: int = 10,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Get top N experiments by a specific metric.

    Args:
        metric: Metric column name to rank by
        minimize: If True, lower is better; if False, higher is better
        top_n: Number of top experiments to return
        output_dir: Custom directory

    Returns:
        DataFrame with top experiments

    Example:
        >>> # Get experiments with lowest perplexity
        >>> best = get_best_experiments('val_perplexity', minimize=True, top_n=5)
        >>>
        >>> # Get experiments with highest BLEU score
        >>> best = get_best_experiments('bleu_score', minimize=False, top_n=5)
    """
    order = "ASC" if minimize else "DESC"

    query = f"""
        SELECT DISTINCT ON (experiment_name)
            *
        FROM experiments
        ORDER BY experiment_name, {metric} {order}
    """

    df = query_experiments(query, output_dir, use_experiments_table=True)

    # Sort again and limit
    df = df.nsmallest(top_n, metric) if minimize else df.nlargest(top_n, metric)

    return df


def compare_experiments(
    experiment_names: List[str],
    metrics: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Compare specific experiments side-by-side.

    Args:
        experiment_names: List of experiment names to compare
        metrics: Specific metrics to include (None = all columns)
        output_dir: Custom directory

    Returns:
        DataFrame with comparison

    Example:
        >>> comparison = compare_experiments(
        ...     ['exp_001', 'exp_002', 'exp_003'],
        ...     metrics=['val_perplexity', 'bleu_score']
        ... )
    """
    names_str = ", ".join([f"'{name}'" for name in experiment_names])

    if metrics:
        metrics_str = ", ".join(["experiment_name"] + metrics)
        query = f"SELECT {metrics_str} FROM experiments WHERE experiment_name IN ({names_str})"
    else:
        query = f"SELECT * FROM experiments WHERE experiment_name IN ({names_str})"

    return query_experiments(query, output_dir, use_experiments_table=True)
