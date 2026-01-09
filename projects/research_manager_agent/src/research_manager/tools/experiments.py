"""Tools for querying and analyzing experiments."""

import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from research_manager.tools.decorators import tool
from research_manager.tools.registry import ToolCategory


# Natural language to SQL patterns
NL_PATTERNS = [
    # Best/worst queries
    (r"best\s+(\w+)", "SELECT * FROM experiments ORDER BY {0} ASC LIMIT 10"),
    (r"worst\s+(\w+)", "SELECT * FROM experiments ORDER BY {0} DESC LIMIT 10"),
    (r"highest\s+(\w+)", "SELECT * FROM experiments ORDER BY {0} DESC LIMIT 10"),
    (r"lowest\s+(\w+)", "SELECT * FROM experiments ORDER BY {0} ASC LIMIT 10"),
    # Top N queries
    (r"top\s+(\d+)\s+by\s+(\w+)", "SELECT * FROM experiments ORDER BY {1} ASC LIMIT {0}"),
    (r"top\s+(\d+)", "SELECT * FROM experiments LIMIT {0}"),
    # Filter queries
    (r"(\w+)\s*<\s*([\d.]+)", "SELECT * FROM experiments WHERE {0} < {1}"),
    (r"(\w+)\s*>\s*([\d.]+)", "SELECT * FROM experiments WHERE {0} > {1}"),
    (r"(\w+)\s*=\s*([\d.]+)", "SELECT * FROM experiments WHERE {0} = {1}"),
    # Recent experiments
    (r"recent|latest", "SELECT * FROM experiments ORDER BY saved_at DESC LIMIT 10"),
    # All experiments
    (r"all|list", "SELECT DISTINCT experiment_name FROM experiments"),
]


def _natural_language_to_sql(query: str) -> Optional[str]:
    """Convert natural language query to SQL.

    Args:
        query: Natural language query string.

    Returns:
        SQL query string or None if no pattern matches.
    """
    query_lower = query.lower().strip()

    # Check if it's already SQL
    if query_lower.startswith("select"):
        return query

    # Try each pattern
    for pattern, sql_template in NL_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            groups = match.groups()
            return sql_template.format(*groups)

    return None


@tool(
    name="query_experiments",
    description="Query experiment results using natural language or SQL. Find best runs, compare metrics, filter by criteria.",
    category=ToolCategory.EXPERIMENTS,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query or SQL. Examples: 'best perplexity', 'top 5 by loss', 'perplexity < 20', or full SQL",
            },
            "project": {
                "type": "string",
                "description": "Filter to specific project (optional)",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default: 20)",
            },
        },
        "required": ["query"],
    },
    read_only=True,
    examples=[
        "query_experiments(query='best perplexity')",
        "query_experiments(query='top 5 by loss')",
        "query_experiments(query='SELECT * FROM experiments WHERE perplexity < 20')",
    ],
)
async def query_experiments(
    query: str,
    project: Optional[str] = None,
    limit: int = 20,
    experiments_dir: Optional[Path] = None,
) -> dict:
    """Query experiment results.

    Args:
        query: Natural language or SQL query.
        project: Filter to specific project.
        limit: Maximum results.
        experiments_dir: Override experiments directory.

    Returns:
        Dictionary with query results.
    """
    try:
        from common.utils.experiment_storage import (
            query_experiments as _query_experiments,
            list_experiments,
            get_experiments_dir,
        )
    except ImportError:
        return {"error": "Experiment storage not available. Is common package installed?"}

    if experiments_dir is None:
        experiments_dir = get_experiments_dir()

    # Check if any experiments exist
    available = list_experiments(experiments_dir)
    if not available:
        return {
            "error": "No experiments found",
            "experiments_dir": str(experiments_dir),
            "hint": "Run some training experiments first, or check the experiments directory.",
        }

    # Convert natural language to SQL
    sql_query = _natural_language_to_sql(query)

    if sql_query is None:
        # Fall back to a simple search
        sql_query = (
            f"SELECT * FROM experiments WHERE experiment_name LIKE '%{query}%' LIMIT {limit}"
        )

    # Add project filter if specified
    if project:
        if "WHERE" in sql_query.upper():
            sql_query = sql_query.replace("WHERE", f"WHERE experiment_name LIKE '%{project}%' AND")
        else:
            where_pos = sql_query.upper().find("FROM EXPERIMENTS")
            if where_pos > 0:
                insert_pos = where_pos + len("FROM EXPERIMENTS")
                sql_query = (
                    sql_query[:insert_pos]
                    + f" WHERE experiment_name LIKE '%{project}%'"
                    + sql_query[insert_pos:]
                )

    # Apply limit if not already in query
    if "LIMIT" not in sql_query.upper():
        sql_query += f" LIMIT {limit}"

    try:
        df = _query_experiments(sql_query, experiments_dir)

        # Convert to list of dicts
        results = df.to_dict(orient="records")

        return {
            "query": query,
            "sql": sql_query,
            "results": results,
            "count": len(results),
            "available_experiments": available[:10],  # Show first 10
            "total_experiments": len(available),
        }
    except Exception as e:
        return {
            "error": f"Query failed: {e}",
            "query": query,
            "sql": sql_query,
            "available_experiments": available,
            "hint": "Try a simpler query or check the SQL syntax.",
        }


@tool(
    name="list_experiments",
    description="List all available experiments with optional summary statistics.",
    category=ToolCategory.EXPERIMENTS,
    parameters={
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Filter to specific project",
            },
            "include_summary": {
                "type": "boolean",
                "description": "Include summary statistics (default: true)",
            },
        },
        "required": [],
    },
    read_only=True,
    examples=[
        "list_experiments()",
        "list_experiments(project='custom_transformer')",
    ],
)
async def list_experiments_tool(
    project: Optional[str] = None,
    include_summary: bool = True,
    experiments_dir: Optional[Path] = None,
) -> dict:
    """List all available experiments.

    Args:
        project: Filter to specific project.
        include_summary: Include summary statistics.
        experiments_dir: Override experiments directory.

    Returns:
        Dictionary with experiment list.
    """
    try:
        from common.utils.experiment_storage import (
            list_experiments,
            get_experiment_summary,
            get_experiments_dir,
        )
    except ImportError:
        return {"error": "Experiment storage not available. Is common package installed?"}

    if experiments_dir is None:
        experiments_dir = get_experiments_dir()

    experiments = list_experiments(experiments_dir)

    # Filter by project
    if project:
        experiments = [e for e in experiments if project.lower() in e.lower()]

    result = {
        "experiments": experiments,
        "count": len(experiments),
        "experiments_dir": str(experiments_dir),
    }

    if include_summary and experiments:
        try:
            summary_df = get_experiment_summary(experiments_dir)
            if project:
                summary_df = summary_df[
                    summary_df["experiment_name"].str.contains(project, case=False)
                ]
            result["summary"] = summary_df.to_dict(orient="records")
        except Exception as e:
            result["summary_error"] = str(e)

    return result


@tool(
    name="analyze_logs",
    description="Analyze training logs to extract metrics, errors, and patterns.",
    category=ToolCategory.EXPERIMENTS,
    parameters={
        "type": "object",
        "properties": {
            "log_path": {
                "type": "string",
                "description": "Path to log file or directory",
            },
            "experiment": {
                "type": "string",
                "description": "Experiment name to find logs for",
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for",
            },
        },
        "required": [],
    },
    read_only=True,
    examples=[
        "analyze_logs(experiment='custom_transformer_001')",
        "analyze_logs(log_path='logs/train.log')",
        "analyze_logs(pattern='loss: [\\d.]+')",
    ],
)
async def analyze_logs(
    log_path: Optional[str] = None,
    experiment: Optional[str] = None,
    pattern: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> dict:
    """Analyze training logs.

    Args:
        log_path: Path to log file or directory.
        experiment: Experiment name to find logs for.
        pattern: Regex pattern to search for.
        repo_root: Repository root path.

    Returns:
        Dictionary with log analysis.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    # Find log files
    log_files = []

    if log_path:
        path = repo_root / log_path
        if path.is_file():
            log_files.append(path)
        elif path.is_dir():
            log_files.extend(path.glob("**/*.log"))
            log_files.extend(path.glob("**/*.txt"))
    elif experiment:
        # Search common log locations
        search_paths = [
            repo_root / "assets" / "outputs" / "logs",
            repo_root / "logs",
            repo_root / "outputs",
        ]
        for search_path in search_paths:
            if search_path.exists():
                log_files.extend(search_path.glob(f"**/*{experiment}*"))
    else:
        # Default: look for recent logs
        log_dirs = [
            repo_root / "assets" / "outputs" / "logs",
            repo_root / "logs",
        ]
        for log_dir in log_dirs:
            if log_dir.exists():
                log_files.extend(
                    sorted(log_dir.glob("**/*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[
                        :5
                    ]
                )

    if not log_files:
        return {
            "error": "No log files found",
            "searched": [str(log_path) if log_path else "default log directories"],
            "hint": "Specify a log_path or experiment name.",
        }

    analysis = {
        "files_analyzed": [],
        "metrics": {},
        "errors": [],
        "warnings": [],
        "pattern_matches": [],
    }

    # Common metric patterns
    metric_patterns = [
        (r"loss[:\s]+([0-9.]+)", "loss"),
        (r"perplexity[:\s]+([0-9.]+)", "perplexity"),
        (r"accuracy[:\s]+([0-9.]+)", "accuracy"),
        (r"epoch[:\s]+(\d+)", "epoch"),
        (r"step[:\s]+(\d+)", "step"),
        (r"lr[:\s]+([0-9.e-]+)", "learning_rate"),
    ]

    for log_file in log_files[:10]:  # Limit to 10 files
        try:
            content = log_file.read_text(errors="ignore")
            lines = content.split("\n")

            file_info = {
                "path": str(log_file.relative_to(repo_root)),
                "size_kb": log_file.stat().st_size / 1024,
                "lines": len(lines),
            }
            analysis["files_analyzed"].append(file_info)

            # Extract metrics
            for regex, metric_name in metric_patterns:
                matches = re.findall(regex, content, re.IGNORECASE)
                if matches:
                    values = [float(m) for m in matches if m]
                    if values:
                        if metric_name not in analysis["metrics"]:
                            analysis["metrics"][metric_name] = {
                                "first": values[0],
                                "last": values[-1],
                                "min": min(values),
                                "max": max(values),
                                "count": len(values),
                            }

            # Find errors
            for i, line in enumerate(lines):
                if re.search(r"\berror\b", line, re.IGNORECASE):
                    analysis["errors"].append(
                        {
                            "file": str(log_file.name),
                            "line": i + 1,
                            "content": line[:200],
                        }
                    )
                if re.search(r"\bwarning\b", line, re.IGNORECASE):
                    analysis["warnings"].append(
                        {
                            "file": str(log_file.name),
                            "line": i + 1,
                            "content": line[:200],
                        }
                    )

            # Custom pattern search
            if pattern:
                for match in re.finditer(pattern, content):
                    analysis["pattern_matches"].append(
                        {
                            "file": str(log_file.name),
                            "match": match.group()[:200],
                        }
                    )

        except Exception as e:
            analysis["files_analyzed"].append(
                {
                    "path": str(log_file),
                    "error": str(e),
                }
            )

    # Limit results
    analysis["errors"] = analysis["errors"][:20]
    analysis["warnings"] = analysis["warnings"][:20]
    analysis["pattern_matches"] = analysis["pattern_matches"][:20]

    return analysis


@tool(
    name="find_checkpoint",
    description="Find checkpoint files for experiments. Locate model weights, optimizer states, and training checkpoints.",
    category=ToolCategory.EXPERIMENTS,
    parameters={
        "type": "object",
        "properties": {
            "experiment": {
                "type": "string",
                "description": "Experiment name to find checkpoints for",
            },
            "project": {
                "type": "string",
                "description": "Project name to search in",
            },
            "latest_only": {
                "type": "boolean",
                "description": "Only return the latest checkpoint (default: false)",
            },
        },
        "required": [],
    },
    read_only=True,
    examples=[
        "find_checkpoint(experiment='custom_transformer_001')",
        "find_checkpoint(project='custom_transformer', latest_only=True)",
    ],
)
async def find_checkpoint(
    experiment: Optional[str] = None,
    project: Optional[str] = None,
    latest_only: bool = False,
    repo_root: Optional[Path] = None,
) -> dict:
    """Find checkpoint files.

    Args:
        experiment: Experiment name to find checkpoints for.
        project: Project name to search in.
        latest_only: Only return latest checkpoint.
        repo_root: Repository root path.

    Returns:
        Dictionary with checkpoint information.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    # Search locations
    search_paths = [
        repo_root / "assets" / "outputs" / "checkpoints",
        repo_root / "checkpoints",
    ]

    if project:
        search_paths.append(repo_root / "projects" / project / "checkpoints")
        search_paths.append(repo_root / "projects" / project / "outputs")

    # Checkpoint extensions
    checkpoint_extensions = [".pt", ".pth", ".ckpt", ".safetensors", ".bin"]

    checkpoints = []

    for search_path in search_paths:
        if not search_path.exists():
            continue

        for ext in checkpoint_extensions:
            for ckpt_file in search_path.glob(f"**/*{ext}"):
                # Filter by experiment if specified
                if experiment and experiment.lower() not in ckpt_file.name.lower():
                    continue

                # Get file info
                stat = ckpt_file.stat()
                checkpoints.append(
                    {
                        "name": ckpt_file.name,
                        "path": str(ckpt_file.relative_to(repo_root)),
                        "size_mb": stat.st_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": ext,
                    }
                )

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x["modified"], reverse=True)

    if not checkpoints:
        return {
            "error": "No checkpoints found",
            "searched": [str(p) for p in search_paths if p.exists()],
            "experiment": experiment,
            "project": project,
        }

    if latest_only:
        return {
            "checkpoint": checkpoints[0],
            "total_found": len(checkpoints),
        }

    return {
        "checkpoints": checkpoints[:20],  # Limit to 20
        "total_found": len(checkpoints),
        "experiment": experiment,
        "project": project,
    }


@tool(
    name="compare_runs",
    description="Compare multiple experiment runs side-by-side. Shows metric differences and config changes.",
    category=ToolCategory.EXPERIMENTS,
    parameters={
        "type": "object",
        "properties": {
            "experiments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of experiment names to compare",
            },
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific metrics to compare (optional, defaults to all)",
            },
        },
        "required": ["experiments"],
    },
    read_only=True,
    examples=[
        "compare_runs(experiments=['exp_001', 'exp_002'])",
        "compare_runs(experiments=['exp_001', 'exp_002'], metrics=['perplexity', 'loss'])",
    ],
)
async def compare_runs(
    experiments: list[str],
    metrics: Optional[list[str]] = None,
    experiments_dir: Optional[Path] = None,
) -> dict:
    """Compare multiple experiment runs.

    Args:
        experiments: List of experiment names.
        metrics: Specific metrics to compare.
        experiments_dir: Override experiments directory.

    Returns:
        Dictionary with comparison results.
    """
    if len(experiments) < 2:
        return {"error": "Need at least 2 experiments to compare"}

    try:
        from common.utils.experiment_storage import (
            load_experiment,
            get_experiments_dir,
        )
    except ImportError:
        return {"error": "Experiment storage not available. Is common package installed?"}

    if experiments_dir is None:
        experiments_dir = get_experiments_dir()

    # Load each experiment
    experiment_data = {}
    errors = []

    for exp_name in experiments:
        try:
            df = load_experiment(exp_name, experiments_dir)
            experiment_data[exp_name] = df
        except FileNotFoundError:
            errors.append(f"Experiment not found: {exp_name}")
        except Exception as e:
            errors.append(f"Error loading {exp_name}: {e}")

    if errors:
        return {
            "error": "Failed to load some experiments",
            "details": errors,
            "loaded": list(experiment_data.keys()),
        }

    if len(experiment_data) < 2:
        return {"error": "Need at least 2 valid experiments to compare"}

    # Build comparison
    comparison = {
        "experiments": list(experiment_data.keys()),
        "metrics_comparison": {},
        "row_counts": {},
        "columns": {},
    }

    # Determine which metrics to compare
    all_columns = set()
    for df in experiment_data.values():
        all_columns.update(df.columns)

    # Exclude metadata columns
    exclude_cols = {"experiment_name", "saved_at"}
    metric_cols = [c for c in all_columns if c not in exclude_cols and not c.startswith("meta_")]

    if metrics:
        metric_cols = [c for c in metric_cols if c in metrics]

    comparison["available_metrics"] = metric_cols

    # Compare each metric
    for metric in metric_cols:
        comparison["metrics_comparison"][metric] = {}

        for exp_name, df in experiment_data.items():
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    comparison["metrics_comparison"][metric][exp_name] = {
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "last": float(values.iloc[-1]),
                        "count": len(values),
                    }

    # Row counts
    for exp_name, df in experiment_data.items():
        comparison["row_counts"][exp_name] = len(df)

    # Column differences
    for exp_name, df in experiment_data.items():
        comparison["columns"][exp_name] = list(df.columns)

    # Find winner for each metric
    comparison["best_by_metric"] = {}
    for metric, values in comparison["metrics_comparison"].items():
        if values:
            # Assume lower is better for loss/perplexity, higher for accuracy
            minimize = "loss" in metric.lower() or "perplexity" in metric.lower()

            best_exp = None
            best_val = None

            for exp_name, stats in values.items():
                val = stats["last"]  # Use last value
                if best_val is None:
                    best_exp = exp_name
                    best_val = val
                elif minimize and val < best_val:
                    best_exp = exp_name
                    best_val = val
                elif not minimize and val > best_val:
                    best_exp = exp_name
                    best_val = val

            comparison["best_by_metric"][metric] = {
                "experiment": best_exp,
                "value": best_val,
                "minimize": minimize,
            }

    return comparison
