"""Regression tracking for agent evaluation.

This module provides tools for tracking evaluation results over time
and detecting regressions in agent behavior.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class RegressionMetrics:
    """Metrics tracked for regression detection."""

    pass_rate: float
    total_scenarios: int
    passed_scenarios: int
    failed_scenarios: int
    avg_execution_time_ms: float
    tool_accuracy: float  # % of scenarios using correct tools
    timestamp: datetime = field(default_factory=datetime.now)
    version: str = ""
    git_commit: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pass_rate": self.pass_rate,
            "total_scenarios": self.total_scenarios,
            "passed_scenarios": self.passed_scenarios,
            "failed_scenarios": self.failed_scenarios,
            "avg_execution_time_ms": self.avg_execution_time_ms,
            "tool_accuracy": self.tool_accuracy,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "git_commit": self.git_commit,
        }


class RegressionTracker:
    """Tracks evaluation results over time for regression detection.

    Stores results in JSON format and can save to experiment storage
    for integration with the broader ML experiment tracking system.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize tracker.

        Args:
            storage_path: Path to store regression data.
        """
        self.storage_path = storage_path
        self.history: list[RegressionMetrics] = []

        if storage_path and storage_path.exists():
            self._load()

    def _load(self) -> None:
        """Load history from storage."""
        if not self.storage_path:
            return

        try:
            data = json.loads(self.storage_path.read_text())
            for entry in data.get("history", []):
                metrics = RegressionMetrics(
                    pass_rate=entry["pass_rate"],
                    total_scenarios=entry["total_scenarios"],
                    passed_scenarios=entry["passed_scenarios"],
                    failed_scenarios=entry["failed_scenarios"],
                    avg_execution_time_ms=entry["avg_execution_time_ms"],
                    tool_accuracy=entry["tool_accuracy"],
                    timestamp=datetime.fromisoformat(entry["timestamp"]),
                    version=entry.get("version", ""),
                    git_commit=entry.get("git_commit", ""),
                )
                self.history.append(metrics)
        except (json.JSONDecodeError, KeyError):
            pass

    def _save(self) -> None:
        """Save history to storage."""
        if not self.storage_path:
            return

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"history": [m.to_dict() for m in self.history]}
        self.storage_path.write_text(json.dumps(data, indent=2))

    def record_evaluation(
        self,
        eval_results: dict,
        version: str = "",
        git_commit: str = "",
    ) -> RegressionMetrics:
        """Record an evaluation run.

        Args:
            eval_results: Results from AgentEvaluator.run_evaluation().
            version: Version string for this run.
            git_commit: Git commit hash.

        Returns:
            Metrics for this evaluation.
        """
        # Calculate metrics
        by_scenario = eval_results.get("by_scenario", [])

        tool_correct_count = sum(1 for s in by_scenario if s.get("tools_correct", False))
        tool_accuracy = tool_correct_count / len(by_scenario) if by_scenario else 0

        avg_time = (
            sum(s.get("execution_time_ms", 0) for s in by_scenario) / len(by_scenario)
            if by_scenario
            else 0
        )

        metrics = RegressionMetrics(
            pass_rate=eval_results.get("pass_rate", 0),
            total_scenarios=eval_results.get("total", 0),
            passed_scenarios=eval_results.get("passed", 0),
            failed_scenarios=eval_results.get("failed", 0),
            avg_execution_time_ms=avg_time,
            tool_accuracy=tool_accuracy,
            version=version,
            git_commit=git_commit,
        )

        self.history.append(metrics)
        self._save()

        return metrics

    def get_baseline(self, version: Optional[str] = None) -> Optional[RegressionMetrics]:
        """Get baseline metrics for comparison.

        Args:
            version: Specific version to get. If None, returns most recent.

        Returns:
            Baseline metrics or None if no history.
        """
        if not self.history:
            return None

        if version:
            for metrics in reversed(self.history):
                if metrics.version == version:
                    return metrics
            return None

        return self.history[-1]

    def check_regression(
        self,
        current: RegressionMetrics,
        baseline: Optional[RegressionMetrics] = None,
        pass_rate_threshold: float = 0.05,
        time_threshold_pct: float = 0.20,
    ) -> dict:
        """Check for regressions against baseline.

        Args:
            current: Current evaluation metrics.
            baseline: Baseline to compare against. If None, uses most recent.
            pass_rate_threshold: Max allowed decrease in pass rate.
            time_threshold_pct: Max allowed increase in execution time (%).

        Returns:
            Dictionary with regression analysis.
        """
        if baseline is None:
            baseline = self.get_baseline()

        if baseline is None:
            return {
                "has_baseline": False,
                "regressions": [],
                "improvements": [],
            }

        regressions = []
        improvements = []

        # Check pass rate
        pass_rate_change = current.pass_rate - baseline.pass_rate
        if pass_rate_change < -pass_rate_threshold:
            regressions.append(
                {
                    "metric": "pass_rate",
                    "baseline": baseline.pass_rate,
                    "current": current.pass_rate,
                    "change": pass_rate_change,
                }
            )
        elif pass_rate_change > pass_rate_threshold:
            improvements.append(
                {
                    "metric": "pass_rate",
                    "baseline": baseline.pass_rate,
                    "current": current.pass_rate,
                    "change": pass_rate_change,
                }
            )

        # Check tool accuracy
        tool_acc_change = current.tool_accuracy - baseline.tool_accuracy
        if tool_acc_change < -pass_rate_threshold:
            regressions.append(
                {
                    "metric": "tool_accuracy",
                    "baseline": baseline.tool_accuracy,
                    "current": current.tool_accuracy,
                    "change": tool_acc_change,
                }
            )
        elif tool_acc_change > pass_rate_threshold:
            improvements.append(
                {
                    "metric": "tool_accuracy",
                    "baseline": baseline.tool_accuracy,
                    "current": current.tool_accuracy,
                    "change": tool_acc_change,
                }
            )

        # Check execution time
        if baseline.avg_execution_time_ms > 0:
            time_change_pct = (
                current.avg_execution_time_ms - baseline.avg_execution_time_ms
            ) / baseline.avg_execution_time_ms
            if time_change_pct > time_threshold_pct:
                regressions.append(
                    {
                        "metric": "execution_time",
                        "baseline": baseline.avg_execution_time_ms,
                        "current": current.avg_execution_time_ms,
                        "change_pct": time_change_pct,
                    }
                )
            elif time_change_pct < -time_threshold_pct:
                improvements.append(
                    {
                        "metric": "execution_time",
                        "baseline": baseline.avg_execution_time_ms,
                        "current": current.avg_execution_time_ms,
                        "change_pct": time_change_pct,
                    }
                )

        return {
            "has_baseline": True,
            "baseline_version": baseline.version,
            "current_version": current.version,
            "regressions": regressions,
            "improvements": improvements,
            "is_regression": len(regressions) > 0,
            "is_improvement": len(improvements) > 0 and len(regressions) == 0,
        }

    def get_trend(self, metric: str = "pass_rate", last_n: int = 10) -> dict:
        """Get trend for a metric over recent evaluations.

        Args:
            metric: Metric to analyze (pass_rate, tool_accuracy, etc.)
            last_n: Number of recent evaluations to consider.

        Returns:
            Trend analysis.
        """
        if len(self.history) < 2:
            return {"trend": "insufficient_data", "values": []}

        recent = self.history[-last_n:]
        values = [getattr(m, metric, 0) for m in recent]

        # Calculate trend
        if len(values) >= 2:
            first_half = sum(values[: len(values) // 2]) / (len(values) // 2)
            second_half = sum(values[len(values) // 2 :]) / (len(values) - len(values) // 2)

            if second_half > first_half * 1.05:
                trend = "improving"
            elif second_half < first_half * 0.95:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {
            "metric": metric,
            "trend": trend,
            "values": values,
            "timestamps": [m.timestamp.isoformat() for m in recent],
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "avg": sum(values) / len(values) if values else 0,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history to DataFrame for analysis.

        Returns:
            DataFrame with all historical metrics.
        """
        if not self.history:
            return pd.DataFrame()

        return pd.DataFrame([m.to_dict() for m in self.history])

    def save_to_experiments(self, experiment_name: str = "agent_eval") -> None:
        """Save regression data to experiment storage.

        Args:
            experiment_name: Name for the experiment.
        """
        try:
            from common.utils.experiment_storage import save_experiment

            df = self.to_dataframe()
            if not df.empty:
                save_experiment(
                    experiment_name,
                    df,
                    metadata={
                        "type": "agent_evaluation",
                        "total_runs": len(self.history),
                    },
                )
        except ImportError:
            pass  # experiment_storage not available
