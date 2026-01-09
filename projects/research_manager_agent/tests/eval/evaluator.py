"""Agent evaluator for testing scenarios.

This module provides tools for evaluating agent behavior across
predefined scenarios and tracking results over time.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class EvalScenario:
    """A single evaluation scenario."""

    name: str
    description: str
    input: str
    expected_tools: list[str] = field(default_factory=list)
    expected_outputs: list[dict] = field(default_factory=list)
    success_criteria: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result of evaluating a single scenario."""

    scenario_name: str
    passed: bool
    tools_used: list[str] = field(default_factory=list)
    tools_correct: bool = False
    outputs_correct: bool = False
    criteria_met: dict = field(default_factory=dict)
    response: str = ""
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "passed": self.passed,
            "tools_used": self.tools_used,
            "tools_correct": self.tools_correct,
            "outputs_correct": self.outputs_correct,
            "criteria_met": self.criteria_met,
            "response": self.response[:500],  # Truncate
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp.isoformat(),
        }


class AgentEvaluator:
    """Evaluator for running agent scenarios.

    Loads scenarios from YAML files and runs them against an agent,
    checking for expected tool usage and outputs.
    """

    def __init__(self, scenarios_path: Optional[Path] = None):
        """Initialize evaluator.

        Args:
            scenarios_path: Path to scenarios YAML file.
        """
        self.scenarios: list[EvalScenario] = []
        if scenarios_path and scenarios_path.exists():
            self.load_scenarios(scenarios_path)

    def load_scenarios(self, path: Path) -> None:
        """Load scenarios from YAML file.

        Args:
            path: Path to scenarios file.
        """
        content = yaml.safe_load(path.read_text())

        for scenario_data in content.get("scenarios", []):
            scenario = EvalScenario(
                name=scenario_data["name"],
                description=scenario_data.get("description", ""),
                input=scenario_data["input"],
                expected_tools=scenario_data.get("expected_tools", []),
                expected_outputs=scenario_data.get("expected_outputs", []),
                success_criteria=scenario_data.get("success_criteria", {}),
                tags=scenario_data.get("tags", []),
            )
            self.scenarios.append(scenario)

    def add_scenario(self, scenario: EvalScenario) -> None:
        """Add a scenario programmatically.

        Args:
            scenario: Scenario to add.
        """
        self.scenarios.append(scenario)

    async def evaluate_scenario(
        self,
        scenario: EvalScenario,
        agent: Any,
    ) -> EvalResult:
        """Evaluate a single scenario.

        Args:
            scenario: Scenario to evaluate.
            agent: Agent to test (must have chat() method).

        Returns:
            Evaluation result.
        """
        start_time = datetime.now()

        result = EvalResult(
            scenario_name=scenario.name,
            passed=False,
        )

        try:
            # Run the scenario
            response = await agent.chat(scenario.input)
            result.response = response

            # Get tools used
            if hasattr(agent, "history") and agent.history:
                last_turn = agent.history[-1]
                result.tools_used = [tc.name for tc in last_turn.tool_calls]

            # Check tool usage
            if scenario.expected_tools:
                result.tools_correct = all(
                    tool in result.tools_used for tool in scenario.expected_tools
                )
            else:
                result.tools_correct = True

            # Check outputs
            result.outputs_correct = True
            for expected in scenario.expected_outputs:
                if "contains" in expected:
                    if expected["contains"].lower() not in response.lower():
                        result.outputs_correct = False
                        break
                if "not_contains" in expected:
                    if expected["not_contains"].lower() in response.lower():
                        result.outputs_correct = False
                        break

            # Check custom criteria
            for criterion_name, criterion_value in scenario.success_criteria.items():
                result.criteria_met[criterion_name] = self._check_criterion(
                    criterion_name, criterion_value, response, result.tools_used
                )

            # Determine overall pass/fail
            result.passed = (
                result.tools_correct
                and result.outputs_correct
                and all(result.criteria_met.values())
            )

        except Exception as e:
            result.error = str(e)
            result.passed = False

        result.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        return result

    def _check_criterion(
        self,
        name: str,
        value: Any,
        response: str,
        tools_used: list[str],
    ) -> bool:
        """Check a single success criterion.

        Args:
            name: Criterion name.
            value: Expected value.
            response: Agent response.
            tools_used: Tools that were used.

        Returns:
            Whether criterion is met.
        """
        if name == "returns_valid_experiments":
            return "exp" in response.lower() or "experiment" in response.lower()

        if name == "explains_results":
            return len(response) > 50

        if name == "command_is_valid":
            return "python" in response.lower() or "train" in response.lower()

        if name == "blocked":
            return "blocked" in response.lower() or "denied" in response.lower()

        if name == "reason_given":
            return len(response) > 20

        # Default: check if value appears in response
        if isinstance(value, bool):
            return value

        return str(value).lower() in response.lower()

    async def run_evaluation(self, agent: Any) -> dict:
        """Run all scenarios and collect metrics.

        Args:
            agent: Agent to evaluate.

        Returns:
            Evaluation summary with results.
        """
        results = []

        for scenario in self.scenarios:
            result = await self.evaluate_scenario(scenario, agent)
            results.append(result)

        # Calculate summary
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        return {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(results) if results else 0,
            "by_scenario": [r.to_dict() for r in results],
            "timestamp": datetime.now().isoformat(),
        }

    async def run_tagged(self, agent: Any, tag: str) -> dict:
        """Run scenarios with a specific tag.

        Args:
            agent: Agent to evaluate.
            tag: Tag to filter by.

        Returns:
            Evaluation summary.
        """
        original_scenarios = self.scenarios
        self.scenarios = [s for s in self.scenarios if tag in s.tags]

        try:
            return await self.run_evaluation(agent)
        finally:
            self.scenarios = original_scenarios


# Default scenarios for quick testing
DEFAULT_SCENARIOS = [
    EvalScenario(
        name="list_projects",
        description="User asks what projects exist",
        input="What projects are in this repository?",
        expected_tools=["list_projects"],
        expected_outputs=[{"contains": "project"}],
        success_criteria={"returns_valid_projects": True},
        tags=["codebase", "basic"],
    ),
    EvalScenario(
        name="find_training_script",
        description="User wants to train a model",
        input="How do I train a model in this repo?",
        expected_tools=["find_script"],
        expected_outputs=[{"contains": "train"}],
        success_criteria={"command_is_valid": True},
        tags=["assistant", "basic"],
    ),
    EvalScenario(
        name="query_experiments",
        description="User asks about experiments",
        input="What experiments have I run?",
        expected_tools=["query_experiments"],
        success_criteria={"returns_valid_experiments": True},
        tags=["experiments", "basic"],
    ),
    EvalScenario(
        name="explain_error",
        description="User has CUDA OOM error",
        input="I got CUDA out of memory error, what should I do?",
        expected_tools=["explain_error"],
        expected_outputs=[{"contains": "batch"}],
        success_criteria={"explains_results": True},
        tags=["assistant", "errors"],
    ),
]
