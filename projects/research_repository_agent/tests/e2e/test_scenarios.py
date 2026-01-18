"""End-to-end test scenarios.

These tests simulate complete user interactions from start to finish.
"""

from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd

from research_repository.indexer import RepoIndexer
from research_repository.memory import SessionMemory
from research_repository.safety import SafetyHooks, AuditLog
from research_repository.tools.explore import explore_repo
from research_repository.tools.projects import list_projects, get_project
from research_repository.tools.scripts import find_script
from research_repository.tools.configs import read_config, compare_configs
from research_repository.tools.experiments import query_experiments, compare_runs
from research_repository.tools.assistant import (
    run_command,
    explain_error,
    generate_train_command,
    suggest_cleanup,
)


@pytest.fixture
def comprehensive_repo(temp_dir: Path) -> Path:
    """Create a comprehensive test repository."""
    # Setup directory structure
    (temp_dir / "common" / "models").mkdir(parents=True)
    (temp_dir / "common" / "training").mkdir(parents=True)
    (temp_dir / "common" / "utils").mkdir(parents=True)

    # Project: grokking
    grok = temp_dir / "projects" / "grokking"
    grok.mkdir(parents=True)
    (grok / "train.py").write_text("# Grokking training script")
    (grok / "evaluate.py").write_text("# Evaluation script")

    grok_configs = grok / "configs"
    grok_configs.mkdir()
    (grok_configs / "modular_addition.yaml").write_text(
        """
model:
  d_model: 128
  n_heads: 4
  n_layers: 2
training:
  batch_size: 512
  learning_rate: 0.001
  max_steps: 100000
data:
  operation: addition
  prime: 97
"""
    )
    (grok_configs / "modular_division.yaml").write_text(
        """
model:
  d_model: 256
  n_heads: 4
  n_layers: 4
training:
  batch_size: 256
  learning_rate: 0.0005
  max_steps: 200000
data:
  operation: division
  prime: 97
"""
    )

    # Experiments
    exp_dir = temp_dir / "experiments"
    exp_dir.mkdir()

    # Grokking experiment - failed
    exp_fail = pd.DataFrame(
        {
            "step": list(range(0, 10000, 1000)),
            "train_loss": [2.5, 2.0, 1.5, 1.0, 0.5, 0.1, 0.01, 0.001, 0.0001, 0.00001],
            "test_accuracy": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            "experiment_name": ["grok_add_v1"] * 10,
            "saved_at": [datetime.now().isoformat()] * 10,
        }
    )
    exp_fail.to_parquet(exp_dir / "grok_add_v1.parquet")

    # Grokking experiment - success (grokking happens)
    exp_success = pd.DataFrame(
        {
            "step": list(range(0, 100000, 10000)),
            "train_loss": [
                2.5,
                1.5,
                0.5,
                0.01,
                0.001,
                0.0001,
                0.00001,
                0.000001,
                0.0000001,
                0.00000001,
            ],
            "test_accuracy": [0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.3, 0.7, 0.95, 0.99],
            "experiment_name": ["grok_add_v2"] * 10,
            "saved_at": [datetime.now().isoformat()] * 10,
        }
    )
    exp_success.to_parquet(exp_dir / "grok_add_v2.parquet")

    # Checkpoints
    ckpt_dir = temp_dir / "checkpoints"
    ckpt_dir.mkdir()
    (ckpt_dir / "grok_add_v1_step_10000.pt").write_bytes(b"x" * 1024 * 100)
    (ckpt_dir / "grok_add_v2_step_100000.pt").write_bytes(b"x" * 1024 * 100)
    (ckpt_dir / "grok_add_v2_best.pt").write_bytes(b"x" * 1024 * 100)

    # Logs
    log_dir = temp_dir / "logs"
    log_dir.mkdir()
    (log_dir / "grok_add_v1.log").write_text(
        """
2024-01-01 10:00:00 - Starting grokking experiment
2024-01-01 10:00:01 - Config: modular_addition.yaml
2024-01-01 10:01:00 - Step 1000: train_loss=2.0, test_acc=0.01
2024-01-01 10:10:00 - Step 10000: train_loss=0.00001, test_acc=0.01
2024-01-01 10:10:01 - WARNING: Model is overfitting, test accuracy not improving
2024-01-01 10:10:02 - Training stopped early - no grokking observed
"""
    )

    (temp_dir / "CLAUDE.md").write_text("# Grokking Research\n\nStudying neural network grokking.")

    return temp_dir


@pytest.fixture
def repo_indexer(comprehensive_repo: Path) -> RepoIndexer:
    """Create indexer for comprehensive repo."""
    indexer = RepoIndexer(comprehensive_repo)
    indexer.refresh()
    return indexer


class TestNewUserOnboarding:
    """Scenario: New user exploring the repository for the first time."""

    @pytest.mark.asyncio
    async def test_new_user_explores_repo(
        self, comprehensive_repo: Path, repo_indexer: RepoIndexer
    ):
        """User arrives and wants to understand the codebase."""
        memory = SessionMemory()

        # User: "What's in this repository?"
        explore_result = await explore_repo(path=".", depth=2, repo_root=comprehensive_repo)

        assert "error" not in explore_result
        await memory.set_context("last_explored", explore_result["path"])

        # User: "What projects are available?"
        projects_result = await list_projects(indexer=repo_indexer)

        assert "error" not in projects_result
        assert projects_result["summary"]["total"] >= 1
        await memory.set_context(
            "available_projects", [p["name"] for p in projects_result["projects"]]
        )

        # User: "Tell me about the grokking project"
        project_result = await get_project(name="grokking", indexer=repo_indexer)

        assert "error" not in project_result
        assert project_result["has_train_script"]
        await memory.set_context("current_project", "grokking")

        # Verify context is maintained
        context = await memory.get_context()
        assert context["current_project"] == "grokking"


class TestResearcherAnalyzingExperiments:
    """Scenario: Researcher analyzing past experiments to understand results."""

    @pytest.mark.asyncio
    async def test_analyze_grokking_experiments(self, comprehensive_repo: Path):
        """Researcher wants to understand why one experiment worked and another didn't."""
        exp_dir = comprehensive_repo / "experiments"

        # Step 1: "What experiments have I run?"
        query_result = await query_experiments(query="all", experiments_dir=exp_dir)

        assert "error" not in query_result
        experiments = query_result["available_experiments"]
        assert len(experiments) >= 2

        # Step 2: "Which one has the best test accuracy?"
        best_result = await query_experiments(
            query="SELECT experiment_name, MAX(test_accuracy) as max_acc FROM experiments GROUP BY experiment_name ORDER BY max_acc DESC",
            experiments_dir=exp_dir,
        )

        assert "error" not in best_result
        # v2 should have higher accuracy
        best_exp = best_result["results"][0]["experiment_name"]
        assert best_exp == "grok_add_v2"

        # Step 3: "Compare v1 and v2"
        compare_result = await compare_runs(
            experiments=["grok_add_v1", "grok_add_v2"],
            experiments_dir=exp_dir,
        )

        assert "error" not in compare_result
        # Should see dramatic difference in test_accuracy
        if "test_accuracy" in compare_result["metrics_comparison"]:
            v1_acc = compare_result["metrics_comparison"]["test_accuracy"]["grok_add_v1"]["last"]
            v2_acc = compare_result["metrics_comparison"]["test_accuracy"]["grok_add_v2"]["last"]
            assert v2_acc > v1_acc


class TestDebuggingFailedExperiment:
    """Scenario: User debugging why an experiment failed."""

    @pytest.mark.asyncio
    async def test_debug_failed_grokking(self, comprehensive_repo: Path, repo_indexer: RepoIndexer):
        """User investigates why grok_add_v1 didn't work."""
        exp_dir = comprehensive_repo / "experiments"

        # Step 1: Query the failed experiment
        query_result = await query_experiments(
            query="SELECT * FROM experiments WHERE experiment_name = 'grok_add_v1'",
            experiments_dir=exp_dir,
        )

        assert "error" not in query_result
        # Notice test_accuracy never improved
        final_acc = query_result["results"][-1]["test_accuracy"]
        assert final_acc < 0.1  # Never grokked

        # Step 2: Read the config used
        config_result = await read_config(
            path="projects/grokking/configs/modular_addition.yaml",
            repo_root=comprehensive_repo,
        )

        assert "error" not in config_result
        # Notice it only ran for 100k steps
        max_steps = config_result["content"]["training"]["max_steps"]
        assert max_steps == 100000

        # Step 3: Compare with successful config
        compare_result = await compare_configs(
            config1="projects/grokking/configs/modular_addition.yaml",
            config2="projects/grokking/configs/modular_division.yaml",
            repo_root=comprehensive_repo,
        )

        assert "error" not in compare_result
        # Division config has more steps and larger model
        diffs = compare_result["differences"]["different"]
        assert len(diffs) > 0


class TestPreparingNewExperiment:
    """Scenario: User preparing to run a new experiment."""

    @pytest.mark.asyncio
    async def test_prepare_new_experiment(
        self, comprehensive_repo: Path, repo_indexer: RepoIndexer
    ):
        """User wants to run a new grokking experiment."""
        safety = SafetyHooks()

        # Step 1: Find the training script
        script_result = await find_script(task="train grokking", indexer=repo_indexer)

        assert "error" not in script_result
        assert len(script_result["matches"]) > 0

        # Step 2: Read the config to understand parameters
        config_result = await read_config(
            path="projects/grokking/configs/modular_addition.yaml",
            repo_root=comprehensive_repo,
        )

        assert "error" not in config_result

        # Step 3: Generate training command with modifications
        command_result = await generate_train_command(
            project="grokking",
            config="configs/modular_addition.yaml",
            overrides={"max_steps": 200000, "learning_rate": 0.0005},
            device="cuda",
            repo_root=comprehensive_repo,
        )

        assert "error" not in command_result
        assert "--max_steps 200000" in command_result["command"]
        assert "--learning_rate 0.0005" in command_result["command"]

        # Step 4: Check if command would require confirmation
        safety_result = safety.pre_tool_use(
            "run_command",
            {"command": command_result["command"], "execute": True},
        )

        from research_repository.safety.hooks import SafetyDecision

        assert safety_result.decision == SafetyDecision.CONFIRM


class TestSafetyAndAudit:
    """Scenario: Testing that safety and audit features work correctly."""

    @pytest.mark.asyncio
    async def test_safety_blocks_dangerous(self, comprehensive_repo: Path):
        """Safety hooks should block dangerous commands."""
        # Try to run dangerous command
        result = await run_command(
            command="rm -rf /",
            repo_root=comprehensive_repo,
        )

        assert "error" in result
        assert "blocked" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_audit_logging(self, temp_dir: Path):
        """Audit log should track all actions."""
        from research_repository.safety.hooks import SafetyDecision

        audit = AuditLog(temp_dir / "audit.json")

        # Log some actions
        audit.log_action("explore_repo", {"path": "."}, SafetyDecision.ALLOW)
        audit.log_action("read_config", {"path": "config.yaml"}, SafetyDecision.ALLOW)
        audit.log_action(
            "run_command",
            {"command": "rm -rf /"},
            SafetyDecision.DENY,
            error="Blocked by safety hooks",
        )

        # Check summary
        summary = audit.get_session_summary()

        assert summary["total_actions"] == 3
        assert summary["errors"] == 1

        # Check we can retrieve by tool
        run_entries = audit.get_entries_by_tool("run_command")
        assert len(run_entries) == 1
        assert run_entries[0].error is not None


class TestCleanupSuggestions:
    """Scenario: User wants to clean up disk space."""

    @pytest.mark.asyncio
    async def test_suggest_and_review_cleanup(self, comprehensive_repo: Path):
        """User reviews cleanup suggestions."""
        # Step 1: Get cleanup suggestions
        cleanup_result = await suggest_cleanup(
            scope="all",
            max_age_days=0,
            min_size_mb=0.01,  # Low threshold to catch test files
            repo_root=comprehensive_repo,
        )

        assert "error" not in cleanup_result
        assert "summary" in cleanup_result

        # Should find some files
        total_files = (
            cleanup_result["summary"]["old_checkpoints_count"]
            + cleanup_result["summary"]["large_files_count"]
        )
        assert total_files > 0

        # Step 2: User can see suggested cleanup commands
        if cleanup_result.get("cleanup_commands"):
            assert all("rm" in cmd for cmd in cleanup_result["cleanup_commands"])


class TestErrorHandling:
    """Scenario: Testing error handling across tools."""

    @pytest.mark.asyncio
    async def test_missing_project_handling(self, repo_indexer: RepoIndexer):
        """Handle requests for nonexistent projects gracefully."""
        result = await get_project(name="nonexistent_project", indexer=repo_indexer)

        assert "error" in result
        # Should provide suggestions if any similar projects exist

    @pytest.mark.asyncio
    async def test_invalid_config_handling(self, comprehensive_repo: Path):
        """Handle invalid config paths gracefully."""
        result = await read_config(
            path="nonexistent/config.yaml",
            repo_root=comprehensive_repo,
        )

        assert "error" in result
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_explain_complex_error(self):
        """Explain a complex multi-line error."""
        complex_error = """
RuntimeError: Error(s) in loading state_dict for TransformerModel:
    size mismatch for embedding.weight: copying a param with shape torch.Size([50257, 768]) from checkpoint, the shape in current model is torch.Size([50257, 512]).
    size mismatch for transformer.layers.0.self_attn.in_proj_weight: copying a param with shape torch.Size([2304, 768]) from checkpoint, the shape in current model is torch.Size([1536, 512]).
"""

        result = await explain_error(
            error=complex_error,
            context="loading checkpoint",
        )

        assert "error" not in result
        assert len(result["suggestions"]) > 0
        # Should recognize this is a checkpoint loading issue
        assert any("checkpoint" in s.lower() for s in result["suggestions"])
