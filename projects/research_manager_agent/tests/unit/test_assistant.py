"""Tests for assistant tools."""

from pathlib import Path

import pytest

from research_manager.tools.assistant import (
    run_command,
    suggest_cleanup,
    explain_error,
    generate_train_command,
)


class TestRunCommand:
    """Tests for run_command tool."""

    @pytest.mark.asyncio
    async def test_run_simple_command(self, temp_dir: Path):
        """Test running a simple command."""
        result = await run_command(
            command="echo 'hello world'",
            repo_root=temp_dir,
        )

        assert "error" not in result or result.get("success", False)
        assert result.get("exit_code") == 0
        assert "hello world" in result.get("stdout", "")

    @pytest.mark.asyncio
    async def test_run_command_with_description(self, temp_dir: Path):
        """Test command with description."""
        result = await run_command(
            command="pwd",
            description="Print working directory",
            repo_root=temp_dir,
        )

        assert result["description"] == "Print working directory"
        assert result.get("success", False)

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, temp_dir: Path):
        """Test dry run mode."""
        result = await run_command(
            command="echo test",
            dry_run=True,
            repo_root=temp_dir,
        )

        assert result["dry_run"] is True
        assert "not executed" in result.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_block_dangerous_command(self, temp_dir: Path):
        """Test that dangerous commands are blocked."""
        result = await run_command(
            command="rm -rf /",
            repo_root=temp_dir,
        )

        assert "error" in result
        assert "blocked" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_block_force_push(self, temp_dir: Path):
        """Test that force push is blocked."""
        result = await run_command(
            command="git push --force origin main",
            repo_root=temp_dir,
        )

        assert "error" in result
        assert "blocked" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_working_directory(self, temp_dir: Path):
        """Test command with working directory."""
        # Create a subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        result = await run_command(
            command="pwd",
            working_dir="subdir",
            repo_root=temp_dir,
        )

        assert result.get("success", False)
        assert "subdir" in result.get("stdout", "")

    @pytest.mark.asyncio
    async def test_invalid_working_directory(self, temp_dir: Path):
        """Test with invalid working directory."""
        result = await run_command(
            command="ls",
            working_dir="nonexistent",
            repo_root=temp_dir,
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_command_failure(self, temp_dir: Path):
        """Test handling command failure."""
        result = await run_command(
            command="exit 1",
            repo_root=temp_dir,
        )

        assert result.get("success") is False
        assert result.get("exit_code") == 1

    @pytest.mark.asyncio
    async def test_command_timeout(self, temp_dir: Path):
        """Test command timeout."""
        result = await run_command(
            command="sleep 10",
            timeout=1,
            repo_root=temp_dir,
        )

        assert "error" in result
        assert "timed out" in result["error"].lower()


class TestSuggestCleanup:
    """Tests for suggest_cleanup tool."""

    @pytest.fixture
    def repo_with_files(self, temp_dir: Path) -> Path:
        """Create a repo with various files for cleanup testing."""
        # Create checkpoint directory with old files
        ckpt_dir = temp_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "old_model.pt").write_bytes(b"x" * 1024 * 1024 * 20)  # 20MB
        (ckpt_dir / "recent_model.pt").write_bytes(b"x" * 1024 * 1024 * 5)  # 5MB

        # Create log directory
        log_dir = temp_dir / "logs"
        log_dir.mkdir(parents=True)
        (log_dir / "train.log").write_text("training log content" * 1000)

        # Create cache files
        cache_dir = temp_dir / "__pycache__"
        cache_dir.mkdir(parents=True)
        (cache_dir / "module.cpython-39.pyc").write_bytes(b"x" * 1024)

        return temp_dir

    @pytest.mark.asyncio
    async def test_suggest_all_cleanup(self, repo_with_files: Path):
        """Test suggesting all cleanup categories."""
        result = await suggest_cleanup(
            scope="all",
            max_age_days=0,  # Consider all files as old
            min_size_mb=1,
            repo_root=repo_with_files,
        )

        assert "error" not in result
        assert "summary" in result
        assert "large_files" in result

    @pytest.mark.asyncio
    async def test_suggest_checkpoints_only(self, repo_with_files: Path):
        """Test suggesting checkpoint cleanup only."""
        result = await suggest_cleanup(
            scope="checkpoints",
            max_age_days=0,
            repo_root=repo_with_files,
        )

        assert "error" not in result
        assert result["summary"]["scope"] == "checkpoints"

    @pytest.mark.asyncio
    async def test_suggest_logs_only(self, repo_with_files: Path):
        """Test suggesting log cleanup only."""
        result = await suggest_cleanup(
            scope="logs",
            max_age_days=0,
            repo_root=repo_with_files,
        )

        assert "error" not in result
        assert result["summary"]["scope"] == "logs"

    @pytest.mark.asyncio
    async def test_suggest_cache_only(self, repo_with_files: Path):
        """Test suggesting cache cleanup only."""
        result = await suggest_cleanup(
            scope="cache",
            repo_root=repo_with_files,
        )

        assert "error" not in result
        assert result["summary"]["scope"] == "cache"

    @pytest.mark.asyncio
    async def test_large_files_detection(self, repo_with_files: Path):
        """Test detection of large files."""
        result = await suggest_cleanup(
            scope="all",
            min_size_mb=10,
            repo_root=repo_with_files,
        )

        assert "large_files" in result
        # Should find the 20MB checkpoint
        if result["large_files"]:
            assert any(f["size_mb"] >= 10 for f in result["large_files"])

    @pytest.mark.asyncio
    async def test_summary_statistics(self, repo_with_files: Path):
        """Test that summary statistics are included."""
        result = await suggest_cleanup(
            scope="all",
            max_age_days=0,
            repo_root=repo_with_files,
        )

        summary = result["summary"]
        assert "old_checkpoints_count" in summary
        assert "large_files_count" in summary
        assert "total_reclaimable_mb" in summary

    @pytest.mark.asyncio
    async def test_empty_repo(self, temp_dir: Path):
        """Test with empty repository."""
        result = await suggest_cleanup(repo_root=temp_dir)

        assert "error" not in result
        assert result["summary"]["old_checkpoints_count"] == 0


class TestExplainError:
    """Tests for explain_error tool."""

    @pytest.mark.asyncio
    async def test_explain_cuda_oom(self):
        """Test explaining CUDA out of memory error."""
        result = await explain_error(
            error="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        )

        assert "error" not in result
        assert result["category"] == "GPU Memory"
        assert "batch" in " ".join(result["suggestions"]).lower()

    @pytest.mark.asyncio
    async def test_explain_module_not_found(self):
        """Test explaining ModuleNotFoundError."""
        result = await explain_error(error="ModuleNotFoundError: No module named 'transformers'")

        assert result["category"] == "Missing Import"
        assert "install" in " ".join(result["suggestions"]).lower()

    @pytest.mark.asyncio
    async def test_explain_file_not_found(self):
        """Test explaining FileNotFoundError."""
        result = await explain_error(
            error="FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'"
        )

        assert result["category"] == "Missing File"
        assert "path" in " ".join(result["suggestions"]).lower()

    @pytest.mark.asyncio
    async def test_explain_key_error(self):
        """Test explaining KeyError."""
        result = await explain_error(error="KeyError: 'learning_rate'")

        assert result["category"] == "Missing Key"

    @pytest.mark.asyncio
    async def test_explain_nan_loss(self):
        """Test explaining NaN in training."""
        result = await explain_error(error="Loss is NaN at step 100")

        assert result["category"] == "NaN in Training"
        assert "learning rate" in " ".join(result["suggestions"]).lower()

    @pytest.mark.asyncio
    async def test_explain_with_context(self):
        """Test explanation with additional context."""
        result = await explain_error(
            error="RuntimeError: Expected tensor for argument",
            context="training transformer model",
        )

        assert result["context"] == "training transformer model"
        # Should have training-specific suggestions
        assert any("train" in s.lower() for s in result["suggestions"])

    @pytest.mark.asyncio
    async def test_explain_unknown_error(self):
        """Test explaining unknown error."""
        result = await explain_error(error="SomeUnknownError: xyz123")

        assert result["category"] == "Unknown"
        assert "search" in " ".join(result["suggestions"]).lower()

    @pytest.mark.asyncio
    async def test_traceback_parsing(self):
        """Test parsing traceback for file locations."""
        traceback = """Traceback (most recent call last):
  File "train.py", line 42, in main
    model.forward(x)
  File "model.py", line 100, in forward
    return self.layer(x)
RuntimeError: size mismatch"""

        result = await explain_error(error=traceback)

        assert "traceback_locations" in result
        assert len(result["traceback_locations"]) >= 2
        assert result["traceback_locations"][0]["file"] == "train.py"
        assert result["traceback_locations"][0]["line"] == 42

    @pytest.mark.asyncio
    async def test_related_docs(self):
        """Test that related docs are included for CUDA errors."""
        result = await explain_error(error="CUDA error: device-side assert triggered")

        assert "related_docs" in result
        if result["related_docs"]:
            assert any("pytorch" in d["url"].lower() for d in result["related_docs"])


class TestGenerateTrainCommand:
    """Tests for generate_train_command tool."""

    @pytest.fixture
    def project_setup(self, temp_dir: Path) -> Path:
        """Create a project with train.py and config."""
        project_dir = temp_dir / "projects" / "test_project"
        project_dir.mkdir(parents=True)

        # Create train.py
        (project_dir / "train.py").write_text(
            """
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--device')
parser.add_argument('--resume')
"""
        )

        # Create config directory
        config_dir = project_dir / "configs"
        config_dir.mkdir()
        (config_dir / "default.yaml").write_text("model:\n  d_model: 256")
        (config_dir / "large.yaml").write_text("model:\n  d_model: 512")

        return temp_dir

    @pytest.mark.asyncio
    async def test_generate_basic_command(self, project_setup: Path):
        """Test generating basic training command."""
        result = await generate_train_command(
            project="test_project",
            repo_root=project_setup,
        )

        assert "error" not in result
        assert "command" in result
        assert "train.py" in result["command"]
        assert "test_project" in result["command"]

    @pytest.mark.asyncio
    async def test_generate_with_config(self, project_setup: Path):
        """Test generating command with specific config."""
        result = await generate_train_command(
            project="test_project",
            config="configs/large.yaml",
            repo_root=project_setup,
        )

        assert "error" not in result
        assert "--config" in result["command"]
        assert "large.yaml" in result["command"]

    @pytest.mark.asyncio
    async def test_generate_with_device(self, project_setup: Path):
        """Test generating command with device specification."""
        result = await generate_train_command(
            project="test_project",
            device="cuda",
            repo_root=project_setup,
        )

        assert "error" not in result
        assert "--device cuda" in result["command"]

    @pytest.mark.asyncio
    async def test_generate_with_resume(self, project_setup: Path):
        """Test generating command with checkpoint resume."""
        result = await generate_train_command(
            project="test_project",
            resume="checkpoints/model.pt",
            repo_root=project_setup,
        )

        assert "error" not in result
        assert "--resume" in result["command"]

    @pytest.mark.asyncio
    async def test_generate_with_overrides(self, project_setup: Path):
        """Test generating command with config overrides."""
        result = await generate_train_command(
            project="test_project",
            overrides={"batch_size": 64, "lr": 0.001},
            repo_root=project_setup,
        )

        assert "error" not in result
        assert "--batch_size" in result["command"]
        assert "--lr" in result["command"]

    @pytest.mark.asyncio
    async def test_nonexistent_project(self, temp_dir: Path):
        """Test with nonexistent project."""
        result = await generate_train_command(
            project="nonexistent",
            repo_root=temp_dir,
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_project_without_train_script(self, temp_dir: Path):
        """Test project without train.py."""
        project_dir = temp_dir / "projects" / "no_train"
        project_dir.mkdir(parents=True)

        result = await generate_train_command(
            project="no_train",
            repo_root=temp_dir,
        )

        assert "error" in result
        assert "train.py" in result["error"]

    @pytest.mark.asyncio
    async def test_includes_notes(self, project_setup: Path):
        """Test that helpful notes are included."""
        result = await generate_train_command(
            project="test_project",
            repo_root=project_setup,
        )

        assert "notes" in result
        assert len(result["notes"]) > 0
