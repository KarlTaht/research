"""Tests for config tools."""

from pathlib import Path

import pytest

from research_repository.tools.configs import read_config, compare_configs, list_configs
from research_repository.indexer import RepoIndexer


class TestReadConfig:
    """Tests for read_config tool."""

    @pytest.mark.asyncio
    async def test_read_valid_config(self, mock_repo: Path):
        """Test reading a valid config file."""
        result = await read_config(
            path="projects/test_project/configs/default.yaml", repo_root=mock_repo
        )

        assert "error" not in result
        assert "content" in result
        assert "sections" in result

    @pytest.mark.asyncio
    async def test_read_config_sections(self, mock_repo: Path):
        """Test that config sections are identified."""
        result = await read_config(
            path="projects/test_project/configs/default.yaml", repo_root=mock_repo
        )

        assert "model" in result["sections"]
        assert "training" in result["sections"]

    @pytest.mark.asyncio
    async def test_read_config_specific_section(self, mock_repo: Path):
        """Test reading specific section."""
        result = await read_config(
            path="projects/test_project/configs/default.yaml",
            section="model",
            repo_root=mock_repo,
        )

        assert "error" not in result
        assert result["focused_section"] == "model"
        assert "model" in result["content"]
        assert "training" not in result["content"]

    @pytest.mark.asyncio
    async def test_read_config_invalid_section(self, mock_repo: Path):
        """Test reading invalid section."""
        result = await read_config(
            path="projects/test_project/configs/default.yaml",
            section="nonexistent",
            repo_root=mock_repo,
        )

        assert "error" in result
        assert "available_sections" in result

    @pytest.mark.asyncio
    async def test_read_config_computed_values(self, mock_repo: Path):
        """Test that computed values are included."""
        result = await read_config(
            path="projects/test_project/configs/default.yaml", repo_root=mock_repo
        )

        assert "computed" in result
        # Should compute head_dim from d_model and n_heads
        if result["computed"]:
            assert "head_dim" in result["computed"] or "effective_batch_size" in result["computed"]

    @pytest.mark.asyncio
    async def test_read_config_analysis(self, mock_repo: Path):
        """Test that analysis is included."""
        result = await read_config(
            path="projects/test_project/configs/default.yaml", repo_root=mock_repo
        )

        assert "analysis" in result
        assert "notes" in result["analysis"]
        assert "warnings" in result["analysis"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_config(self, mock_repo: Path):
        """Test reading nonexistent config."""
        result = await read_config(path="nonexistent.yaml", repo_root=mock_repo)

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_read_non_yaml_file(self, mock_repo: Path):
        """Test reading non-YAML file."""
        result = await read_config(path="projects/test_project/train.py", repo_root=mock_repo)

        assert "error" in result
        assert "YAML" in result["error"]


class TestCompareConfigs:
    """Tests for compare_configs tool."""

    @pytest.mark.asyncio
    async def test_compare_same_config(self, mock_repo: Path):
        """Test comparing a config with itself."""
        config_path = "projects/test_project/configs/default.yaml"
        result = await compare_configs(
            config1=config_path, config2=config_path, repo_root=mock_repo
        )

        assert "error" not in result
        assert result["summary"]["different_values"] == 0
        assert result["summary"]["only_in_first"] == 0
        assert result["summary"]["only_in_second"] == 0

    @pytest.mark.asyncio
    async def test_compare_different_configs(self, mock_repo: Path):
        """Test comparing different configs."""
        # Create a second config
        config_dir = mock_repo / "projects" / "test_project" / "configs"
        (config_dir / "large.yaml").write_text(
            """
model:
  d_model: 512
  n_heads: 8
training:
  batch_size: 64
  learning_rate: 0.001
"""
        )

        result = await compare_configs(
            config1="projects/test_project/configs/default.yaml",
            config2="projects/test_project/configs/large.yaml",
            repo_root=mock_repo,
        )

        assert "error" not in result
        assert "differences" in result
        assert result["summary"]["different_values"] > 0

    @pytest.mark.asyncio
    async def test_compare_nonexistent_config(self, mock_repo: Path):
        """Test comparing with nonexistent config."""
        result = await compare_configs(
            config1="projects/test_project/configs/default.yaml",
            config2="nonexistent.yaml",
            repo_root=mock_repo,
        )

        assert "error" in result

    @pytest.mark.asyncio
    async def test_compare_shows_diff_details(self, mock_repo: Path):
        """Test that diff includes value details."""
        # Create a second config
        config_dir = mock_repo / "projects" / "test_project" / "configs"
        (config_dir / "small.yaml").write_text(
            """
model:
  d_model: 128
"""
        )

        result = await compare_configs(
            config1="projects/test_project/configs/default.yaml",
            config2="projects/test_project/configs/small.yaml",
            repo_root=mock_repo,
        )

        assert "error" not in result
        diffs = result["differences"]

        # Should show what's different
        if diffs["different"]:
            for key, values in diffs["different"].items():
                assert "first" in values
                assert "second" in values


class TestListConfigs:
    """Tests for list_configs tool."""

    @pytest.mark.asyncio
    async def test_list_all_configs(self, indexer: RepoIndexer):
        """Test listing all configs."""
        result = await list_configs(indexer=indexer)

        assert "error" not in result
        assert "configs" in result
        assert result["total"] >= 1
        assert "by_project" in result

    @pytest.mark.asyncio
    async def test_list_by_project(self, indexer: RepoIndexer):
        """Test listing configs by project."""
        result = await list_configs(project="test_project", indexer=indexer)

        assert "error" not in result
        for config in result["configs"]:
            assert config["project"] == "test_project"

    @pytest.mark.asyncio
    async def test_config_metadata(self, indexer: RepoIndexer):
        """Test that config metadata is included."""
        result = await list_configs(indexer=indexer)

        for config in result["configs"]:
            assert "name" in config
            assert "path" in config
            assert "project" in config
            assert "has_model" in config
            assert "has_training" in config

    @pytest.mark.asyncio
    async def test_no_indexer_error(self):
        """Test error when indexer not provided."""
        result = await list_configs(indexer=None)

        assert "error" in result
