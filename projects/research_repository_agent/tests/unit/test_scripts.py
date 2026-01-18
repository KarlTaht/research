"""Tests for script tools."""

import pytest

from research_repository.tools.scripts import find_script, list_scripts
from research_repository.indexer import RepoIndexer


class TestFindScript:
    """Tests for find_script tool."""

    @pytest.mark.asyncio
    async def test_find_train_script(self, indexer: RepoIndexer):
        """Test finding training scripts."""
        result = await find_script(task="train a model", indexer=indexer)

        assert "error" not in result
        assert "matches" in result
        assert len(result["matches"]) > 0

        # Should find train.py scripts
        match_names = [m["name"] for m in result["matches"]]
        assert any("train" in name.lower() for name in match_names)

    @pytest.mark.asyncio
    async def test_find_eval_script(self, indexer: RepoIndexer):
        """Test finding evaluation scripts."""
        result = await find_script(task="evaluate model", indexer=indexer)

        assert "error" not in result
        assert "matches" in result

        # Should search for evaluate purposes
        assert (
            "evaluate" in result["searched_purposes"] or "validate" in result["searched_purposes"]
        )

    @pytest.mark.asyncio
    async def test_find_script_in_project(self, indexer: RepoIndexer):
        """Test finding scripts in specific project."""
        result = await find_script(task="train", project="test_project", indexer=indexer)

        assert "error" not in result
        # All matches should be from test_project
        for match in result["matches"]:
            assert match["project"] == "test_project"

    @pytest.mark.asyncio
    async def test_find_download_script(self, indexer: RepoIndexer):
        """Test finding download scripts."""
        result = await find_script(task="download dataset", indexer=indexer)

        assert "error" not in result
        assert "download" in result["searched_purposes"]

    @pytest.mark.asyncio
    async def test_find_script_unknown_task(self, indexer: RepoIndexer):
        """Test finding scripts for unknown task."""
        result = await find_script(task="xyz unknown task", indexer=indexer)

        assert "error" not in result
        # Should still return results with default purposes
        assert "matches" in result

    @pytest.mark.asyncio
    async def test_find_script_returns_example_command(self, indexer: RepoIndexer):
        """Test that matches include example commands."""
        result = await find_script(task="train", indexer=indexer)

        for match in result["matches"]:
            if match.get("example_command"):
                assert "python" in match["example_command"]

    @pytest.mark.asyncio
    async def test_no_indexer_error(self):
        """Test error when indexer not provided."""
        result = await find_script(task="train", indexer=None)

        assert "error" in result


class TestListScripts:
    """Tests for list_scripts tool."""

    @pytest.mark.asyncio
    async def test_list_all_scripts(self, indexer: RepoIndexer):
        """Test listing all scripts."""
        result = await list_scripts(indexer=indexer)

        assert "error" not in result
        assert "scripts" in result
        assert result["total"] > 0
        assert "by_purpose" in result

    @pytest.mark.asyncio
    async def test_list_by_purpose(self, indexer: RepoIndexer):
        """Test listing scripts by purpose."""
        result = await list_scripts(purpose="train", indexer=indexer)

        assert "error" not in result
        for script in result["scripts"]:
            assert script["purpose"] == "train"

    @pytest.mark.asyncio
    async def test_list_by_project(self, indexer: RepoIndexer):
        """Test listing scripts by project."""
        result = await list_scripts(project="test_project", indexer=indexer)

        assert "error" not in result
        for script in result["scripts"]:
            assert script["project"] == "test_project"

    @pytest.mark.asyncio
    async def test_no_indexer_error(self):
        """Test error when indexer not provided."""
        result = await list_scripts(indexer=None)

        assert "error" in result
