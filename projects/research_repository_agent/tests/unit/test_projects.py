"""Tests for project tools."""

import pytest

from research_repository.tools.projects import list_projects, get_project
from research_repository.indexer import RepoIndexer


class TestListProjects:
    """Tests for list_projects tool."""

    @pytest.mark.asyncio
    async def test_list_all_projects(self, indexer: RepoIndexer):
        """Test listing all projects."""
        result = await list_projects(indexer=indexer)

        assert "error" not in result
        assert "projects" in result
        assert "summary" in result
        assert result["summary"]["total"] >= 2  # test_project and another_project

    @pytest.mark.asyncio
    async def test_list_excludes_archived_by_default(self, indexer: RepoIndexer):
        """Test that archived projects are excluded by default."""
        result = await list_projects(include_archived=False, indexer=indexer)

        project_names = [p["name"] for p in result["projects"]]
        assert "old_project" not in project_names

    @pytest.mark.asyncio
    async def test_list_includes_archived(self, indexer: RepoIndexer):
        """Test including archived projects."""
        result = await list_projects(include_archived=True, indexer=indexer)

        project_names = [p["name"] for p in result["projects"]]
        assert "old_project" in project_names

    @pytest.mark.asyncio
    async def test_filter_by_train_script(self, indexer: RepoIndexer):
        """Test filtering by train script presence."""
        result = await list_projects(has_train_script=True, indexer=indexer)

        for project in result["projects"]:
            assert "train.py" in project["available_scripts"]

    @pytest.mark.asyncio
    async def test_summary_stats(self, indexer: RepoIndexer):
        """Test that summary stats are correct."""
        result = await list_projects(indexer=indexer)

        summary = result["summary"]
        assert summary["total"] == len(result["projects"])
        assert "with_train_script" in summary
        assert "with_configs" in summary
        assert "types" in summary

    @pytest.mark.asyncio
    async def test_no_indexer_error(self):
        """Test error when indexer not provided."""
        result = await list_projects(indexer=None)

        assert "error" in result


class TestGetProject:
    """Tests for get_project tool."""

    @pytest.mark.asyncio
    async def test_get_existing_project(self, indexer: RepoIndexer):
        """Test getting an existing project."""
        result = await get_project(name="test_project", indexer=indexer)

        assert "error" not in result
        assert result["name"] == "test_project"
        assert "scripts" in result
        assert "configs" in result

    @pytest.mark.asyncio
    async def test_get_project_with_scripts(self, indexer: RepoIndexer):
        """Test that project scripts are included."""
        result = await get_project(name="test_project", indexer=indexer)

        assert "scripts" in result
        script_names = [s["name"] for s in result["scripts"]]
        assert "train.py" in script_names

    @pytest.mark.asyncio
    async def test_get_project_with_configs(self, indexer: RepoIndexer):
        """Test that project configs are included."""
        result = await get_project(name="test_project", indexer=indexer)

        assert "configs" in result
        assert len(result["configs"]) >= 1

    @pytest.mark.asyncio
    async def test_get_nonexistent_project(self, indexer: RepoIndexer):
        """Test getting a nonexistent project."""
        result = await get_project(name="nonexistent_project", indexer=indexer)

        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_get_project_suggestions(self, indexer: RepoIndexer):
        """Test that similar project names are suggested."""
        result = await get_project(name="test", indexer=indexer)

        assert "error" in result
        # Should suggest test_project
        if "suggestions" in result and result["suggestions"]:
            assert any("test" in s.lower() for s in result["suggestions"])

    @pytest.mark.asyncio
    async def test_no_indexer_error(self):
        """Test error when indexer not provided."""
        result = await get_project(name="test_project", indexer=None)

        assert "error" in result
