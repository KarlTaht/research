"""Tests for explore_repo tool."""

from pathlib import Path

import pytest

from research_manager.tools.explore import explore_repo


class TestExploreRepo:
    """Tests for explore_repo tool."""

    @pytest.mark.asyncio
    async def test_explore_root(self, mock_repo: Path):
        """Test exploring repo root."""
        result = await explore_repo(path=".", repo_root=mock_repo)

        assert "error" not in result
        assert result["path"] == "."
        assert "contents" in result

    @pytest.mark.asyncio
    async def test_explore_project(self, mock_repo: Path):
        """Test exploring a specific project."""
        result = await explore_repo(path="projects/test_project", repo_root=mock_repo)

        assert "error" not in result
        assert result["type"] == "project"
        assert "contents" in result

        # Should find key files
        key_files = result["contents"]["key_files"]
        key_file_names = [f["name"] for f in key_files]
        assert "train.py" in key_file_names or any(
            "train.py" in f["name"] for f in result["contents"]["files"]
        )

    @pytest.mark.asyncio
    async def test_explore_with_depth(self, mock_repo: Path):
        """Test depth limiting."""
        result = await explore_repo(path=".", depth=1, repo_root=mock_repo)

        assert "error" not in result
        # At depth 1, subdirectories should not have nested contents
        for dir_info in result["contents"]["directories"]:
            if "contents" in dir_info:
                # Should be truncated or shallow
                assert dir_info["contents"].get("truncated", False) or not dir_info["contents"].get(
                    "directories"
                )

    @pytest.mark.asyncio
    async def test_explore_nonexistent_path(self, mock_repo: Path):
        """Test exploring nonexistent path."""
        result = await explore_repo(path="nonexistent", repo_root=mock_repo)

        assert "error" in result
        assert "does not exist" in result["error"]

    @pytest.mark.asyncio
    async def test_explore_outside_repo(self, mock_repo: Path):
        """Test exploring path outside repo."""
        result = await explore_repo(path="../../../etc", repo_root=mock_repo)

        assert "error" in result
        assert "outside" in result["error"]

    @pytest.mark.asyncio
    async def test_explore_includes_description(self, mock_repo: Path):
        """Test that description is extracted from README."""
        result = await explore_repo(path="projects/test_project", repo_root=mock_repo)

        assert "error" not in result
        # Should have description from README.md
        assert "description" in result or result.get("doc_file") is not None

    @pytest.mark.asyncio
    async def test_explore_git_status(self, mock_repo: Path):
        """Test that git status is included."""
        result = await explore_repo(path=".", repo_root=mock_repo)

        assert "git_status" in result
        # May be "not_in_git", "clean", "dirty", or "unknown"
        assert result["git_status"]["status"] in {"not_in_git", "clean", "dirty", "unknown"}

    @pytest.mark.asyncio
    async def test_explore_hidden_files(self, mock_repo: Path):
        """Test that hidden files are excluded by default."""
        # Create a hidden file
        (mock_repo / ".hidden_file").write_text("hidden")

        result = await explore_repo(path=".", show_hidden=False, repo_root=mock_repo)

        # Hidden files should not appear
        all_files = result["contents"]["files"] + result["contents"]["key_files"]
        file_names = [f["name"] for f in all_files]
        assert ".hidden_file" not in file_names

    @pytest.mark.asyncio
    async def test_explore_show_hidden(self, mock_repo: Path):
        """Test that hidden files are shown when requested."""
        # Create a hidden file
        (mock_repo / ".hidden_file").write_text("hidden")

        result = await explore_repo(path=".", show_hidden=True, repo_root=mock_repo)

        # Hidden files should appear (though .git is still filtered)
        all_files = result["contents"]["files"] + result["contents"]["key_files"]
        file_names = [f["name"] for f in all_files]
        assert ".hidden_file" in file_names

    @pytest.mark.asyncio
    async def test_classify_directory_types(self, mock_repo: Path):
        """Test that directory types are classified correctly."""
        result = await explore_repo(path=".", depth=2, repo_root=mock_repo)

        dir_types = {d["name"]: d["type"] for d in result["contents"]["directories"]}

        # projects/test_project should be "project" type
        if "projects" in dir_types:
            projects_result = await explore_repo(path="projects", repo_root=mock_repo)
            for d in projects_result["contents"]["directories"]:
                if d["name"] == "test_project":
                    assert d["type"] == "project"
