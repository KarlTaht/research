"""Tests for the repository indexer."""

from pathlib import Path


from research_manager.indexer import RepoIndexer


class TestRepoIndexer:
    """Tests for RepoIndexer."""

    def test_index_projects(self, indexer: RepoIndexer):
        """Test that projects are discovered."""
        projects = indexer.list_projects(include_archived=False)

        # Should find test_project and another_project, but not archived
        project_names = [p.name for p in projects]
        assert "test_project" in project_names
        assert "another_project" in project_names

    def test_index_archived_projects(self, indexer: RepoIndexer):
        """Test that archived projects are discovered when requested."""
        projects = indexer.list_projects(include_archived=True)
        project_names = [p.name for p in projects]
        assert "old_project" in project_names

    def test_project_has_scripts(self, indexer: RepoIndexer):
        """Test that project scripts are detected."""
        project = indexer.get_project("test_project")
        assert project is not None
        assert project.has_train_script is True
        assert project.has_eval_script is True

    def test_project_description(self, indexer: RepoIndexer):
        """Test that project description is extracted from README."""
        project = indexer.get_project("test_project")
        assert project is not None
        assert project.description is not None
        assert "test" in project.description.lower()

    def test_index_scripts(self, indexer: RepoIndexer):
        """Test that scripts are discovered."""
        # Find training scripts
        train_scripts = indexer.find_scripts_by_purpose("train")
        assert len(train_scripts) >= 2  # test_project and another_project

    def test_find_scripts_by_project(self, indexer: RepoIndexer):
        """Test finding scripts by project."""
        scripts = indexer.find_scripts_by_project("test_project")
        script_names = [s.name for s in scripts]
        assert "train.py" in script_names
        assert "evaluate.py" in script_names

    def test_script_arguments(self, indexer: RepoIndexer):
        """Test that script arguments are extracted."""
        scripts = indexer.find_scripts_by_project("test_project")
        train_script = next(s for s in scripts if s.name == "train.py")
        assert "--config" in train_script.arguments

    def test_index_configs(self, indexer: RepoIndexer):
        """Test that configs are discovered."""
        configs = indexer.list_configs_for_project("test_project")
        assert len(configs) == 1
        assert configs[0].path.name == "default.yaml"

    def test_config_sections(self, indexer: RepoIndexer):
        """Test that config sections are parsed."""
        configs = indexer.list_configs_for_project("test_project")
        config = configs[0]

        assert config.model_params is not None
        assert config.model_params["d_model"] == 256

        assert config.training_params is not None
        assert config.training_params["batch_size"] == 32

    def test_search_configs(self, indexer: RepoIndexer):
        """Test searching configs."""
        results = indexer.search_configs("d_model")
        assert len(results) >= 1

    def test_get_stats(self, indexer: RepoIndexer):
        """Test getting indexer stats."""
        stats = indexer.get_stats()

        assert stats["projects"] >= 2
        assert stats["scripts"] >= 3
        assert stats["configs"] >= 1
        assert stats["last_indexed"] is not None

    def test_refresh(self, mock_repo: Path):
        """Test refreshing the index."""
        indexer = RepoIndexer(mock_repo)
        indexer.refresh()

        # Add a new project
        new_project = mock_repo / "projects" / "new_project"
        new_project.mkdir()
        (new_project / "train.py").write_text("# New training script")

        # Before refresh, shouldn't find new project
        assert indexer.get_project("new_project") is None

        # After refresh, should find it
        indexer.refresh()
        assert indexer.get_project("new_project") is not None
