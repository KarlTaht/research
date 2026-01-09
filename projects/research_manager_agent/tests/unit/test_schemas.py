"""Tests for core schemas."""

from datetime import datetime
from pathlib import Path


from research_manager.schemas import Project, Experiment, Script, Config


class TestProject:
    """Tests for Project schema."""

    def test_create_project(self):
        """Test creating a project."""
        project = Project(
            name="test_project",
            path=Path("/tmp/test"),
            type="project",
        )
        assert project.name == "test_project"
        assert project.type == "project"
        assert project.has_train_script is False

    def test_project_with_scripts(self):
        """Test project with scripts."""
        project = Project(
            name="test_project",
            path=Path("/tmp/test"),
            type="project",
            has_train_script=True,
            has_eval_script=True,
            config_files=["default.yaml", "large.yaml"],
        )
        assert project.has_train_script is True
        assert project.has_eval_script is True
        assert len(project.config_files) == 2

    def test_project_to_dict(self):
        """Test converting project to dict."""
        project = Project(
            name="test_project",
            path=Path("/tmp/test"),
            type="project",
            description="Test description",
            last_modified=datetime(2024, 1, 1, 12, 0, 0),
        )
        d = project.to_dict()
        assert d["name"] == "test_project"
        assert d["path"] == "/tmp/test"
        assert d["description"] == "Test description"
        assert d["last_modified"] == "2024-01-01T12:00:00"


class TestExperiment:
    """Tests for Experiment schema."""

    def test_create_experiment(self):
        """Test creating an experiment."""
        exp = Experiment(
            name="exp_001",
            project="test_project",
            metrics={"perplexity": 15.2, "loss": 2.1},
        )
        assert exp.name == "exp_001"
        assert exp.project == "test_project"
        assert exp.metrics["perplexity"] == 15.2

    def test_experiment_to_dict(self):
        """Test converting experiment to dict."""
        exp = Experiment(
            name="exp_001",
            project="test_project",
            config_path=Path("/tmp/config.yaml"),
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
        )
        d = exp.to_dict()
        assert d["name"] == "exp_001"
        assert d["config_path"] == "/tmp/config.yaml"
        assert d["timestamp"] == "2024-01-01T12:00:00"


class TestScript:
    """Tests for Script schema."""

    def test_create_script(self):
        """Test creating a script."""
        script = Script(
            name="train.py",
            path=Path("/tmp/train.py"),
            purpose="train",
            project="test_project",
        )
        assert script.name == "train.py"
        assert script.purpose == "train"

    def test_script_with_arguments(self):
        """Test script with arguments."""
        script = Script(
            name="train.py",
            path=Path("/tmp/train.py"),
            purpose="train",
            arguments=["--config", "--epochs", "--lr"],
            example_command="python train.py --config config.yaml",
        )
        assert len(script.arguments) == 3
        assert "--config" in script.arguments


class TestConfig:
    """Tests for Config schema."""

    def test_create_config(self):
        """Test creating a config."""
        config = Config(
            path=Path("/tmp/config.yaml"),
            project="test_project",
            sections={"model": {"d_model": 256}},
            model_params={"d_model": 256},
        )
        assert config.project == "test_project"
        assert config.model_params["d_model"] == 256

    def test_config_to_dict(self):
        """Test converting config to dict."""
        config = Config(
            path=Path("/tmp/config.yaml"),
            project="test_project",
            sections={"model": {"d_model": 256}, "training": {"batch_size": 32}},
        )
        d = config.to_dict()
        assert d["project"] == "test_project"
        assert "model" in d["sections"]
