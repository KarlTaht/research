"""Repository indexer for discovering projects, scripts, and configs."""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from research_repository.schemas import Config, Project, Script


class RepoIndexer:
    """Indexes the monorepo structure for fast lookups.

    Discovers projects, scripts, and configuration files on startup
    and maintains an in-memory index for quick access.
    """

    def __init__(self, repo_root: Path):
        """Initialize the indexer.

        Args:
            repo_root: Root directory of the monorepo.
        """
        self.repo_root = repo_root.resolve()
        self.projects: dict[str, Project] = {}
        self.scripts: dict[str, Script] = {}
        self.configs: dict[str, Config] = {}
        self._last_indexed: Optional[datetime] = None

    def refresh(self) -> None:
        """Re-index the entire repository."""
        self.projects.clear()
        self.scripts.clear()
        self.configs.clear()

        self._index_projects()
        self._index_scripts()
        self._index_configs()

        self._last_indexed = datetime.now()

    def _index_projects(self) -> None:
        """Discover all projects in the repository."""
        # Look in projects/ and papers/ directories
        for dir_name in ["projects", "papers"]:
            projects_dir = self.repo_root / dir_name
            if not projects_dir.exists():
                continue

            for project_path in projects_dir.iterdir():
                if not project_path.is_dir():
                    continue

                # Skip hidden directories and archive
                if project_path.name.startswith("."):
                    continue

                project_type = "archive" if project_path.name == "archive" else dir_name.rstrip("s")

                # Handle archive subdirectories
                if project_type == "archive":
                    for archived_path in project_path.iterdir():
                        if archived_path.is_dir() and not archived_path.name.startswith("."):
                            project = self._create_project(archived_path, "archive")
                            self.projects[project.name] = project
                else:
                    project = self._create_project(project_path, project_type)
                    self.projects[project.name] = project

    def _create_project(self, path: Path, project_type: str) -> Project:
        """Create a Project from a directory path."""
        # Get description from README or CLAUDE.md
        description = None
        for readme_name in ["README.md", "CLAUDE.md"]:
            readme_path = path / readme_name
            if readme_path.exists():
                try:
                    content = readme_path.read_text()
                    # Extract first paragraph or heading
                    lines = content.strip().split("\n")
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            description = line[:200]
                            break
                        elif line.startswith("# "):
                            description = line[2:].strip()
                            break
                except Exception:
                    pass
                break

        # Check for common scripts
        has_train = (path / "train.py").exists()
        has_eval = (path / "evaluate.py").exists() or (path / "eval.py").exists()

        # Find config files
        config_files = []
        configs_dir = path / "configs"
        if configs_dir.exists():
            config_files = [str(f.relative_to(path)) for f in configs_dir.glob("*.yaml")]
            config_files.extend([str(f.relative_to(path)) for f in configs_dir.glob("*.yml")])

        # Also check for root-level configs
        for config_file in path.glob("*.yaml"):
            if config_file.name not in ["pyproject.yaml"]:
                config_files.append(config_file.name)

        # Get last modified time
        try:
            stat = path.stat()
            last_modified = datetime.fromtimestamp(stat.st_mtime)
        except Exception:
            last_modified = None

        return Project(
            name=path.name,
            path=path,
            type=project_type,
            description=description,
            has_train_script=has_train,
            has_eval_script=has_eval,
            config_files=config_files,
            last_modified=last_modified,
        )

    def _index_scripts(self) -> None:
        """Find and categorize Python scripts in the repository."""
        # Script patterns and their purposes
        script_patterns = {
            r"train.*\.py$": "train",
            r"eval.*\.py$": "evaluate",
            r"validate.*\.py$": "validate",
            r"download.*\.py$": "download",
            r"test_.*\.py$": "test",
            r"analyze.*\.py$": "analyze",
            r"visualize.*\.py$": "visualize",
            r"run.*\.py$": "run",
        }

        # Search in projects/ and tools/
        search_dirs = [
            self.repo_root / "projects",
            self.repo_root / "tools",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for py_file in search_dir.rglob("*.py"):
                # Skip __init__.py and hidden files
                if py_file.name.startswith("_") or py_file.name.startswith("."):
                    continue

                # Skip test directories
                if "test" in py_file.parts and py_file.name.startswith("test_"):
                    continue

                # Determine purpose
                purpose = "utility"
                for pattern, purpose_name in script_patterns.items():
                    if re.match(pattern, py_file.name):
                        purpose = purpose_name
                        break

                # Determine project
                project = None
                try:
                    rel_path = py_file.relative_to(self.repo_root / "projects")
                    project = rel_path.parts[0]
                except ValueError:
                    pass

                # Extract arguments from argparse patterns
                arguments = self._extract_arguments(py_file)

                script = Script(
                    name=py_file.name,
                    path=py_file,
                    purpose=purpose,
                    project=project,
                    arguments=arguments,
                    example_command=self._generate_example_command(py_file, arguments),
                )
                self.scripts[str(py_file.relative_to(self.repo_root))] = script

    def _extract_arguments(self, script_path: Path) -> list[str]:
        """Extract command-line arguments from a script."""
        arguments = []

        try:
            content = script_path.read_text()

            # Look for argparse add_argument calls
            pattern = r"add_argument\(['\"](-+[\w-]+)['\"]"
            matches = re.findall(pattern, content)
            arguments.extend(matches[:10])  # Limit to first 10

        except Exception:
            pass

        return arguments

    def _generate_example_command(self, script_path: Path, arguments: list[str]) -> str:
        """Generate an example command for running the script."""
        rel_path = script_path.relative_to(self.repo_root)

        # Base command
        cmd = f"python {rel_path}"

        # Add common arguments with placeholders
        if "--config" in arguments:
            cmd += " --config <config.yaml>"
        elif "-c" in arguments:
            cmd += " -c <config.yaml>"

        return cmd

    def _index_configs(self) -> None:
        """Parse and index YAML configuration files."""
        # Search in projects/*/configs/
        for project_path in (self.repo_root / "projects").iterdir():
            if not project_path.is_dir():
                continue

            configs_dir = project_path / "configs"
            if not configs_dir.exists():
                continue

            for config_file in configs_dir.glob("*.yaml"):
                self._index_config(config_file, project_path.name)

            for config_file in configs_dir.glob("*.yml"):
                self._index_config(config_file, project_path.name)

    def _index_config(self, config_path: Path, project: str) -> None:
        """Index a single configuration file."""
        try:
            content = yaml.safe_load(config_path.read_text())
            if content is None:
                content = {}

            # Extract common sections
            model_params = content.get("model", content.get("model_config"))
            training_params = content.get("training", content.get("train_config"))
            data_params = content.get("data", content.get("data_config"))

            config = Config(
                path=config_path,
                project=project,
                sections=content,
                model_params=model_params,
                training_params=training_params,
                data_params=data_params,
            )
            self.configs[str(config_path.relative_to(self.repo_root))] = config

        except Exception as e:
            # Skip invalid YAML files
            print(f"Warning: Could not parse config {config_path}: {e}")

    # Query methods

    def get_project(self, name: str) -> Optional[Project]:
        """Get a project by name."""
        return self.projects.get(name)

    def list_projects(self, include_archived: bool = False) -> list[Project]:
        """List all projects."""
        projects = list(self.projects.values())
        if not include_archived:
            projects = [p for p in projects if p.type != "archive"]
        return sorted(projects, key=lambda p: p.name)

    def find_scripts_by_purpose(self, purpose: str) -> list[Script]:
        """Find scripts by their purpose (train, evaluate, etc.)."""
        return [s for s in self.scripts.values() if s.purpose == purpose]

    def find_scripts_by_project(self, project: str) -> list[Script]:
        """Find scripts belonging to a project."""
        return [s for s in self.scripts.values() if s.project == project]

    def get_config(self, path: str) -> Optional[Config]:
        """Get a config by path."""
        return self.configs.get(path)

    def list_configs_for_project(self, project: str) -> list[Config]:
        """List all configs for a project."""
        return [c for c in self.configs.values() if c.project == project]

    def search_configs(self, query: str) -> list[Config]:
        """Search configs by name or content."""
        query_lower = query.lower()
        results = []

        for config in self.configs.values():
            # Match on path
            if query_lower in str(config.path).lower():
                results.append(config)
                continue

            # Match on content
            content_str = str(config.sections).lower()
            if query_lower in content_str:
                results.append(config)

        return results

    def get_stats(self) -> dict:
        """Get indexer statistics."""
        return {
            "projects": len(self.projects),
            "scripts": len(self.scripts),
            "configs": len(self.configs),
            "last_indexed": self._last_indexed.isoformat() if self._last_indexed else None,
        }
