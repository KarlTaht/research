"""Main entry point for the research manager agent."""

import asyncio
from pathlib import Path
from typing import Optional

from research_manager.indexer import RepoIndexer
from research_manager.memory import PersistentMemory, SessionMemory
from research_manager.safety import SafetyHooks, AuditLog
from research_manager.tools import get_global_registry
from research_manager.tools.loader import load_tools
from research_manager.ui import TerminalUI


class ResearchManagerAgent:
    """Research Manager Agent for navigating ML research monorepos.

    Provides tools for:
    - Exploring repository structure
    - Querying experiments
    - Managing checkpoints and configs
    - Generating training commands
    """

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        memory_path: Optional[Path] = None,
        audit_path: Optional[Path] = None,
    ):
        """Initialize the research manager agent.

        Args:
            repo_root: Root of the monorepo. Defaults to current directory.
            memory_path: Path for persistent memory. If None, uses session-only.
            audit_path: Path for audit log. If None, uses in-memory only.
        """
        self.repo_root = (repo_root or Path.cwd()).resolve()

        # Initialize components
        self.indexer = RepoIndexer(self.repo_root)

        # Memory: use persistent if path provided, otherwise session-only
        if memory_path:
            self.memory = PersistentMemory(memory_path)
        else:
            self.memory = SessionMemory()

        # Safety and audit
        self.safety = SafetyHooks()
        self.audit = AuditLog(audit_path)

        # Tool registry
        self.registry = get_global_registry()

        # UI
        self.ui = TerminalUI()

        # Index the repository
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the agent (index repo, load tools)."""
        if self._initialized:
            return

        self.ui.show_info("Indexing repository...")
        self.indexer.refresh()

        # Load tools
        load_tools(registry=self.registry)

        stats = self.indexer.get_stats()
        self.ui.show_info(
            f"Ready! Found {stats['projects']} projects, "
            f"{stats['scripts']} scripts, {stats['configs']} configs"
        )

        self._initialized = True

    async def get_context(self) -> dict:
        """Get current agent context.

        Returns:
            Dictionary with current state (project, recent items, etc.)
        """
        memory_context = await self.memory.get_context()
        return {
            **memory_context,
            "repo_root": str(self.repo_root),
            "indexer_stats": self.indexer.get_stats(),
            "tools": self.registry.list_names(),
        }

    async def set_project(self, project_name: str) -> bool:
        """Set the current project context.

        Args:
            project_name: Name of the project to set as current.

        Returns:
            True if project exists and was set, False otherwise.
        """
        project = self.indexer.get_project(project_name)
        if project:
            await self.memory.set_context("current_project", project_name)
            return True
        return False

    def list_projects(self, include_archived: bool = False) -> list[dict]:
        """List all projects in the repository.

        Args:
            include_archived: Whether to include archived projects.

        Returns:
            List of project dictionaries.
        """
        projects = self.indexer.list_projects(include_archived=include_archived)
        return [p.to_dict() for p in projects]

    def get_project(self, name: str) -> Optional[dict]:
        """Get a specific project by name.

        Args:
            name: Project name.

        Returns:
            Project dictionary if found, None otherwise.
        """
        project = self.indexer.get_project(name)
        return project.to_dict() if project else None

    def list_scripts(
        self, project: Optional[str] = None, purpose: Optional[str] = None
    ) -> list[dict]:
        """List scripts in the repository.

        Args:
            project: Filter by project name.
            purpose: Filter by purpose (train, evaluate, etc.)

        Returns:
            List of script dictionaries.
        """
        if project:
            scripts = self.indexer.find_scripts_by_project(project)
        elif purpose:
            scripts = self.indexer.find_scripts_by_purpose(purpose)
        else:
            scripts = list(self.indexer.scripts.values())

        return [s.to_dict() for s in scripts]

    def list_configs(self, project: Optional[str] = None) -> list[dict]:
        """List configuration files.

        Args:
            project: Filter by project name.

        Returns:
            List of config dictionaries.
        """
        if project:
            configs = self.indexer.list_configs_for_project(project)
        else:
            configs = list(self.indexer.configs.values())

        return [c.to_dict() for c in configs]

    def get_session_summary(self) -> dict:
        """Get summary of current session.

        Returns:
            Session summary with action counts, errors, etc.
        """
        return self.audit.get_session_summary()


def find_repo_root(start_path: Optional[Path] = None) -> Path:
    """Find the root of the monorepo by looking for markers.

    Args:
        start_path: Path to start searching from.

    Returns:
        Path to the repo root.

    Raises:
        ValueError: If no repo root could be found.
    """
    path = (start_path or Path.cwd()).resolve()

    # Markers that indicate repo root
    markers = ["CLAUDE.md", ".git", "pyproject.toml", "common"]

    while path != path.parent:
        for marker in markers:
            if (path / marker).exists():
                # Additional check: must have projects/ or common/ directory
                if (path / "projects").exists() or (path / "common").exists():
                    return path
        path = path.parent

    raise ValueError("Could not find monorepo root. Are you in the research directory?")


async def run_interactive() -> None:
    """Run the agent in interactive mode."""
    try:
        repo_root = find_repo_root()
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Set up paths for persistent storage
    storage_dir = repo_root / ".research_manager"
    storage_dir.mkdir(exist_ok=True)

    agent = ResearchManagerAgent(
        repo_root=repo_root,
        memory_path=storage_dir / "memory.json",
        audit_path=storage_dir / "audit.json",
    )

    await agent.initialize()

    # Print help
    agent.ui.show_markdown(
        """
# Research Manager Agent

Available commands:
- `projects` - List all projects
- `scripts [project]` - List scripts
- `configs [project]` - List configs
- `context` - Show current context
- `quit` - Exit

Or ask questions in natural language!
    """
    )

    # Simple REPL for now
    # Full agent loop with Claude SDK will be added in Phase 2
    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            if user_input.lower() == "projects":
                projects = agent.list_projects()
                agent.ui.show_projects_table(projects)

            elif user_input.lower().startswith("scripts"):
                parts = user_input.split(maxsplit=1)
                project = parts[1] if len(parts) > 1 else None
                scripts = agent.list_scripts(project=project)
                agent.ui.show_result(scripts, "Scripts")

            elif user_input.lower().startswith("configs"):
                parts = user_input.split(maxsplit=1)
                project = parts[1] if len(parts) > 1 else None
                configs = agent.list_configs(project=project)
                agent.ui.show_result(configs, "Configs")

            elif user_input.lower() == "context":
                context = await agent.get_context()
                agent.ui.show_result(context, "Current Context")

            else:
                agent.ui.show_info(
                    "Natural language queries require Claude SDK integration. " "Coming in Phase 2!"
                )

        except KeyboardInterrupt:
            break
        except Exception as e:
            agent.ui.show_error(str(e))

    agent.ui.show_info("Goodbye!")


def main() -> None:
    """Main entry point."""
    asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
