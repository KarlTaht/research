"""Tool for exploring repository structure."""

import subprocess
from pathlib import Path
from typing import Optional

from research_manager.tools.decorators import tool
from research_manager.tools.registry import ToolCategory


@tool(
    name="explore_repo",
    description="Explore the repository structure. Returns directory tree, key files, and git status.",
    category=ToolCategory.CODEBASE,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Subdirectory to explore (relative to repo root). Default: root",
            },
            "depth": {
                "type": "integer",
                "description": "Maximum depth to traverse. Default: 2",
            },
            "show_hidden": {
                "type": "boolean",
                "description": "Include hidden files/directories. Default: false",
            },
        },
        "required": [],
    },
    read_only=True,
    examples=[
        "explore_repo() - Show repo root structure",
        "explore_repo(path='projects/custom_transformer') - Explore specific project",
        "explore_repo(depth=3) - Show deeper structure",
    ],
)
async def explore_repo(
    path: str = ".",
    depth: int = 2,
    show_hidden: bool = False,
    repo_root: Optional[Path] = None,
) -> dict:
    """Explore repository structure and return detailed information.

    Args:
        path: Subdirectory to explore (relative to repo root).
        depth: Maximum depth to traverse.
        show_hidden: Include hidden files/directories.
        repo_root: Root of the repository (injected by agent).

    Returns:
        Dictionary with structure information.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    target_path = (repo_root / path).resolve()

    # Ensure we're within repo
    try:
        target_path.relative_to(repo_root)
    except ValueError:
        return {"error": f"Path {path} is outside the repository"}

    if not target_path.exists():
        return {"error": f"Path {path} does not exist"}

    if not target_path.is_dir():
        return {"error": f"Path {path} is not a directory"}

    result = {
        "path": str(target_path.relative_to(repo_root)),
        "absolute_path": str(target_path),
        "type": _classify_directory(target_path),
        "contents": _scan_directory(target_path, repo_root, depth, show_hidden),
        "git_status": _get_git_status(target_path, repo_root),
    }

    # Add description if README or CLAUDE.md exists
    for doc_file in ["README.md", "CLAUDE.md"]:
        doc_path = target_path / doc_file
        if doc_path.exists():
            result["description"] = _extract_description(doc_path)
            result["doc_file"] = doc_file
            break

    return result


def _classify_directory(path: Path) -> str:
    """Classify a directory by its contents."""
    if (path / "train.py").exists():
        return "project"
    if (path / "pyproject.toml").exists():
        return "package"
    if (path / "SKILL.md").exists():
        return "skill"
    if path.name == "configs":
        return "configs"
    if path.name == "tests":
        return "tests"
    if path.name == "archive":
        return "archive"
    if any(path.glob("*.py")):
        return "python_module"
    return "directory"


def _scan_directory(
    path: Path,
    repo_root: Path,
    depth: int,
    show_hidden: bool,
    current_depth: int = 0,
) -> dict:
    """Recursively scan directory contents."""
    if current_depth >= depth:
        return {"truncated": True, "reason": f"Max depth {depth} reached"}

    contents = {
        "directories": [],
        "files": [],
        "key_files": [],
    }

    # Key files to highlight
    key_file_names = {
        "train.py",
        "evaluate.py",
        "eval.py",
        "main.py",
        "config.yaml",
        "config.yml",
        "pyproject.toml",
        "README.md",
        "CLAUDE.md",
        "SKILL.md",
        "__init__.py",
    }

    try:
        items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        return {"error": "Permission denied"}

    for item in items:
        # Skip hidden files unless requested
        if item.name.startswith(".") and not show_hidden:
            continue

        # Skip common non-essential directories
        if item.name in {"__pycache__", ".git", "node_modules", ".venv", "venv"}:
            continue

        rel_path = str(item.relative_to(repo_root))

        if item.is_dir():
            dir_info = {
                "name": item.name,
                "path": rel_path,
                "type": _classify_directory(item),
            }

            # Recurse if not at max depth
            if current_depth + 1 < depth:
                dir_info["contents"] = _scan_directory(
                    item, repo_root, depth, show_hidden, current_depth + 1
                )

            contents["directories"].append(dir_info)

        elif item.is_file():
            file_info = {
                "name": item.name,
                "path": rel_path,
                "size_bytes": item.stat().st_size,
            }

            if item.name in key_file_names or item.suffix in {".yaml", ".yml"}:
                contents["key_files"].append(file_info)
            else:
                contents["files"].append(file_info)

        elif item.is_symlink():
            try:
                target = item.resolve()
                contents["files"].append(
                    {
                        "name": item.name,
                        "path": rel_path,
                        "symlink_to": str(target),
                    }
                )
            except (OSError, ValueError):
                contents["files"].append(
                    {
                        "name": item.name,
                        "path": rel_path,
                        "symlink_broken": True,
                    }
                )

    return contents


def _get_git_status(path: Path, repo_root: Path) -> dict:
    """Get git status for the directory."""
    try:
        # Get status for this directory
        result = subprocess.run(
            ["git", "status", "--porcelain", str(path.relative_to(repo_root))],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            return {"status": "not_in_git"}

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

        modified = []
        untracked = []
        staged = []

        for line in lines:
            if not line:
                continue
            status = line[:2]
            file_path = line[3:]

            if status[0] in "MADRC":  # Staged changes
                staged.append(file_path)
            if status[1] == "M":  # Modified
                modified.append(file_path)
            if status == "??":  # Untracked
                untracked.append(file_path)

        if not modified and not untracked and not staged:
            return {"status": "clean"}

        return {
            "status": "dirty",
            "modified": modified[:10],  # Limit to 10
            "untracked": untracked[:10],
            "staged": staged[:10],
            "total_changes": len(modified) + len(untracked) + len(staged),
        }

    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"status": "unknown"}


def _extract_description(doc_path: Path) -> str:
    """Extract description from README or CLAUDE.md."""
    try:
        content = doc_path.read_text()
        lines = content.strip().split("\n")

        # Skip title line (# Header)
        description_lines = []
        started = False

        for line in lines:
            stripped = line.strip()

            # Skip empty lines at the start
            if not started and not stripped:
                continue

            # Skip the title
            if not started and stripped.startswith("# "):
                started = True
                continue

            # Stop at next header or after first paragraph
            if started:
                if stripped.startswith("#"):
                    break
                if not stripped and description_lines:
                    break
                if stripped:
                    description_lines.append(stripped)

        return " ".join(description_lines)[:300]

    except Exception:
        return ""
