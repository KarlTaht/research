"""Tool for finding and querying scripts."""

from typing import Optional

from research_manager.tools.decorators import tool
from research_manager.tools.registry import ToolCategory
from research_manager.indexer import RepoIndexer


# Mapping from task descriptions to script purposes
TASK_MAPPINGS = {
    "train": ["train"],
    "training": ["train"],
    "learn": ["train"],
    "fit": ["train"],
    "evaluate": ["evaluate", "validate"],
    "eval": ["evaluate", "validate"],
    "test": ["evaluate", "validate", "test"],
    "validate": ["validate", "evaluate"],
    "download": ["download"],
    "fetch": ["download"],
    "get data": ["download"],
    "analyze": ["analyze"],
    "analysis": ["analyze"],
    "visualize": ["visualize", "analyze"],
    "plot": ["visualize"],
    "run": ["run", "train"],
}


@tool(
    name="find_script",
    description="Find scripts that match a task description. Describe what you want to do.",
    category=ToolCategory.CODEBASE,
    parameters={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What you want to do (e.g., 'train a model', 'download dataset')",
            },
            "project": {
                "type": "string",
                "description": "Specific project to search in (optional)",
            },
        },
        "required": ["task"],
    },
    read_only=True,
    examples=[
        "find_script(task='train a model')",
        "find_script(task='download dataset')",
        "find_script(task='evaluate', project='custom_transformer')",
    ],
)
async def find_script(
    task: str,
    project: Optional[str] = None,
    indexer: Optional[RepoIndexer] = None,
) -> dict:
    """Find scripts matching a task description.

    Args:
        task: Description of what you want to do.
        project: Specific project to search in.
        indexer: Repository indexer (injected by agent).

    Returns:
        Dictionary with matching scripts and suggestions.
    """
    if indexer is None:
        return {"error": "Indexer not available. Please initialize the agent first."}

    task_lower = task.lower()

    # Determine which purposes to search for
    target_purposes = set()
    for keyword, purposes in TASK_MAPPINGS.items():
        if keyword in task_lower:
            target_purposes.update(purposes)

    # If no mapping found, try to match script names directly
    if not target_purposes:
        target_purposes = {"train", "evaluate", "run"}  # Default purposes

    # Find matching scripts
    matches = []

    if project:
        # Search specific project
        scripts = indexer.find_scripts_by_project(project)
        for script in scripts:
            if script.purpose in target_purposes or _matches_task(script, task_lower):
                matches.append(_script_to_dict(script, relevance="high"))
    else:
        # Search all scripts
        for purpose in target_purposes:
            scripts = indexer.find_scripts_by_purpose(purpose)
            for script in scripts:
                matches.append(_script_to_dict(script, relevance="high"))

        # Also search by name matching
        for script in indexer.scripts.values():
            if script not in [m["_script"] for m in matches if "_script" in m]:
                if _matches_task(script, task_lower):
                    matches.append(_script_to_dict(script, relevance="medium"))

    # Remove duplicates and internal fields
    seen = set()
    unique_matches = []
    for match in matches:
        match.pop("_script", None)
        key = match["path"]
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)

    # Sort by relevance then name
    unique_matches.sort(key=lambda x: (x["relevance"] != "high", x["name"]))

    return {
        "task": task,
        "matches": unique_matches[:10],  # Limit to 10
        "total_found": len(unique_matches),
        "searched_purposes": list(target_purposes),
    }


def _matches_task(script, task_lower: str) -> bool:
    """Check if script matches task by name."""
    name_lower = script.name.lower()

    # Check if task words appear in script name
    task_words = task_lower.split()
    for word in task_words:
        if len(word) > 3 and word in name_lower:
            return True

    return False


def _script_to_dict(script, relevance: str = "medium") -> dict:
    """Convert script to dictionary with extra info."""
    return {
        "name": script.name,
        "path": str(script.path),
        "purpose": script.purpose,
        "project": script.project,
        "arguments": script.arguments,
        "example_command": script.example_command,
        "relevance": relevance,
        "_script": script,  # For deduplication, removed later
    }


@tool(
    name="list_scripts",
    description="List all scripts in the repository, optionally filtered by purpose or project.",
    category=ToolCategory.CODEBASE,
    parameters={
        "type": "object",
        "properties": {
            "purpose": {
                "type": "string",
                "description": "Filter by purpose: train, evaluate, download, analyze, etc.",
            },
            "project": {
                "type": "string",
                "description": "Filter by project name",
            },
        },
        "required": [],
    },
    read_only=True,
    examples=[
        "list_scripts() - List all scripts",
        "list_scripts(purpose='train') - List training scripts",
        "list_scripts(project='custom_transformer') - Scripts in a project",
    ],
)
async def list_scripts(
    purpose: Optional[str] = None,
    project: Optional[str] = None,
    indexer: Optional[RepoIndexer] = None,
) -> dict:
    """List scripts in the repository.

    Args:
        purpose: Filter by script purpose.
        project: Filter by project name.
        indexer: Repository indexer (injected by agent).

    Returns:
        Dictionary with script list and summary.
    """
    if indexer is None:
        return {"error": "Indexer not available. Please initialize the agent first."}

    if project:
        scripts = indexer.find_scripts_by_project(project)
    elif purpose:
        scripts = indexer.find_scripts_by_purpose(purpose)
    else:
        scripts = list(indexer.scripts.values())

    script_list = [
        {
            "name": s.name,
            "path": str(s.path),
            "purpose": s.purpose,
            "project": s.project,
            "example_command": s.example_command,
        }
        for s in scripts
    ]

    # Summary by purpose
    purpose_counts = {}
    for s in scripts:
        purpose_counts[s.purpose] = purpose_counts.get(s.purpose, 0) + 1

    return {
        "scripts": script_list,
        "total": len(script_list),
        "by_purpose": purpose_counts,
    }
