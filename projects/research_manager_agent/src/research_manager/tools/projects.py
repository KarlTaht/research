"""Tool for listing and querying projects."""

from typing import Optional

from research_manager.tools.decorators import tool
from research_manager.tools.registry import ToolCategory
from research_manager.indexer import RepoIndexer


@tool(
    name="list_projects",
    description="List all projects in the repository with their status, scripts, and configs.",
    category=ToolCategory.CODEBASE,
    parameters={
        "type": "object",
        "properties": {
            "include_archived": {
                "type": "boolean",
                "description": "Include archived/deprecated projects. Default: false",
            },
            "filter_type": {
                "type": "string",
                "description": "Filter by project type: 'project', 'paper', 'archive'. Default: all",
            },
            "has_train_script": {
                "type": "boolean",
                "description": "Only show projects with train.py. Default: false (show all)",
            },
        },
        "required": [],
    },
    read_only=True,
    examples=[
        "list_projects() - List all active projects",
        "list_projects(include_archived=True) - Include archived projects",
        "list_projects(has_train_script=True) - Only trainable projects",
    ],
)
async def list_projects(
    include_archived: bool = False,
    filter_type: Optional[str] = None,
    has_train_script: Optional[bool] = None,
    indexer: Optional[RepoIndexer] = None,
) -> dict:
    """List all projects in the repository.

    Args:
        include_archived: Include archived projects.
        filter_type: Filter by project type.
        has_train_script: Only projects with train.py.
        indexer: Repository indexer (injected by agent).

    Returns:
        Dictionary with project list and summary.
    """
    if indexer is None:
        return {"error": "Indexer not available. Please initialize the agent first."}

    projects = indexer.list_projects(include_archived=include_archived)

    # Apply filters
    if filter_type:
        projects = [p for p in projects if p.type == filter_type]

    if has_train_script is not None:
        projects = [p for p in projects if p.has_train_script == has_train_script]

    # Convert to dicts and add extra info
    project_list = []
    for project in projects:
        info = project.to_dict()

        # Add script summary
        scripts = []
        if project.has_train_script:
            scripts.append("train.py")
        if project.has_eval_script:
            scripts.append("evaluate.py")
        info["available_scripts"] = scripts

        # Add config count
        info["config_count"] = len(project.config_files)

        project_list.append(info)

    # Summary stats
    summary = {
        "total": len(project_list),
        "with_train_script": sum(1 for p in project_list if "train.py" in p["available_scripts"]),
        "with_configs": sum(1 for p in project_list if p["config_count"] > 0),
        "types": {},
    }

    for p in project_list:
        ptype = p["type"]
        summary["types"][ptype] = summary["types"].get(ptype, 0) + 1

    return {
        "projects": project_list,
        "summary": summary,
    }


@tool(
    name="get_project",
    description="Get detailed information about a specific project.",
    category=ToolCategory.CODEBASE,
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Project name (directory name)",
            },
        },
        "required": ["name"],
    },
    read_only=True,
    examples=[
        "get_project(name='custom_transformer')",
        "get_project(name='embedded_attention')",
    ],
)
async def get_project(
    name: str,
    indexer: Optional[RepoIndexer] = None,
) -> dict:
    """Get detailed information about a specific project.

    Args:
        name: Project name.
        indexer: Repository indexer (injected by agent).

    Returns:
        Dictionary with detailed project information.
    """
    if indexer is None:
        return {"error": "Indexer not available. Please initialize the agent first."}

    project = indexer.get_project(name)

    if project is None:
        # Try to find similar names
        all_projects = indexer.list_projects(include_archived=True)
        similar = [p.name for p in all_projects if name.lower() in p.name.lower()]

        return {
            "error": f"Project '{name}' not found",
            "suggestions": similar[:5] if similar else None,
        }

    info = project.to_dict()

    # Add scripts info
    scripts = indexer.find_scripts_by_project(name)
    info["scripts"] = [
        {
            "name": s.name,
            "purpose": s.purpose,
            "path": str(s.path),
            "arguments": s.arguments,
            "example_command": s.example_command,
        }
        for s in scripts
    ]

    # Add configs info
    configs = indexer.list_configs_for_project(name)
    info["configs"] = [
        {
            "name": c.path.name,
            "path": str(c.path),
            "has_model_params": c.model_params is not None,
            "has_training_params": c.training_params is not None,
        }
        for c in configs
    ]

    return info
