"""Tools for reading and comparing configuration files."""

from pathlib import Path
from typing import Optional

import yaml

from research_repository.tools.decorators import tool
from research_repository.tools.registry import ToolCategory
from research_repository.indexer import RepoIndexer


@tool(
    name="read_config",
    description="Read and explain a YAML configuration file. Shows structure, parameters, and computed values.",
    category=ToolCategory.CODEBASE,
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to config file (relative to repo root)",
            },
            "section": {
                "type": "string",
                "description": "Specific section to focus on (e.g., 'model', 'training')",
            },
        },
        "required": ["path"],
    },
    read_only=True,
    examples=[
        "read_config(path='projects/custom_transformer/configs/default.yaml')",
        "read_config(path='configs/tinystories.yaml', section='model')",
    ],
)
async def read_config(
    path: str,
    section: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> dict:
    """Read and parse a YAML configuration file.

    Args:
        path: Path to config file.
        section: Specific section to focus on.
        repo_root: Repository root (injected by agent).

    Returns:
        Dictionary with parsed config and analysis.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    config_path = repo_root / path

    if not config_path.exists():
        return {"error": f"Config file not found: {path}"}

    if config_path.suffix not in {".yaml", ".yml"}:
        return {"error": f"Not a YAML file: {path}"}

    try:
        content = yaml.safe_load(config_path.read_text())
    except yaml.YAMLError as e:
        return {"error": f"Invalid YAML: {e}"}

    if content is None:
        return {"error": "Config file is empty"}

    result = {
        "path": path,
        "sections": list(content.keys()) if isinstance(content, dict) else [],
    }

    if section:
        if section in content:
            result["content"] = {section: content[section]}
            result["focused_section"] = section
        else:
            result["error"] = f"Section '{section}' not found"
            result["available_sections"] = list(content.keys())
            result["content"] = content
    else:
        result["content"] = content

    # Add computed values for ML configs
    if isinstance(content, dict):
        result["computed"] = _compute_derived_values(content)
        result["analysis"] = _analyze_config(content)

    return result


def _compute_derived_values(config: dict) -> dict:
    """Compute derived values from config."""
    computed = {}

    # Model parameters
    model = config.get("model", config.get("model_config", {}))
    if model:
        d_model = model.get("d_model", model.get("hidden_size", model.get("embed_dim")))
        n_heads = model.get("n_heads", model.get("num_attention_heads", model.get("num_heads")))
        n_layers = model.get("n_layers", model.get("num_hidden_layers", model.get("n_blocks")))
        vocab_size = model.get("vocab_size")

        if d_model and n_heads:
            computed["head_dim"] = d_model // n_heads

        # Rough parameter estimate for transformer
        if d_model and n_layers and vocab_size:
            # Embedding + n_layers * (attention + ffn) + output
            embed_params = vocab_size * d_model
            ffn_dim = model.get("d_ff", model.get("intermediate_size", d_model * 4))
            layer_params = (4 * d_model * d_model) + (2 * d_model * ffn_dim)  # Rough estimate
            total_params = embed_params + (n_layers * layer_params) + embed_params
            computed["estimated_params"] = total_params
            computed["estimated_params_str"] = _format_params(total_params)

    # Training parameters
    training = config.get("training", config.get("train_config", {}))
    if training:
        batch_size = training.get("batch_size", training.get("per_device_train_batch_size"))
        grad_accum = training.get("gradient_accumulation_steps", 1)

        if batch_size and grad_accum:
            computed["effective_batch_size"] = batch_size * grad_accum

        # Estimate training tokens
        max_steps = training.get("max_steps")
        seq_len = (
            model.get("max_seq_len", model.get("max_position_embeddings", 512)) if model else 512
        )

        if max_steps and batch_size:
            computed["total_tokens"] = max_steps * batch_size * seq_len
            computed["total_tokens_str"] = _format_tokens(computed["total_tokens"])

    return computed


def _analyze_config(config: dict) -> dict:
    """Analyze config for potential issues or notable settings."""
    analysis = {"notes": [], "warnings": []}

    model = config.get("model", config.get("model_config", {}))
    training = config.get("training", config.get("train_config", {}))

    # Check for common issues
    if model:
        d_model = model.get("d_model", model.get("hidden_size"))
        n_heads = model.get("n_heads", model.get("num_attention_heads"))

        if d_model and n_heads and d_model % n_heads != 0:
            analysis["warnings"].append(
                f"d_model ({d_model}) is not divisible by n_heads ({n_heads})"
            )

    if training:
        lr = training.get("learning_rate", training.get("lr"))
        if lr and lr > 1e-2:
            analysis["warnings"].append(f"Learning rate ({lr}) seems high")
        if lr and lr < 1e-6:
            analysis["warnings"].append(f"Learning rate ({lr}) seems very low")

        batch_size = training.get("batch_size")
        if batch_size and batch_size > 256:
            analysis["notes"].append(f"Large batch size ({batch_size}) - ensure enough GPU memory")

    return analysis


def _format_params(params: int) -> str:
    """Format parameter count in human-readable form."""
    if params >= 1e9:
        return f"{params / 1e9:.1f}B"
    if params >= 1e6:
        return f"{params / 1e6:.1f}M"
    if params >= 1e3:
        return f"{params / 1e3:.1f}K"
    return str(params)


def _format_tokens(tokens: int) -> str:
    """Format token count in human-readable form."""
    if tokens >= 1e12:
        return f"{tokens / 1e12:.1f}T"
    if tokens >= 1e9:
        return f"{tokens / 1e9:.1f}B"
    if tokens >= 1e6:
        return f"{tokens / 1e6:.1f}M"
    if tokens >= 1e3:
        return f"{tokens / 1e3:.1f}K"
    return str(tokens)


@tool(
    name="compare_configs",
    description="Compare two configuration files and show differences.",
    category=ToolCategory.CODEBASE,
    parameters={
        "type": "object",
        "properties": {
            "config1": {
                "type": "string",
                "description": "Path to first config file",
            },
            "config2": {
                "type": "string",
                "description": "Path to second config file",
            },
        },
        "required": ["config1", "config2"],
    },
    read_only=True,
    examples=[
        "compare_configs(config1='configs/small.yaml', config2='configs/large.yaml')",
    ],
)
async def compare_configs(
    config1: str,
    config2: str,
    repo_root: Optional[Path] = None,
) -> dict:
    """Compare two configuration files.

    Args:
        config1: Path to first config.
        config2: Path to second config.
        repo_root: Repository root (injected by agent).

    Returns:
        Dictionary with differences.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    path1 = repo_root / config1
    path2 = repo_root / config2

    errors = []
    if not path1.exists():
        errors.append(f"Config not found: {config1}")
    if not path2.exists():
        errors.append(f"Config not found: {config2}")

    if errors:
        return {"error": "; ".join(errors)}

    try:
        content1 = yaml.safe_load(path1.read_text()) or {}
        content2 = yaml.safe_load(path2.read_text()) or {}
    except yaml.YAMLError as e:
        return {"error": f"Invalid YAML: {e}"}

    differences = _diff_dicts(content1, content2)

    return {
        "config1": config1,
        "config2": config2,
        "differences": differences,
        "summary": {
            "only_in_first": len(differences.get("only_in_first", {})),
            "only_in_second": len(differences.get("only_in_second", {})),
            "different_values": len(differences.get("different", {})),
            "identical": len(differences.get("identical", {})),
        },
    }


def _diff_dicts(
    d1: dict,
    d2: dict,
    path: str = "",
) -> dict:
    """Recursively diff two dictionaries."""
    result = {
        "only_in_first": {},
        "only_in_second": {},
        "different": {},
        "identical": {},
    }

    all_keys = set(d1.keys()) | set(d2.keys())

    for key in all_keys:
        full_path = f"{path}.{key}" if path else key

        if key not in d2:
            result["only_in_first"][full_path] = d1[key]
        elif key not in d1:
            result["only_in_second"][full_path] = d2[key]
        elif d1[key] == d2[key]:
            result["identical"][full_path] = d1[key]
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            # Recurse for nested dicts
            nested = _diff_dicts(d1[key], d2[key], full_path)
            for category in result:
                result[category].update(nested[category])
        else:
            result["different"][full_path] = {
                "first": d1[key],
                "second": d2[key],
            }

    return result


@tool(
    name="list_configs",
    description="List all configuration files, optionally filtered by project.",
    category=ToolCategory.CODEBASE,
    parameters={
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Filter by project name",
            },
        },
        "required": [],
    },
    read_only=True,
    examples=[
        "list_configs() - List all configs",
        "list_configs(project='custom_transformer') - Configs for a project",
    ],
)
async def list_configs(
    project: Optional[str] = None,
    indexer: Optional[RepoIndexer] = None,
) -> dict:
    """List configuration files.

    Args:
        project: Filter by project name.
        indexer: Repository indexer (injected by agent).

    Returns:
        Dictionary with config list.
    """
    if indexer is None:
        return {"error": "Indexer not available. Please initialize the agent first."}

    if project:
        configs = indexer.list_configs_for_project(project)
    else:
        configs = list(indexer.configs.values())

    config_list = [
        {
            "name": c.path.name,
            "path": str(c.path),
            "project": c.project,
            "sections": list(c.sections.keys()) if c.sections else [],
            "has_model": c.model_params is not None,
            "has_training": c.training_params is not None,
        }
        for c in configs
    ]

    # Group by project
    by_project = {}
    for c in config_list:
        proj = c["project"]
        if proj not in by_project:
            by_project[proj] = []
        by_project[proj].append(c["name"])

    return {
        "configs": config_list,
        "total": len(config_list),
        "by_project": by_project,
    }
