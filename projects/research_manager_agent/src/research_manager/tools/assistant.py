"""Tools for interactive assistance - command execution, cleanup, error explanation."""

import re
import subprocess
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

from research_manager.tools.decorators import tool
from research_manager.tools.registry import ToolCategory
from research_manager.safety import SafetyHooks


# Common error patterns and their explanations
ERROR_PATTERNS = {
    r"CUDA out of memory": {
        "category": "GPU Memory",
        "explanation": "The model or batch size is too large for available GPU memory.",
        "suggestions": [
            "Reduce batch_size in your config",
            "Use gradient accumulation to simulate larger batches",
            "Enable mixed precision training (fp16/bf16)",
            "Use gradient checkpointing to trade compute for memory",
            "Try a smaller model variant",
        ],
    },
    r"RuntimeError: Expected .* but got .*": {
        "category": "Tensor Shape Mismatch",
        "explanation": "Tensor dimensions don't match between operations.",
        "suggestions": [
            "Check input/output dimensions in your model",
            "Verify data preprocessing produces correct shapes",
            "Print tensor shapes at each layer to find mismatch",
            "Check if batch dimension is being handled correctly",
        ],
    },
    r"ModuleNotFoundError: No module named": {
        "category": "Missing Import",
        "explanation": "A required Python package is not installed.",
        "suggestions": [
            "Install the missing package: uv pip install <package>",
            "Check if you're in the correct virtual environment",
            "Verify the package name is spelled correctly",
        ],
    },
    r"FileNotFoundError": {
        "category": "Missing File",
        "explanation": "A required file or directory doesn't exist.",
        "suggestions": [
            "Check if the path is correct",
            "Verify the file was created/downloaded",
            "Check for typos in the filename",
            "Ensure parent directories exist",
        ],
    },
    r"KeyError": {
        "category": "Missing Key",
        "explanation": "Trying to access a dictionary key that doesn't exist.",
        "suggestions": [
            "Check the config file for the expected key",
            "Use .get() with a default value",
            "Print available keys to debug",
        ],
    },
    r"ValueError: .* not in list": {
        "category": "Value Error",
        "explanation": "Trying to find a value that doesn't exist in a list.",
        "suggestions": [
            "Check if the value exists before accessing",
            "Verify the data being searched is correct",
        ],
    },
    r"torch\.cuda\.is_available\(\) .* False": {
        "category": "CUDA Not Available",
        "explanation": "PyTorch cannot access GPU/CUDA.",
        "suggestions": [
            "Check NVIDIA driver: nvidia-smi",
            "Verify CUDA version matches PyTorch: python -c 'import torch; print(torch.version.cuda)'",
            "Reinstall PyTorch with correct CUDA version",
            "Check if GPU is visible: nvidia-smi -L",
        ],
    },
    r"Permission denied": {
        "category": "Permission Error",
        "explanation": "Insufficient permissions to access a file or resource.",
        "suggestions": [
            "Check file permissions: ls -la <file>",
            "Ensure you own the file or have read/write access",
            "Don't use sudo for pip/uv installs",
        ],
    },
    r"Connection refused|ConnectionError": {
        "category": "Network Error",
        "explanation": "Cannot connect to a server or service.",
        "suggestions": [
            "Check your internet connection",
            "Verify the server/service is running",
            "Check firewall settings",
            "Try again later (may be temporary)",
        ],
    },
    r"gradient.*nan|loss.*nan|NaN": {
        "category": "NaN in Training",
        "explanation": "Training produced NaN values, often due to numerical instability.",
        "suggestions": [
            "Reduce learning rate",
            "Add gradient clipping",
            "Check for division by zero in loss",
            "Enable anomaly detection: torch.autograd.set_detect_anomaly(True)",
            "Check input data for invalid values",
        ],
    },
}


@tool(
    name="run_command",
    description="Generate and execute shell commands safely. Shows command before execution and requires confirmation for risky operations.",
    category=ToolCategory.ASSISTANT,
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute",
            },
            "description": {
                "type": "string",
                "description": "Brief description of what this command does",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for command (optional)",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 60)",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, show command without executing",
            },
        },
        "required": ["command"],
    },
    read_only=False,
    requires_confirmation=True,
    examples=[
        "run_command(command='python train.py --config config.yaml')",
        "run_command(command='nvidia-smi', description='Check GPU status')",
        "run_command(command='ls -la', dry_run=True)",
    ],
)
async def run_command(
    command: str,
    description: Optional[str] = None,
    working_dir: Optional[str] = None,
    timeout: int = 60,
    dry_run: bool = False,
    repo_root: Optional[Path] = None,
) -> dict:
    """Execute a shell command safely.

    Args:
        command: Shell command to execute.
        description: Brief description of the command.
        working_dir: Working directory for execution.
        timeout: Command timeout in seconds.
        dry_run: If True, show command without executing.
        repo_root: Repository root path.

    Returns:
        Dictionary with command output and status.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    # Check safety
    from research_manager.safety.hooks import SafetyDecision

    safety = SafetyHooks()
    safety_result = safety.pre_tool_use("run_command", {"command": command, "execute": not dry_run})

    if safety_result.decision == SafetyDecision.DENY:
        return {
            "error": "Command blocked by safety hooks",
            "reason": safety_result.reason,
            "command": command,
        }

    result = {
        "command": command,
        "description": description,
        "requires_confirmation": safety_result.decision == SafetyDecision.CONFIRM,
    }

    if dry_run:
        result["dry_run"] = True
        result["message"] = "Command not executed (dry run mode)"
        return result

    # Determine working directory
    if working_dir:
        cwd = repo_root / working_dir
        if not cwd.exists():
            return {"error": f"Working directory not found: {working_dir}"}
    else:
        cwd = repo_root

    # Execute command
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        result["exit_code"] = proc.returncode
        result["stdout"] = proc.stdout[:10000] if proc.stdout else ""
        result["stderr"] = proc.stderr[:10000] if proc.stderr else ""
        result["success"] = proc.returncode == 0

        if proc.returncode != 0:
            result["error"] = f"Command failed with exit code {proc.returncode}"

    except subprocess.TimeoutExpired:
        result["error"] = f"Command timed out after {timeout} seconds"
        result["success"] = False
    except Exception as e:
        result["error"] = str(e)
        result["success"] = False

    return result


@tool(
    name="suggest_cleanup",
    description="Analyze the repository and suggest cleanup actions. Identifies old checkpoints, orphan files, large files, and organizational issues.",
    category=ToolCategory.ASSISTANT,
    parameters={
        "type": "object",
        "properties": {
            "scope": {
                "type": "string",
                "enum": ["checkpoints", "logs", "cache", "all"],
                "description": "What to analyze (default: all)",
            },
            "max_age_days": {
                "type": "integer",
                "description": "Consider files older than this for cleanup (default: 30)",
            },
            "min_size_mb": {
                "type": "number",
                "description": "Minimum file size in MB to report (default: 10)",
            },
        },
        "required": [],
    },
    read_only=True,
    examples=[
        "suggest_cleanup()",
        "suggest_cleanup(scope='checkpoints', max_age_days=7)",
        "suggest_cleanup(min_size_mb=100)",
    ],
)
async def suggest_cleanup(
    scope: str = "all",
    max_age_days: int = 30,
    min_size_mb: float = 10.0,
    repo_root: Optional[Path] = None,
) -> dict:
    """Suggest cleanup actions for the repository.

    Args:
        scope: What to analyze (checkpoints, logs, cache, all).
        max_age_days: Age threshold in days.
        min_size_mb: Minimum file size to report.
        repo_root: Repository root path.

    Returns:
        Dictionary with cleanup suggestions.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    cutoff_date = datetime.now() - timedelta(days=max_age_days)
    min_size_bytes = min_size_mb * 1024 * 1024

    suggestions = {
        "old_checkpoints": [],
        "large_files": [],
        "old_logs": [],
        "cache_files": [],
        "orphan_files": [],
        "summary": {},
    }

    total_reclaimable = 0

    # Checkpoint patterns
    checkpoint_patterns = ["**/*.pt", "**/*.pth", "**/*.ckpt", "**/*.safetensors"]
    checkpoint_dirs = [
        repo_root / "assets" / "outputs" / "checkpoints",
        repo_root / "checkpoints",
    ]

    # Log patterns
    log_patterns = ["**/*.log", "**/logs/**/*"]
    log_dirs = [
        repo_root / "assets" / "outputs" / "logs",
        repo_root / "logs",
    ]

    # Cache patterns
    cache_patterns = ["**/__pycache__/**", "**/*.pyc", "**/.cache/**", "**/wandb/**"]

    if scope in ("checkpoints", "all"):
        for ckpt_dir in checkpoint_dirs:
            if not ckpt_dir.exists():
                continue
            for pattern in checkpoint_patterns:
                for f in ckpt_dir.glob(pattern):
                    if not f.is_file():
                        continue
                    stat = f.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    size_mb = stat.st_size / (1024 * 1024)

                    if mtime < cutoff_date:
                        suggestions["old_checkpoints"].append(
                            {
                                "path": str(f.relative_to(repo_root)),
                                "size_mb": round(size_mb, 2),
                                "modified": mtime.isoformat(),
                                "age_days": (datetime.now() - mtime).days,
                            }
                        )
                        total_reclaimable += stat.st_size

    if scope in ("logs", "all"):
        for log_dir in log_dirs:
            if not log_dir.exists():
                continue
            for pattern in log_patterns:
                for f in log_dir.glob(pattern):
                    if not f.is_file():
                        continue
                    stat = f.stat()
                    mtime = datetime.fromtimestamp(stat.st_mtime)
                    size_mb = stat.st_size / (1024 * 1024)

                    if mtime < cutoff_date:
                        suggestions["old_logs"].append(
                            {
                                "path": str(f.relative_to(repo_root)),
                                "size_mb": round(size_mb, 2),
                                "modified": mtime.isoformat(),
                            }
                        )
                        total_reclaimable += stat.st_size

    if scope in ("cache", "all"):
        for pattern in cache_patterns:
            for f in repo_root.glob(pattern):
                if not f.is_file():
                    continue
                try:
                    stat = f.stat()
                    suggestions["cache_files"].append(
                        {
                            "path": str(f.relative_to(repo_root)),
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        }
                    )
                    total_reclaimable += stat.st_size
                except (OSError, PermissionError):
                    continue

    # Find large files anywhere
    if scope == "all":
        for f in repo_root.rglob("*"):
            if not f.is_file():
                continue
            # Skip .git directory
            if ".git" in f.parts:
                continue
            try:
                stat = f.stat()
                if stat.st_size >= min_size_bytes:
                    suggestions["large_files"].append(
                        {
                            "path": str(f.relative_to(repo_root)),
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        }
                    )
            except (OSError, PermissionError):
                continue

    # Sort large files by size
    suggestions["large_files"].sort(key=lambda x: x["size_mb"], reverse=True)
    suggestions["large_files"] = suggestions["large_files"][:20]  # Limit

    # Limit other lists
    suggestions["old_checkpoints"] = suggestions["old_checkpoints"][:20]
    suggestions["old_logs"] = suggestions["old_logs"][:20]
    suggestions["cache_files"] = suggestions["cache_files"][:50]

    # Summary
    suggestions["summary"] = {
        "old_checkpoints_count": len(suggestions["old_checkpoints"]),
        "large_files_count": len(suggestions["large_files"]),
        "old_logs_count": len(suggestions["old_logs"]),
        "cache_files_count": len(suggestions["cache_files"]),
        "total_reclaimable_mb": round(total_reclaimable / (1024 * 1024), 2),
        "scope": scope,
        "max_age_days": max_age_days,
        "min_size_mb": min_size_mb,
    }

    # Generate cleanup commands
    if suggestions["old_checkpoints"]:
        suggestions["cleanup_commands"] = []
        for ckpt in suggestions["old_checkpoints"][:5]:
            suggestions["cleanup_commands"].append(f"rm '{ckpt['path']}'")

    return suggestions


@tool(
    name="explain_error",
    description="Explain an error message with context-aware suggestions. Analyzes Python tracebacks and common ML errors.",
    category=ToolCategory.ASSISTANT,
    parameters={
        "type": "object",
        "properties": {
            "error": {
                "type": "string",
                "description": "The error message or traceback to explain",
            },
            "context": {
                "type": "string",
                "description": "Additional context (e.g., what you were trying to do)",
            },
        },
        "required": ["error"],
    },
    read_only=True,
    examples=[
        "explain_error(error='CUDA out of memory')",
        "explain_error(error='ModuleNotFoundError: No module named torch')",
        "explain_error(error='RuntimeError: Expected tensor for argument', context='training transformer')",
    ],
)
async def explain_error(
    error: str,
    context: Optional[str] = None,
) -> dict:
    """Explain an error message with suggestions.

    Args:
        error: Error message or traceback.
        context: Additional context about the operation.

    Returns:
        Dictionary with explanation and suggestions.
    """
    result = {
        "original_error": error[:2000],
        "context": context,
        "matched_patterns": [],
        "explanation": None,
        "suggestions": [],
        "related_docs": [],
    }

    # Check against known patterns
    for pattern, info in ERROR_PATTERNS.items():
        if re.search(pattern, error, re.IGNORECASE):
            result["matched_patterns"].append(
                {
                    "pattern": pattern,
                    "category": info["category"],
                    "explanation": info["explanation"],
                    "suggestions": info["suggestions"],
                }
            )

    # Extract the most relevant explanation
    if result["matched_patterns"]:
        primary = result["matched_patterns"][0]
        result["explanation"] = primary["explanation"]
        result["suggestions"] = primary["suggestions"]
        result["category"] = primary["category"]
    else:
        # Generic analysis
        result["explanation"] = "Could not match to a known error pattern."
        result["suggestions"] = [
            "Search for the error message online",
            "Check the full traceback for the root cause",
            "Verify your environment and dependencies",
            "Try reproducing with a minimal example",
        ]
        result["category"] = "Unknown"

    # Parse traceback for file locations
    traceback_files = re.findall(r'File "([^"]+)", line (\d+)', error)
    if traceback_files:
        result["traceback_locations"] = [
            {"file": fname, "line": int(line_num)} for fname, line_num in traceback_files[:5]
        ]

    # Add context-specific suggestions
    if context:
        context_lower = context.lower()
        if "train" in context_lower:
            result["suggestions"].append("Check training config for common issues")
            result["suggestions"].append("Verify dataset loading works correctly")
        if "eval" in context_lower:
            result["suggestions"].append("Ensure model and data are on same device")
            result["suggestions"].append("Check if model is in eval mode: model.eval()")
        if "checkpoint" in context_lower or "load" in context_lower:
            result["suggestions"].append("Verify checkpoint file exists and is valid")
            result["suggestions"].append("Check if model architecture matches checkpoint")

    # Add related documentation links
    if "CUDA" in error or "cuda" in error:
        result["related_docs"].append(
            {
                "title": "PyTorch CUDA Debugging",
                "url": "https://pytorch.org/docs/stable/notes/cuda.html",
            }
        )
    if "tensor" in error.lower() or "shape" in error.lower():
        result["related_docs"].append(
            {
                "title": "PyTorch Tensor Operations",
                "url": "https://pytorch.org/docs/stable/tensors.html",
            }
        )

    return result


@tool(
    name="generate_train_command",
    description="Generate a training command based on project, config, and options.",
    category=ToolCategory.ASSISTANT,
    parameters={
        "type": "object",
        "properties": {
            "project": {
                "type": "string",
                "description": "Project name to train",
            },
            "config": {
                "type": "string",
                "description": "Config file to use (optional, uses default if not specified)",
            },
            "overrides": {
                "type": "object",
                "description": "Config overrides as key-value pairs",
            },
            "device": {
                "type": "string",
                "description": "Device to train on (cuda, cpu, auto)",
            },
            "resume": {
                "type": "string",
                "description": "Checkpoint path to resume from",
            },
        },
        "required": ["project"],
    },
    read_only=True,
    examples=[
        "generate_train_command(project='custom_transformer')",
        "generate_train_command(project='custom_transformer', config='configs/large.yaml')",
        "generate_train_command(project='custom_transformer', overrides={'batch_size': 64})",
    ],
)
async def generate_train_command(
    project: str,
    config: Optional[str] = None,
    overrides: Optional[dict] = None,
    device: Optional[str] = None,
    resume: Optional[str] = None,
    repo_root: Optional[Path] = None,
) -> dict:
    """Generate a training command for a project.

    Args:
        project: Project name.
        config: Config file path.
        overrides: Config overrides.
        device: Training device.
        resume: Checkpoint to resume from.
        repo_root: Repository root path.

    Returns:
        Dictionary with command and explanation.
    """
    if repo_root is None:
        repo_root = Path.cwd()

    project_dir = repo_root / "projects" / project
    if not project_dir.exists():
        return {"error": f"Project not found: {project}"}

    # Find train script
    train_script = project_dir / "train.py"
    if not train_script.exists():
        return {"error": f"No train.py found in {project}"}

    # Build command parts
    cmd_parts = ["python", f"projects/{project}/train.py"]

    # Config file
    if config:
        config_path = project_dir / config
        if not config_path.exists():
            # Try relative to project
            if not (repo_root / config).exists():
                return {"error": f"Config not found: {config}"}
            cmd_parts.extend(["--config", config])
        else:
            cmd_parts.extend(["--config", f"projects/{project}/{config}"])
    else:
        # Look for default config
        default_configs = list(project_dir.glob("configs/default.yaml")) + list(
            project_dir.glob("config.yaml")
        )
        if default_configs:
            rel_config = default_configs[0].relative_to(repo_root)
            cmd_parts.extend(["--config", str(rel_config)])

    # Device
    if device:
        cmd_parts.extend(["--device", device])

    # Resume
    if resume:
        cmd_parts.extend(["--resume", resume])

    # Overrides
    if overrides:
        for key, value in overrides.items():
            cmd_parts.extend([f"--{key}", str(value)])

    command = " ".join(cmd_parts)

    return {
        "command": command,
        "project": project,
        "config": config,
        "overrides": overrides,
        "explanation": f"Train {project} model" + (f" with {config}" if config else ""),
        "notes": [
            "Make sure your virtual environment is activated",
            "Check GPU availability with nvidia-smi before starting",
            "Consider using screen/tmux for long training runs",
        ],
    }
