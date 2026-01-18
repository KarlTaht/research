# Research Manager Agent - Implementation Plan

## Overview

A specialized agent for navigating and managing an ML research monorepo. Unlike a literature research agent that searches external papers, this agent understands **your codebase**, **your experiments**, and **your workflow**.

### Core Value Proposition

| Problem | Solution |
|---------|----------|
| "What experiments have I already run?" | Query experiment database with natural language |
| "Which script do I use to train a model?" | Discover and explain relevant scripts |
| "What does this config do?" | Parse and explain YAML configurations |
| "My repo is a mess" | Analyze structure, suggest cleanup |
| "How do I run X?" | Generate and explain commands |

### What This Agent Is NOT

- **Not a literature agent**: Doesn't search Semantic Scholar or arXiv
- **Not a coding agent**: Doesn't write new code (that's Claude Code's job)
- **Not a training agent**: Doesn't run long training jobs autonomously

### Target Interaction Pattern

```
You: "What transformer experiments have I run with perplexity under 20?"

Agent: [Uses query_experiments tool]
       "I found 3 experiments matching your criteria:
        1. tinystories_10m_full_3ep (perplexity: 15.2, 2024-12-01)
        2. tinystories_custom_tok (perplexity: 18.7, 2024-11-28)
        3. transformer_debug_v2 (perplexity: 19.1, 2024-11-25)

        The best performing one used config: configs/tinystories_10m_full.yaml
        with d_model=288, n_blocks=8. Want me to show the full config?"

You: "Yes, and how do I resume training from that checkpoint?"

Agent: [Uses read_config + find_script tools]
       "Here's the config breakdown: [explains each section]

        To resume training, run:
        python train.py --config configs/tinystories_10m_full.yaml --resume

        The checkpoint is at: assets/outputs/checkpoints/tinystories_10m_full_3ep/"
```

---

## Architecture

### Why Claude Agent SDK

Using the SDK (not custom ReAct loop) because:
1. **Built-in agent loop** - think→act→observe handled automatically
2. **`@tool` decorator** - Clean MCP tool definitions
3. **Safety hooks** - `PreToolUse`/`PostToolUse` for guardrails
4. **Cost tracking** - Built-in token/cost metrics
5. **Focus on domain logic** - Not agent infrastructure

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (You)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Research Manager Agent                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Claude Agent SDK                       │  │
│  │              (agent loop, message handling)               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │  Codebase Tools │ │ Experiment Tools│ │ Assistant Tools │   │
│  │                 │ │                 │ │                 │   │
│  │ - explore_repo  │ │ - query_exp     │ │ - run_command   │   │
│  │ - find_script   │ │ - analyze_logs  │ │ - suggest_clean │   │
│  │ - read_config   │ │ - compare_runs  │ │ - explain_error │   │
│  │ - list_projects │ │ - find_ckpt     │ │                 │   │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘   │
│           │                   │                   │             │
│           └───────────────────┼───────────────────┘             │
│                               ▼                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Safety Layer                           │  │
│  │         (PreToolUse hooks, audit logging)                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Monorepo Filesystem                          │
│                                                                 │
│  projects/    common/    assets/    tools/    configs/          │
│  (code)       (shared)   (data)     (CLIs)    (YAML)           │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| **Codebase** | `explore_repo`, `find_script`, `read_config`, `list_projects` | Understand what exists |
| **Experiments** | `query_experiments`, `analyze_logs`, `compare_runs`, `find_checkpoint` | Know what's been done |
| **Assistant** | `run_command`, `suggest_cleanup`, `explain_error` | Help do things |

---

## Phase 1: Foundation

### Step 1: Project Setup

**Files to create:**
```
projects/research_repository_agent/
├── pyproject.toml
├── src/
│   └── research_repository/
│       ├── __init__.py
│       └── main.py
└── tests/
    └── __init__.py
```

**Dependencies:**
```toml
[project]
name = "research-repository-agent"
version = "0.1.0"
dependencies = [
    "claude-code-sdk",      # Agent SDK (check actual package name)
    "rich",                 # Terminal UI
    "pyyaml",               # Config parsing
    "duckdb",               # Experiment queries
    "pandas",               # Data manipulation
    "pyarrow",              # Parquet support
]

[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]
```

**Note:** Verify the exact Claude Agent SDK package name before implementation.

### Step 2: Core Schemas

**File:** `src/research_repository/schemas.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

@dataclass
class Project:
    """Represents a project or paper implementation in the monorepo."""
    name: str
    path: Path
    type: str                           # "project" | "paper" | "archive"
    description: Optional[str] = None
    has_train_script: bool = False
    has_eval_script: bool = False
    config_files: list[str] = field(default_factory=list)
    last_modified: Optional[datetime] = None

@dataclass
class Experiment:
    """Represents a tracked experiment run."""
    name: str
    project: str
    config_path: Optional[Path] = None
    metrics: dict = field(default_factory=dict)  # perplexity, loss, etc.
    timestamp: Optional[datetime] = None
    checkpoint_path: Optional[Path] = None
    notes: Optional[str] = None

@dataclass
class Script:
    """Represents a runnable script in the repo."""
    name: str
    path: Path
    purpose: str                        # train, evaluate, download, etc.
    project: Optional[str] = None
    arguments: list[str] = field(default_factory=list)
    example_command: Optional[str] = None

@dataclass
class Config:
    """Represents a YAML configuration file."""
    path: Path
    project: str
    sections: dict                      # parsed YAML content
    model_params: Optional[dict] = None
    training_params: Optional[dict] = None
    data_params: Optional[dict] = None
```

### Step 3: Repo Index Builder

**File:** `src/research_repository/indexer.py`

Build a lightweight index of the monorepo structure on startup:

```python
class RepoIndexer:
    """Indexes the monorepo structure for fast lookups."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.projects: dict[str, Project] = {}
        self.scripts: dict[str, Script] = {}
        self.configs: dict[str, Config] = {}

    def index_projects(self) -> list[Project]:
        """Discover all projects in projects/ and papers/ directories."""
        ...

    def index_scripts(self) -> list[Script]:
        """Find all Python scripts and categorize by purpose."""
        ...

    def index_configs(self) -> list[Config]:
        """Parse all YAML config files."""
        ...

    def refresh(self):
        """Re-index the repo (call when files change)."""
        ...
```

**Key discovery logic:**
- Projects: Look in `projects/*/` and `papers/*/`, skip `archive/`
- Scripts: Find `*.py` files, categorize by name (`train.py`, `evaluate.py`, etc.)
- Configs: Find `*.yaml` and `*.yml` in `configs/` subdirectories

### Step 4: Basic Agent Runner

**File:** `src/research_repository/main.py`

```python
from claude_code_sdk import ClaudeCodeSDK, query  # Verify actual API

async def main():
    """Run the research manager agent."""

    # Index the repo on startup
    indexer = RepoIndexer(Path.cwd())
    indexer.refresh()

    # Create agent with tools
    async with ClaudeCodeSDK() as sdk:
        # Register tools (Phase 2)
        # Start interactive loop
        ...

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

**Milestone 1:** Project structure created, schemas defined, basic indexer working

---

## Phase 2: Codebase Understanding Tools

### Step 5: Explore Repo Tool

**File:** `src/research_repository/tools/explore.py`

```python
@tool(
    name="explore_repo",
    description="Explore the repository structure. Returns directory tree and key files.",
    parameters={
        "path": {"type": "string", "description": "Subdirectory to explore (default: root)"},
        "depth": {"type": "integer", "description": "Max depth (default: 2)"}
    }
)
async def explore_repo(path: str = ".", depth: int = 2) -> dict:
    """
    Returns structured info about a directory:
    - Subdirectories with descriptions (if README exists)
    - Key files (train.py, config.yaml, etc.)
    - Git status (modified files)
    """
    ...
```

**Output example:**
```json
{
    "path": "projects/custom_transformer",
    "type": "project",
    "description": "Custom transformer with manual backprop",
    "contents": {
        "scripts": ["train.py", "evaluate.py", "validate.py"],
        "configs": ["configs/tinystories_10m_full.yaml", "configs/tinystories_custom_tokenizer.yaml"],
        "tests": ["tests/test_loss_decreases.py"],
        "symlinks": {"model": "../../common/models/custom_transfromer"}
    },
    "git_status": "clean"
}
```

### Step 6: List Projects Tool

**File:** `src/research_repository/tools/projects.py`

```python
@tool(
    name="list_projects",
    description="List all projects/papers in the repo with their status and purpose.",
    parameters={
        "include_archived": {"type": "boolean", "description": "Include archived projects"}
    }
)
async def list_projects(include_archived: bool = False) -> list[dict]:
    """
    Returns all projects with:
    - Name and path
    - Description (from README or CLAUDE.md)
    - Status (active, archived)
    - Available scripts
    - Recent experiments (if any)
    """
    ...
```

### Step 7: Find Script Tool

**File:** `src/research_repository/tools/scripts.py`

```python
@tool(
    name="find_script",
    description="Find the right script for a task. Describe what you want to do.",
    parameters={
        "task": {"type": "string", "description": "What you want to do (e.g., 'train a model', 'download dataset')"},
        "project": {"type": "string", "description": "Specific project to search in (optional)"}
    }
)
async def find_script(task: str, project: str = None) -> list[dict]:
    """
    Semantic search over scripts to find relevant ones.
    Returns scripts with:
    - Path and purpose
    - Example usage command
    - Required arguments
    """
    ...
```

**Task → Script mapping examples:**
- "train a model" → `train.py`, `projects/*/train.py`
- "download dataset" → `tools/download_hf_dataset.py`
- "check experiments" → `tools/query_experiments.py`
- "evaluate model" → `evaluate.py`, `validate.py`

### Step 8: Read Config Tool

**File:** `src/research_repository/tools/configs.py`

```python
@tool(
    name="read_config",
    description="Read and explain a YAML configuration file.",
    parameters={
        "path": {"type": "string", "description": "Path to config file"},
        "section": {"type": "string", "description": "Specific section to focus on (optional)"}
    }
)
async def read_config(path: str, section: str = None) -> dict:
    """
    Parse config and return:
    - Full parsed content
    - Explanation of each section
    - Computed values (total params, effective batch size, etc.)
    """
    ...

@tool(
    name="compare_configs",
    description="Compare two configuration files and show differences.",
    parameters={
        "config1": {"type": "string", "description": "First config path"},
        "config2": {"type": "string", "description": "Second config path"}
    }
)
async def compare_configs(config1: str, config2: str) -> dict:
    """Show what's different between two configs."""
    ...
```

**Milestone 2:** Agent can explore repo, list projects, find scripts, read configs

---

## Phase 3: Experiment Intelligence

### Step 9: Query Experiments Tool

**File:** `src/research_repository/tools/experiments.py`

```python
@tool(
    name="query_experiments",
    description="Query experiment results using natural language or SQL.",
    parameters={
        "query": {"type": "string", "description": "Natural language query or SQL"},
        "format": {"type": "string", "description": "Output format: 'table' | 'summary' | 'json'"}
    }
)
async def query_experiments(query: str, format: str = "summary") -> dict:
    """
    Interface to the Parquet + DuckDB experiment storage.

    Natural language examples:
    - "best perplexity" → SELECT * ORDER BY perplexity LIMIT 5
    - "experiments from last week" → SELECT * WHERE saved_at > ...
    - "compare exp_001 and exp_002" → side-by-side comparison

    Also accepts raw SQL for power users.
    """
    # Use common.utils.experiment_storage under the hood
    from common.utils import query_experiments as raw_query
    ...
```

### Step 10: Analyze Logs Tool

**File:** `src/research_repository/tools/logs.py`

```python
@tool(
    name="analyze_logs",
    description="Analyze training logs to understand what happened during a run.",
    parameters={
        "experiment": {"type": "string", "description": "Experiment name or log file path"},
        "focus": {"type": "string", "description": "What to focus on: 'loss', 'errors', 'warnings', 'summary'"}
    }
)
async def analyze_logs(experiment: str, focus: str = "summary") -> dict:
    """
    Parse training logs and extract:
    - Loss curve summary (start, end, min, convergence point)
    - Errors and warnings
    - Training speed (samples/sec, time per epoch)
    - Resource usage (if logged)
    - Anomalies (loss spikes, NaN, etc.)
    """
    ...
```

### Step 11: Find Checkpoint Tool

**File:** `src/research_repository/tools/checkpoints.py`

```python
@tool(
    name="find_checkpoint",
    description="Find model checkpoints and their metadata.",
    parameters={
        "experiment": {"type": "string", "description": "Experiment name (optional)"},
        "best_by": {"type": "string", "description": "Metric to rank by (optional)"}
    }
)
async def find_checkpoint(experiment: str = None, best_by: str = None) -> list[dict]:
    """
    Search for checkpoints in assets/outputs/checkpoints/.
    Returns:
    - Path to checkpoint
    - Associated config
    - Metrics at save time
    - Model architecture summary
    """
    ...
```

### Step 12: Compare Runs Tool

**File:** `src/research_repository/tools/compare.py`

```python
@tool(
    name="compare_runs",
    description="Compare multiple experiment runs side by side.",
    parameters={
        "experiments": {"type": "array", "items": {"type": "string"}, "description": "List of experiment names"},
        "metrics": {"type": "array", "items": {"type": "string"}, "description": "Metrics to compare"}
    }
)
async def compare_runs(experiments: list[str], metrics: list[str] = None) -> dict:
    """
    Side-by-side comparison showing:
    - Config differences
    - Metric comparisons
    - Training time differences
    - What changed between runs
    """
    ...
```

**Milestone 3:** Agent can query experiments, analyze logs, find checkpoints, compare runs

---

## Phase 4: Interactive Assistance

### Step 13: Run Command Tool

**File:** `src/research_repository/tools/commands.py`

```python
@tool(
    name="run_command",
    description="Generate and optionally execute a command. Always shows command before running.",
    parameters={
        "intent": {"type": "string", "description": "What you want to do"},
        "execute": {"type": "boolean", "description": "Actually run the command (default: false, just show it)"},
        "project": {"type": "string", "description": "Project context (optional)"}
    }
)
async def run_command(intent: str, execute: bool = False, project: str = None) -> dict:
    """
    Generate command from intent:
    - "train the 10m model" → python train.py --config configs/tinystories_10m_full.yaml
    - "download tinystories" → python tools/download_hf_dataset.py --name roneneldan/TinyStories
    - "check GPU memory" → nvidia-smi

    Safety:
    - Always show command before executing
    - Block dangerous commands (rm -rf, etc.)
    - Require confirmation for long-running commands
    """
    ...
```

**Safety rules:**
- Never execute: `rm -rf`, `git push --force`, anything with `sudo`
- Require confirmation: Training commands, downloads > 1GB
- Auto-approve: Status commands, queries, non-destructive operations

### Step 14: Suggest Cleanup Tool

**File:** `src/research_repository/tools/cleanup.py`

```python
@tool(
    name="suggest_cleanup",
    description="Analyze the repo and suggest organizational improvements.",
    parameters={
        "scope": {"type": "string", "description": "What to analyze: 'all', 'checkpoints', 'experiments', 'code'"}
    }
)
async def suggest_cleanup(scope: str = "all") -> dict:
    """
    Identify organizational issues:

    Checkpoints:
    - Old checkpoints without associated experiments
    - Duplicate checkpoints (same config, similar metrics)
    - Large checkpoints that could be pruned

    Experiments:
    - Failed/incomplete runs
    - Experiments without notes
    - Duplicate experiments

    Code:
    - Unused scripts
    - Projects without recent activity
    - Missing documentation (no README)
    - Broken symlinks

    Returns prioritized suggestions with commands to fix.
    """
    ...
```

### Step 15: Explain Error Tool

**File:** `src/research_repository/tools/errors.py`

```python
@tool(
    name="explain_error",
    description="Explain an error message and suggest fixes in the context of this repo.",
    parameters={
        "error": {"type": "string", "description": "Error message or traceback"},
        "context": {"type": "string", "description": "What you were trying to do"}
    }
)
async def explain_error(error: str, context: str = None) -> dict:
    """
    Parse error and provide repo-specific guidance:
    - Common CUDA errors → suggest nvidia-smi, batch size reduction
    - Import errors → check virtual environment, suggest uv pip install
    - Config errors → point to correct config format
    - Path errors → suggest correct asset paths
    """
    ...
```

**Milestone 4:** Agent can generate commands, suggest cleanup, explain errors

---

## Phase 5: Safety & Observability

### Step 16: Safety Hooks

**File:** `src/research_repository/safety/hooks.py`

```python
from claude_code_sdk import PreToolUse, PostToolUse  # Verify actual API

# Commands that should never be executed
BLOCKED_PATTERNS = [
    r"rm\s+-rf",
    r"git\s+push.*--force",
    r"sudo\s+",
    r">\s*/dev/",
    r"chmod\s+777",
]

# Commands that require confirmation
CONFIRM_PATTERNS = [
    r"python.*train\.py",      # Training runs
    r"download.*--name",        # Large downloads
    r"git\s+reset",             # Git operations
]

async def pre_tool_hook(tool_name: str, tool_input: dict) -> dict:
    """Validate tool calls before execution."""

    if tool_name == "run_command" and tool_input.get("execute"):
        command = tool_input.get("generated_command", "")

        # Block dangerous commands
        for pattern in BLOCKED_PATTERNS:
            if re.search(pattern, command):
                return {"decision": "deny", "reason": f"Blocked: matches {pattern}"}

        # Require confirmation for risky commands
        for pattern in CONFIRM_PATTERNS:
            if re.search(pattern, command):
                return {"decision": "confirm", "reason": "Requires user confirmation"}

    return {"decision": "allow"}
```

### Step 17: Audit Logging

**File:** `src/research_repository/safety/audit.py`

```python
@dataclass
class AuditEntry:
    timestamp: datetime
    tool_name: str
    tool_input: dict
    tool_output: dict
    decision: str              # allow, deny, confirm
    user_confirmed: bool
    execution_time_ms: float

class AuditLog:
    """Track all agent actions for review."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.entries: list[AuditEntry] = []

    def log_action(self, entry: AuditEntry):
        """Log an action to the audit trail."""
        self.entries.append(entry)
        self._persist()

    def get_session_summary(self) -> dict:
        """Summary of this session's actions."""
        ...

    def export_to_parquet(self):
        """Export audit log to Parquet for long-term storage."""
        ...
```

### Step 18: Terminal UI

**File:** `src/research_repository/ui/terminal.py`

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

class TerminalUI:
    """Rich-based terminal interface."""

    def __init__(self):
        self.console = Console()

    def show_thinking(self, thought: str):
        """Display agent's reasoning."""
        self.console.print(Panel(thought, title="Thinking", border_style="blue"))

    def show_tool_call(self, tool_name: str, args: dict):
        """Display tool invocation."""
        ...

    def show_results(self, results: dict):
        """Display tool results (tables, lists, etc.)."""
        ...

    def confirm_action(self, action: str, command: str) -> bool:
        """Prompt user to confirm risky action."""
        ...

    def show_cost(self, tokens: int, cost_usd: float):
        """Display token usage and cost."""
        ...
```

**Milestone 5:** Agent has safety guardrails, audit logging, nice terminal UI

---

## Phase 6: Polish & Integration

### Step 19: Session Memory

**File:** `src/research_repository/memory/session.py`

```python
class SessionMemory:
    """Remember context within a session."""

    def __init__(self):
        self.current_project: Optional[str] = None
        self.recent_experiments: list[str] = []
        self.recent_configs: list[str] = []
        self.conversation_context: list[dict] = []

    def set_project_context(self, project: str):
        """Set the current project for implicit scoping."""
        ...

    def add_to_context(self, key: str, value: any):
        """Add information that might be useful later."""
        ...

    def get_relevant_context(self, query: str) -> dict:
        """Retrieve context relevant to the current query."""
        ...
```

### Step 20: Claude Desktop Integration

**File:** `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "research-repository": {
      "command": "python",
      "args": ["-m", "research_repository.server"],
      "cwd": "/Users/ktaht/Learning/research"
    }
  }
}
```

This allows using the tools from Claude Desktop as well as the CLI.

### Step 21: End-to-End Testing

**File:** `tests/test_e2e.py`

Test scenarios:
1. "List all projects" → Returns correct project list
2. "What experiments used perplexity < 20?" → Correct SQL query
3. "How do I train the transformer?" → Returns train.py with correct args
4. "Compare config A and config B" → Shows meaningful diff
5. "rm -rf /" with execute=True → Blocked by safety

### Step 22: Documentation

- `README.md` - Usage guide with examples
- `ARCHITECTURE.md` - Design decisions
- Inline docstrings for all tools

**Milestone 6:** Production-ready agent with full documentation

---

## File Summary

```
projects/research_repository_agent/
├── IMPLEMENTATION_PLAN.md          # This file
├── CLAUDE.md                       # Project guidance
├── README.md                       # Usage documentation
├── pyproject.toml                  # Dependencies
├── claude_desktop_config.json      # Claude Desktop integration
│
├── src/
│   └── research_repository/
│       ├── __init__.py
│       ├── main.py                 # CLI entry point
│       ├── server.py               # MCP server for Claude Desktop
│       ├── schemas.py              # Project, Experiment, Script, Config
│       ├── indexer.py              # Repo structure indexer
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── explore.py          # explore_repo
│       │   ├── projects.py         # list_projects
│       │   ├── scripts.py          # find_script
│       │   ├── configs.py          # read_config, compare_configs
│       │   ├── experiments.py      # query_experiments
│       │   ├── logs.py             # analyze_logs
│       │   ├── checkpoints.py      # find_checkpoint
│       │   ├── compare.py          # compare_runs
│       │   ├── commands.py         # run_command
│       │   ├── cleanup.py          # suggest_cleanup
│       │   └── errors.py           # explain_error
│       │
│       ├── safety/
│       │   ├── __init__.py
│       │   ├── hooks.py            # PreToolUse/PostToolUse
│       │   └── audit.py            # Action logging
│       │
│       ├── memory/
│       │   ├── __init__.py
│       │   └── session.py          # Session context
│       │
│       └── ui/
│           ├── __init__.py
│           └── terminal.py         # Rich terminal UI
│
└── tests/
    ├── __init__.py
    ├── test_indexer.py
    ├── test_tools.py
    ├── test_safety.py
    └── test_e2e.py
```

---

## Implementation Order

| Order | Step | Description | Dependencies |
|-------|------|-------------|--------------|
| 1 | Step 1-4 | Foundation: setup, schemas, indexer, runner | None |
| 2 | Step 5-6 | `explore_repo`, `list_projects` | Indexer |
| 3 | Step 7-8 | `find_script`, `read_config` | Indexer |
| 4 | Step 9 | `query_experiments` | common.utils |
| 5 | Step 10-12 | `analyze_logs`, `find_checkpoint`, `compare_runs` | Experiments tool |
| 6 | Step 13-15 | `run_command`, `suggest_cleanup`, `explain_error` | All above |
| 7 | Step 16-18 | Safety hooks, audit log, terminal UI | All tools |
| 8 | Step 19-22 | Memory, integration, testing, docs | Everything |

---

## Open Questions

1. **SDK API**: Need to verify exact Claude Agent SDK package name and API shape
2. **Multi-model routing**: Use Haiku for cheap operations? (listing, queries) vs Sonnet for synthesis
3. **Persistence**: Should agent memory persist across sessions?
4. **Natural language → SQL**: Use LLM for translation or pattern matching?
5. **Remote access**: Should agent work over SSH? (for Lambda instances)

---

## Success Criteria

The agent is successful when you can:

1. Ask "what experiments have I run?" and get accurate, queryable results
2. Ask "how do I train X?" and get the exact command to run
3. Ask "what's different between these configs?" and get meaningful comparison
4. Ask "clean up my checkpoints" and get actionable suggestions
5. Trust the agent won't delete your data or run dangerous commands
6. Use it daily without friction

---

## Cost Target

Target: **< $0.50 per interactive session**

Strategy:
- Use Haiku for listing/querying operations
- Use Sonnet for synthesis and explanations
- Cache repo index (don't re-scan every query)
- Batch related tool calls where possible
