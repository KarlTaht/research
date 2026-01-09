# Research Manager Agent - Project Guidance

## Project Purpose

A specialized agent for navigating and managing this ML research monorepo. Helps understand what experiments exist, which scripts to use, and how to organize research work.

**Key difference from literature research agent:** This agent operates on the *internal codebase*, not external papers.

## Quick Start

```bash
# From repo root
cd projects/research_manager_agent

# Install dependencies
uv pip install -e ".[dev]"

# Run the agent
python -m research_manager

# Run tests
pytest tests/
```

## Architecture

- **Claude Agent SDK** for agent loop and tool management (Phase 2+)
- **MCP `@tool` decorator** for tool definitions
- **Safety hooks** via PreToolUse/PostToolUse
- **Parquet + DuckDB** for experiment queries (uses common.utils)
- **Rich** for terminal UI
- **Pluggable memory** with session and persistent backends

## Current Status

**Phase 2: Codebase Tools - COMPLETE**

---

## Implementation Phases

### Phase 1: Foundation (COMPLETE)

Core infrastructure for the agent.

| Component | Status | Files |
|-----------|--------|-------|
| Project setup | Done | `pyproject.toml`, `README.md` |
| Core schemas | Done | `src/research_manager/schemas.py` |
| Memory interface | Done | `src/research_manager/memory/` |
| Tool registry | Done | `src/research_manager/tools/registry.py` |
| @tool decorator | Done | `src/research_manager/tools/decorators.py` |
| Repo indexer | Done | `src/research_manager/indexer.py` |
| Safety hooks | Done | `src/research_manager/safety/hooks.py` |
| Audit logging | Done | `src/research_manager/safety/audit.py` |
| Terminal UI | Done | `src/research_manager/ui/terminal.py` |
| Basic REPL | Done | `src/research_manager/main.py` |
| Unit tests | Done | `tests/unit/` (65 tests) |

### Phase 2: Codebase Tools (COMPLETE)

Tools for understanding the repository structure.

| Tool | Status | Purpose |
|------|--------|---------|
| `explore_repo` | Done | Navigate directory structure |
| `list_projects` | Done | List all projects with status |
| `get_project` | Done | Get details about specific project |
| `find_script` | Done | Find scripts by task description |
| `list_scripts` | Done | List all scripts |
| `read_config` | Done | Parse and explain YAML configs |
| `compare_configs` | Done | Diff two config files |
| `list_configs` | Done | List all configs |

**Skills created:**
- `codebase-navigation` - Repo navigation with CONVENTIONS.md
- `experiment-analysis` - Experiment queries with SQL_PATTERNS.md
- `training-assistant` - Training help with ERROR_PATTERNS.md

**Tests:** 114 unit tests passing

### Phase 3: Experiment Tools (NOT STARTED)

Tools for querying and analyzing experiments.

| Tool | Status | Purpose |
|------|--------|---------|
| `query_experiments` | Pending | Natural language + SQL queries |
| `analyze_logs` | Pending | Parse training logs |
| `find_checkpoint` | Pending | Locate model checkpoints |
| `compare_runs` | Pending | Side-by-side comparison |

**Skill:** `experiment-analysis` - Claude Agent Skill for experiment queries

### Phase 4: Assistant Tools (NOT STARTED)

Tools for interactive assistance.

| Tool | Status | Purpose |
|------|--------|---------|
| `run_command` | Pending | Generate and execute commands |
| `suggest_cleanup` | Pending | Identify organizational issues |
| `explain_error` | Pending | Context-aware error explanation |

**Skill:** `training-assistant` - Claude Agent Skill for training help

### Phase 5: Safety & Observability (PARTIAL)

Safety and monitoring infrastructure.

| Component | Status | Notes |
|-----------|--------|-------|
| Safety hooks | Done | Blocks dangerous commands |
| Audit logging | Done | Tracks all tool calls |
| Terminal UI | Done | Rich-based interface |
| Integration tests | Pending | Tool combinations |
| E2E tests | Pending | Full agent scenarios |

### Phase 6: Polish & Integration (NOT STARTED)

Final polish and full integration.

| Component | Status | Notes |
|-----------|--------|-------|
| Claude Agent SDK integration | Pending | Full agent loop |
| Skills framework | Pending | SKILL.md files |
| Benchmarks | Pending | Performance metrics |
| Regression tracking | Pending | Eval over time |
| Documentation | Partial | README done |

---

## Key Files

### Source Code
- `src/research_manager/main.py` - Agent entry point and REPL
- `src/research_manager/schemas.py` - Project, Experiment, Script, Config
- `src/research_manager/indexer.py` - Repository structure indexer
- `src/research_manager/tools/` - Tool registry and decorators
- `src/research_manager/memory/` - Pluggable memory backends
- `src/research_manager/safety/` - Safety hooks and audit logging
- `src/research_manager/ui/` - Rich terminal UI

### Documentation
- `IMPLEMENTATION_PLAN.md` - Original detailed implementation plan
- `README.md` - User-facing documentation

### Tests
- `tests/unit/` - Unit tests (65 tests)
- `tests/conftest.py` - Pytest fixtures

---

## Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| Codebase | `explore_repo`, `find_script`, `read_config`, `list_projects` | Understand what exists |
| Experiments | `query_experiments`, `analyze_logs`, `compare_runs`, `find_checkpoint` | Know what's been done |
| Assistant | `run_command`, `suggest_cleanup`, `explain_error` | Help do things |

---

## Memory Architecture

Pluggable memory system with three backends:

1. **SessionMemory** - In-memory, resets each session
2. **PersistentMemory** - JSON file, survives restarts
3. **KnowledgeGraphMemory** - Placeholder for future RAG system

```python
from research_manager.memory import SessionMemory, PersistentMemory

# Session only
memory = SessionMemory()

# Persistent across sessions
memory = PersistentMemory(Path("~/.research_manager/memory.json"))
```

---

## Safety Principles

1. **Never execute dangerous commands** - rm -rf, force push, sudo
2. **Show before execute** - Always display command before running
3. **Confirm risky operations** - Training, large downloads require confirmation
4. **Audit everything** - All tool calls logged for review

### Blocked Patterns
- `rm -rf /`, `rm -rf ~`, `rm -rf *`
- `git push --force`, `git push -f`
- `sudo`, `chmod 777`
- Writes to `/dev/`

### Confirmation Required
- `python train.py` (training runs)
- `pip install`, `uv pip install`
- `git reset`
- Any `rm` or `mv` command

---

## Integration Points

Uses these from the monorepo:
- `common.utils.experiment_storage` - Query experiments via Parquet/DuckDB
- `common.data.hf_utils` - Dataset path resolution
- YAML configs in `projects/*/configs/`
- Checkpoints in `assets/outputs/checkpoints/`

---

## Development

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/ -v

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check (when ready)
mypy src/
```

---

## Next Steps

1. **Phase 2**: Implement codebase tools (`explore_repo`, `list_projects`, etc.)
2. **Skills**: Create `.claude/skills/` with SKILL.md files
3. **Claude SDK**: Integrate full agent loop with Claude Agent SDK
4. **Eval Framework**: Add integration tests, E2E scenarios, benchmarks
