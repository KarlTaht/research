# Research Manager Agent - Project Guidance

## Project Purpose

A specialized agent for navigating and managing this ML research monorepo. Helps understand what experiments exist, which scripts to use, and how to organize research work.

**Key difference from literature research agent:** This agent operates on the *internal codebase*, not external papers.

## Quick Start

```bash
# From repo root
cd projects/research_repository_agent

# Install dependencies
uv pip install -e ".[dev]"

# Run the agent
python -m research_repository

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

**Phase 5: Safety & Observability - COMPLETE**

---

## Implementation Phases

### Phase 1: Foundation (COMPLETE)

Core infrastructure for the agent.

| Component | Status | Files |
|-----------|--------|-------|
| Project setup | Done | `pyproject.toml`, `README.md` |
| Core schemas | Done | `src/research_repository/schemas.py` |
| Memory interface | Done | `src/research_repository/memory/` |
| Tool registry | Done | `src/research_repository/tools/registry.py` |
| @tool decorator | Done | `src/research_repository/tools/decorators.py` |
| Repo indexer | Done | `src/research_repository/indexer.py` |
| Safety hooks | Done | `src/research_repository/safety/hooks.py` |
| Audit logging | Done | `src/research_repository/safety/audit.py` |
| Terminal UI | Done | `src/research_repository/ui/terminal.py` |
| Basic REPL | Done | `src/research_repository/main.py` |
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

### Phase 3: Experiment Tools (COMPLETE)

Tools for querying and analyzing experiments.

| Tool | Status | Purpose |
|------|--------|---------|
| `query_experiments` | Done | Natural language + SQL queries |
| `list_experiments` | Done | List available experiments |
| `analyze_logs` | Done | Parse training logs |
| `find_checkpoint` | Done | Locate model checkpoints |
| `compare_runs` | Done | Side-by-side comparison |

**Features:**
- Natural language query conversion (e.g., "best perplexity" → SQL)
- Log metric extraction (loss, perplexity, accuracy)
- Error and warning detection in logs
- Checkpoint file discovery with metadata
- Multi-experiment comparison with best performer identification

**Tests:** 150 unit tests passing (36 new experiment tests)

### Phase 4: Assistant Tools (COMPLETE)

Tools for interactive assistance.

| Tool | Status | Purpose |
|------|--------|---------|
| `run_command` | Done | Execute commands with safety checks |
| `suggest_cleanup` | Done | Find old checkpoints, logs, large files |
| `explain_error` | Done | Context-aware error explanation |
| `generate_train_command` | Done | Generate training commands |

**Features:**
- Command execution with safety hooks (blocks dangerous commands)
- Cleanup suggestions for checkpoints, logs, cache, large files
- Error pattern matching for 10+ common ML errors (CUDA OOM, NaN, etc.)
- Traceback parsing with file/line extraction
- Training command generation with config overrides

**Tests:** 183 unit tests passing (33 new assistant tests)

### Phase 5: Safety & Observability (COMPLETE)

Safety, monitoring, and test infrastructure.

| Component | Status | Notes |
|-----------|--------|-------|
| Safety hooks | Done | Blocks dangerous commands |
| Audit logging | Done | Tracks all tool calls |
| Terminal UI | Done | Rich-based interface |
| Integration tests | Done | 6 tool combination workflows |
| E2E tests | Done | 10 complete user scenarios |

**Integration Tests (6 workflows):**
- Explore → Train workflow
- Experiment analysis workflow
- Error debugging workflow
- Cleanup workflow
- Script discovery workflow
- Config comparison workflow

**E2E Scenarios (10 tests):**
- New user onboarding
- Researcher analyzing experiments
- Debugging failed experiments
- Preparing new experiments
- Safety blocking dangerous commands
- Audit logging verification
- Cleanup suggestions
- Error handling (missing projects, invalid configs, complex errors)

**Tests:** 199 total (183 unit + 16 integration/E2E)

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
- `src/research_repository/main.py` - Agent entry point and REPL
- `src/research_repository/schemas.py` - Project, Experiment, Script, Config
- `src/research_repository/indexer.py` - Repository structure indexer
- `src/research_repository/tools/` - Tool registry and decorators
- `src/research_repository/memory/` - Pluggable memory backends
- `src/research_repository/safety/` - Safety hooks and audit logging
- `src/research_repository/ui/` - Rich terminal UI

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
from research_repository.memory import SessionMemory, PersistentMemory

# Session only
memory = SessionMemory()

# Persistent across sessions
memory = PersistentMemory(Path("~/.research_repository/memory.json"))
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

1. **Phase 6**: Claude Agent SDK integration, benchmarks, regression tracking
