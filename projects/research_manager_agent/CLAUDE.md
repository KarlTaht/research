# Research Manager Agent - Project Guidance

## Project Purpose

A specialized agent for navigating and managing this ML research monorepo. Helps understand what experiments exist, which scripts to use, and how to organize research work.

**Key difference from literature research agent:** This agent operates on the *internal codebase*, not external papers.

## Quick Start

```bash
# From repo root
cd projects/research_manager_agent

# Install dependencies (when implemented)
uv pip install -e ".[dev]"

# Run the agent
python -m research_manager
```

## Architecture

- **Claude Agent SDK** for agent loop and tool management
- **MCP `@tool` decorator** for tool definitions
- **Safety hooks** via PreToolUse/PostToolUse
- **Parquet + DuckDB** for experiment queries (uses common.utils)
- **Rich** for terminal UI

## Tool Categories

| Category | Tools | Purpose |
|----------|-------|---------|
| Codebase | `explore_repo`, `find_script`, `read_config`, `list_projects` | Understand what exists |
| Experiments | `query_experiments`, `analyze_logs`, `compare_runs`, `find_checkpoint` | Know what's been done |
| Assistant | `run_command`, `suggest_cleanup`, `explain_error` | Help do things |

## Key Files

- `IMPLEMENTATION_PLAN.md` - Detailed phased implementation plan
- `src/research_manager/main.py` - Agent entry point
- `src/research_manager/tools/` - All tool implementations
- `src/research_manager/safety/hooks.py` - Safety guardrails

## Integration Points

Uses these from the monorepo:
- `common.utils.experiment_storage` - Query experiments via Parquet/DuckDB
- `common.data.hf_utils` - Dataset path resolution
- YAML configs in `projects/*/configs/`
- Checkpoints in `assets/outputs/checkpoints/`

## Safety Principles

1. **Never execute dangerous commands** - rm -rf, force push, sudo
2. **Show before execute** - Always display command before running
3. **Confirm risky operations** - Training, large downloads require confirmation
4. **Audit everything** - All tool calls logged for review

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/

# Type check
mypy src/
```

## Current Status

**Phase: Planning Complete**

See `IMPLEMENTATION_PLAN.md` for the 6-phase implementation roadmap.
