# Research Manager Agent

A specialized agent for navigating and managing ML research monorepos. Unlike a literature research agent that searches external papers, this agent understands **your codebase**, **your experiments**, and **your workflow**.

## Features

- **Codebase Navigation**: Explore repository structure, find scripts, read configs
- **Experiment Intelligence**: Query experiments, analyze logs, find checkpoints
- **Interactive Assistance**: Generate commands, suggest cleanup, explain errors
- **Safety First**: Blocked dangerous commands, confirmation for risky operations

## Installation

```bash
cd projects/research_repository_agent
uv pip install -e ".[dev]"
```

## Usage

```bash
# Start the interactive agent
research-repository

# Or run directly
python -m research_repository
```

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/ tests/

# Type check
mypy src/
```

## Architecture

See `IMPLEMENTATION_PLAN.md` for detailed architecture documentation.
