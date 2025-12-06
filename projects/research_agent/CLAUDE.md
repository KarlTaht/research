# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research assistant agent that reads papers, maintains evolving hypotheses, and synthesizes research findings over long horizons. Part of the `~/research/` ML monorepo.

## Commands

```bash
# From monorepo root - activate environment first
source .venv/bin/activate

# Install project dependencies (once created)
uv pip install -e "projects/research_agent[dev]"

# Run the agent (planned)
python -m research_agent.main --query "topic to research"

# Run tests
pytest projects/research_agent/tests/

# Format and lint
black projects/research_agent/
ruff check projects/research_agent/
```

## Architecture

### Target Structure
```
research_agent/
├── agent/
│   ├── core.py          # ReAct loop implementation
│   ├── router.py        # LLM routing (Haiku/Sonnet/Opus selection)
│   └── tools/           # MCP tool implementations
│       ├── paper_search.py    # Semantic Scholar API
│       ├── paper_reader.py    # PDF extraction
│       ├── knowledge_ops.py   # Graph operations
│       └── hypothesis.py      # Hypothesis management
├── memory/
│   ├── knowledge_graph.py    # NetworkX graph + SQLite persistence
│   ├── embeddings.py         # sentence-transformers for semantic search
│   └── schemas.py            # Paper, Hypothesis, Entity dataclasses
├── safety/
│   ├── actions.py       # ALLOWED_ACTIONS / FORBIDDEN_ACTIONS whitelist
│   ├── autonomy.py      # Tiered approval (auto vs human-confirm)
│   └── audit.py         # Action logging + rollback
├── ui/
│   ├── terminal.py      # Rich-based live view
│   └── web/             # FastAPI + React (stretch goal)
└── main.py
```

### Key Design Decisions

**ReAct Loop**: Custom implementation, not a framework. Keep it simple and debuggable.

**Multi-Model Routing**:
- Haiku: Summarization, entity extraction (cheap, fast)
- Sonnet: Hypothesis generation, contradiction detection (reasoning)
- Opus: Final synthesis reports (sparingly, expensive)

**Memory System**:
- Papers, hypotheses, entities stored in SQLite
- NetworkX for relationship traversal
- sentence-transformers embeddings for semantic search

**Safety Layers**:
1. Action whitelist (only allowed operations execute)
2. Tiered autonomy (small changes auto-approve, large changes require confirmation)
3. Full audit log with rollback capability

### Knowledge Schema

```python
# Papers: title, summary, key_findings, contradicts/supports lists, embedding
# Hypotheses: statement, confidence (0-1), supporting/contradicting evidence, history
# Entities: name, type (method/concept/researcher), mentioned_in, related_to
```

## Integration Points

### Monorepo Common Package
```python
# Use shared utilities from common/
from common.utils import save_experiment, query_experiments
from common.data import get_datasets_dir
```

### External APIs
- **Semantic Scholar**: Paper search and metadata
- **arXiv**: Paper PDFs and abstracts
- **Anthropic API**: Claude models (requires ANTHROPIC_API_KEY)

### MCP Tools
Custom MCP server exposing:
- `search_papers`: Query Semantic Scholar
- `read_paper`: Extract text from PDF
- `add_hypothesis`: Create new hypothesis
- `update_confidence`: Adjust hypothesis confidence
- `query_graph`: Search knowledge graph

## Key Constraints

- Confidence shifts > 0.3 require human confirmation
- Destructive operations (delete, drop, clear) always blocked
- Every action logged with before/after state for rollback
- Target cost < $1 per research question (use Haiku aggressively)
