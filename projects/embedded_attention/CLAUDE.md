# Embedded Attention Project

## Overview

Chunk-based conversational memory system using RAG (Retrieval-Augmented Generation). Each conversation turn is encoded, stored in DuckDB with HNSW vector indexing, and dynamically retrieved to augment the LLM's context window.

## Quick Start

```bash
# From repo root
source .venv/bin/activate
uv pip install -e ".[embedded_attention]"

# Test basic functionality
python -c "from projects.embedded_attention import create_conversation; print('OK')"

# Interactive validation TUI
python -m projects.embedded_attention.validate
```

## Project Structure

```
projects/embedded_attention/
├── CLAUDE.md              # Project documentation
├── __init__.py            # Re-exports from core/
│
│  # Entry-point scripts
├── evaluate.py            # LongMemEval benchmark CLI
├── validate.py            # Interactive TUI for testing/debugging
│
│  # Core implementation
├── core/
│   ├── __init__.py        # Core module exports
│   ├── chunk.py           # Chunk, ScoredChunk dataclasses
│   ├── chunking.py        # TextChunker
│   ├── chunk_store.py     # DuckDB + HNSW storage
│   ├── embedder.py        # BGE embedding wrapper
│   ├── scoring.py         # RelevanceScorer
│   ├── retriever.py       # Hybrid retrieval
│   ├── context_assembler.py # Token budget + formatting
│   ├── generator.py       # HF/API backends
│   ├── conversation.py    # Thin orchestrator
│   └── builder.py         # ConversationBuilder factory
│
├── configs/
│   └── default.yaml       # Default configuration
│
└── tests/
    ├── __init__.py
    ├── test_chunk.py      # Chunk/ScoredChunk tests
    ├── test_chunking.py   # TextChunker tests
    ├── test_scoring.py    # RelevanceScorer tests
    └── test_conversation.py # Integration tests
```

## Architecture

```
User Message → Chunk → Embed → Store → Retrieve → Score → Assemble → Generate → Store
                ↓                         ↓          ↓
           TextChunker              Retriever   RelevanceScorer
                                   (Hybrid)     (Unified)
```

## Design Principles

1. **Single Responsibility**: Each file has one clear purpose
2. **Data vs Logic Separation**: `chunk.py` is pure data; logic lives elsewhere
3. **Unified Scoring**: All relevance scoring in `scoring.py`, used by both retriever and assembler
4. **Factory Pattern**: `ConversationBuilder` handles wiring; `Conversation` stays thin

## Key Files

| File | Description |
|------|-------------|
| `core/chunk.py` | Pure data classes: Chunk, ScoredChunk |
| `core/chunking.py` | TextChunker for splitting long content |
| `core/chunk_store.py` | DuckDB + HNSW vector storage |
| `core/embedder.py` | BGE sentence embedding wrapper |
| `core/scoring.py` | Unified RelevanceScorer for retrieval + selection |
| `core/retriever.py` | Hybrid retrieval (semantic + recency + linked) |
| `core/context_assembler.py` | Token budget management + formatting |
| `core/generator.py` | HuggingFace/API generation backends |
| `core/conversation.py` | Thin orchestration layer |
| `core/builder.py` | ConversationBuilder factory |
| `evaluate.py` | LongMemEval benchmark integration |
| `validate.py` | Interactive TUI for testing/debugging |
| `configs/default.yaml` | All configuration options |

## Usage

### Basic Conversation

```python
from projects.embedded_attention import create_conversation

# Create with dummy generator (for testing)
conv = create_conversation()

# Chat - responses will use memory
response = conv.chat("My favorite color is blue.")
response = conv.chat("What's my favorite color?")
```

### With HuggingFace Model

```python
from projects.embedded_attention import create_conversation, HFGenerator

generator = HFGenerator.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda"
)

conv = create_conversation(generator=generator)
response = conv.chat("Hello!")
```

### Using the Builder

```python
from projects.embedded_attention import ConversationBuilder

conv = (
    ConversationBuilder()
    .with_generator(generator)
    .with_db_path("conversations.duckdb")
    .with_cross_conversation(True)
    .with_retrieval_config(semantic_top_k=10, min_similarity=0.6)
    .build(conversation_group_id="project-alpha")
)
```

### Cross-Conversation Retrieval

```python
from projects.embedded_attention import create_conversation

# Multiple conversations in the same group share memory
conv1 = create_conversation(
    db_path="shared.duckdb",
    cross_conversation=True,
)
conv1.chat("The API key is xyz123")

# New conversation in same group can retrieve from conv1
conv2 = create_conversation(
    db_path="shared.duckdb",
    cross_conversation=True,
)
conv2.chat("What was the API key?")
```

## Configuration

Use `ConversationConfig` for unified configuration:

```python
from projects.embedded_attention import ConversationBuilder, ConversationConfig

config = ConversationConfig(
    db_path="memory.duckdb",
    embedding_model="BAAI/bge-small-en-v1.5",
    semantic_top_k=10,
    min_similarity=0.6,
    total_budget=16384,
    recency_decay_rate=0.1,
)

conv = ConversationBuilder(config).build()
```

## Interactive Validation

```bash
# Launch interactive TUI
python -m projects.embedded_attention.validate

# With custom database
python -m projects.embedded_attention.validate --db memory.duckdb
```

TUI Modes:
- **[c] Chat**: Interactive conversation with retrieval info display
- **[r] Retrieval**: Inspect retrieval results for a query
- **[s] Store**: Browse chunks in database
- **[a] Analyze**: Detailed scoring breakdown
- **[t] Test**: Run component sanity checks
- **[o] Overfit**: Memory recall tests (verifies RAG retrieval works)
- **[q] Quit**: Exit

## Evaluation

```bash
# Run LongMemEval benchmark
python -m projects.embedded_attention.evaluate --max-examples 100

# With custom parameters
python -m projects.embedded_attention.evaluate --top-k 10 --min-sim 0.5
```

## Testing

```bash
# Run all tests
python -m pytest projects/embedded_attention/tests/ -v -o addopts=""

# Run specific test file
python -m pytest projects/embedded_attention/tests/test_conversation.py -v -o addopts=""
```

Test coverage:
- `test_chunk.py`: Chunk/ScoredChunk dataclass tests (9 tests)
- `test_chunking.py`: TextChunker splitting logic (8 tests)
- `test_scoring.py`: RelevanceScorer scoring/merging (12 tests)
- `test_conversation.py`: Integration and memory recall (12 tests)

## Dependencies

Required (install with `uv pip install -e ".[embedded_attention]"`):
- sentence-transformers>=2.2.0
- duckdb>=1.0.0
- transformers>=4.36.0
- torch>=2.0.0
- datasets>=2.14.0
