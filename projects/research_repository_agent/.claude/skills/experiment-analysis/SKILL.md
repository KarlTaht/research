---
name: experiment-analysis
description: Query and analyze ML experiment results. Use when user asks about experiments, metrics, training runs, or wants to compare results.
---

# Experiment Analysis

Help users query, analyze, and compare ML experiment results.

## When to Use

- "What experiments have I run?"
- "Best perplexity results"
- "Compare exp_001 and exp_002"
- "Show experiments from last week"
- "Which config got the best results?"

## Storage System

Experiments are stored in Parquet files and queried with DuckDB:
- Location: `assets/outputs/experiments/*.parquet`
- Schema: experiment_name, metrics, config, timestamp, etc.

## Available Tools

| Tool | Purpose |
|------|---------|
| `query_experiments` | Natural language or SQL queries |
| `analyze_logs` | Parse training logs for insights |
| `find_checkpoint` | Locate model checkpoints |
| `compare_runs` | Side-by-side comparison |

## Query Patterns

See [SQL_PATTERNS.md](SQL_PATTERNS.md) for common query patterns.

## Common Workflows

### Finding Best Results

```
User: "What's my best perplexity?"

1. Use query_experiments(query="best perplexity")
2. Show top results with experiment names
3. Offer to show config or compare with others
```

### Comparing Experiments

```
User: "Compare exp_001 and exp_002"

1. Use compare_runs(experiments=['exp_001', 'exp_002'])
2. Show config differences
3. Show metric differences
4. Highlight what likely caused performance difference
```

### Finding Checkpoints

```
User: "Where's the checkpoint for my best model?"

1. Query for best experiment
2. Use find_checkpoint(experiment='...')
3. Show path and associated config
```

## Output Formatting

Use `scripts/format_results.py` for consistent table output when showing multiple experiments.
