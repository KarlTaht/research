# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ML research monorepo at `~/research/` for conducting original ML research. Uses centralized assets with symlinks, shared `common/` package for reusable code, and `uv` for fast package management.

## Essential Commands

### Environment Setup
```bash
# Activate virtual environment (ALWAYS do this first)
source .venv/bin/activate

# Install dependencies (use uv, not pip)
uv pip install -e ".[all]"

# Test environment
python tools/test_env.py
```

### Development Commands
```bash
# Format code (line-length=100)
black .

# Lint
ruff check .

# Type check
mypy common/

# Run tests (when added)
pytest
```

### HuggingFace Asset Management
```bash
# Download dataset to assets/datasets/
python tools/download_hf_dataset.py --name squad
python tools/download_hf_dataset.py --name wmt14 --split train --config de-en

# Download model to assets/models/
python tools/download_hf_model.py --repo-id bert-base-uncased
python tools/download_hf_model.py --repo-id gpt2 --filename pytorch_model.bin
```

### Experiment Results Storage
```bash
# Query experiment results with SQL
python tools/query_experiments.py --sql "SELECT * FROM experiments WHERE perplexity < 20"

# List all experiments
python tools/query_experiments.py --list

# Get summary of all experiments
python tools/query_experiments.py --summary

# Get top 5 best models by perplexity
python tools/query_experiments.py --best perplexity --top 5

# Compare specific experiments
python tools/query_experiments.py --compare exp_001 exp_002
```

### Running Experiments
```bash
# From project directory
cd projects/custom_transformer
python train.py --config configs/tinystories.yaml

# From repo root
python -m projects.custom_transformer.train
```

## Architecture

### Monorepo Structure
- **`common/`**: Shared Python package for reusable code across all projects
  - `models/`: Model architectures
    - `base.py`: `BaseLanguageModel` - abstract base class for all language models
  - `training/`: Training loops, optimizers, schedulers
    - `evaluator.py`: `Evaluator` - standard evaluation framework with metrics
  - `data/`: Dataset loaders, HuggingFace utilities (`hf_utils.py`)
  - `utils/`: Logging, checkpointing, metrics, experiment storage
    - `experiment_storage.py`: Parquet + DuckDB experiment tracking
  - `visualization/`: Plotting and analysis

- **`assets/`**: Centralized storage (GITIGNORED)
  - `datasets/`: Raw and processed datasets
  - `models/`: Pretrained models and checkpoints
  - `outputs/experiments/`: Experiment results (Parquet files)

- **`projects/[project-name]/`**: Research projects
  - Symlinks to `assets/datasets/` for data (e.g., `data -> ../../assets/datasets/squad`)
  - Contains `train.py`, `evaluate.py`, `config.yaml`

- **`tools/`**: Standalone CLI utilities
  - `download_hf_dataset.py`: CLI for downloading HF datasets
  - `download_hf_model.py`: CLI for downloading HF models
  - `query_experiments.py`: CLI for querying experiment results
  - `test_env.py`: Environment verification

- **`exploratory/`**: Jupyter notebooks and one-off experiments

### Import Patterns

Always use absolute imports from `common/`:

```python
# CORRECT
from common.data import download_dataset, get_datasets_dir, download_model
from common.models import BaseLanguageModel
from common.training import Evaluator, compute_perplexity
from common.utils import save_experiment, query_experiments
from common.visualization import plot_training_curves

# WRONG - avoid relative imports
from ..common.data import download_dataset
```

### HuggingFace Integration Architecture

Dual-use system for downloading datasets/models:

1. **CLI scripts** (`tools/download_hf_*.py`): For manual downloads
2. **Programmatic API** (`common/data/hf_utils.py`): For use in training scripts

Both use the same underlying functions:
- CLI scripts call functions from `common/data/hf_utils.py`
- Training scripts can import directly from `common.data`

```python
# In training scripts
from common.data import download_dataset, download_model, get_datasets_dir

# Download on demand
dataset = download_dataset('squad')  # Goes to assets/datasets/ by default
model_path = download_model('bert-base-uncased')

# Or load pre-downloaded data
from datasets import load_from_disk
dataset = load_from_disk(get_datasets_dir() / 'squad')
```

### Asset Management Pattern

Large files (datasets, models, outputs) are:
1. Stored centrally in `assets/` directory
2. GITIGNORED (never committed)
3. Symlinked from projects to avoid duplication

```bash
# Download once
python tools/download_hf_dataset.py --name wmt14

# Use in multiple projects via symlinks
cd projects/custom_transformer
ln -s ../../assets/datasets/wmt14 data

cd ../embedded_attention
ln -s ../../assets/datasets/wmt14 data
```

### Experiment Storage Architecture

**System**: Parquet (storage) + DuckDB (querying) hybrid

Experiment results are stored as Parquet files for efficient columnar storage, then queried with DuckDB for fast analytics. This provides:
- Excellent compression (minimal disk space)
- Fast SQL queries across all experiments
- Zero data duplication (DuckDB queries Parquet directly)
- 100% local, works offline

**Programmatic API** (`common/utils/experiment_storage.py`):
```python
from common.utils import save_experiment, query_experiments

# Save results
import pandas as pd
results = pd.DataFrame({'epoch': [1,2,3], 'perplexity': [25, 18, 15]})
save_experiment('exp_001', results, metadata={'model': 'llm-tiny'})

# Query with SQL
best = query_experiments("""
    SELECT experiment_name, MIN(perplexity) as best_perplexity
    FROM experiments
    WHERE epoch >= 5
    GROUP BY experiment_name
    ORDER BY best_perplexity LIMIT 5
""")
```

**Storage Location**: `assets/outputs/experiments/*.parquet`

### Model and Training Infrastructure

**Base Model Class** (`common/models/base.py`):

All language models should extend `BaseLanguageModel` for consistency:

```python
from common.models import BaseLanguageModel
import torch.nn as nn

class MyModel(BaseLanguageModel):
    def __init__(self, vocab_size, **kwargs):
        super().__init__(vocab_size, **kwargs)
        # Your layers here

    def forward(self, input_ids, labels=None):
        # Your implementation
        # Must return dict with 'logits' and optionally 'loss'
        return {'logits': logits, 'loss': loss}

# Inherited methods: generate(), save_checkpoint(), load_checkpoint()
```

**Evaluation Framework** (`common/training/evaluator.py`):

Standard evaluator for computing metrics:

```python
from common.training import Evaluator

evaluator = Evaluator(model, device='cuda')

# Evaluate on validation set
metrics = evaluator.evaluate(val_dataloader)
# Returns: {'loss': 1.23, 'perplexity': 3.45, 'num_tokens': 10000}

# Generate sample outputs
samples = evaluator.generate_samples(
    prompts=['Once upon a time'],
    tokenizer=tokenizer,
    max_length=50
)

# Convert to DataFrame for experiment storage
results_df = evaluator.create_metrics_dataframe(metrics, epoch=5, split='val')
```

**Example Project**: `projects/custom_transformer/`

A complete working example demonstrating:
- Custom transformer with manual backpropagation
- Training with evaluation and checkpointing
- TinyStories dataset with GPT-2 or custom tokenization

See `projects/custom_transformer/CLAUDE.md` for full documentation.

## Key Workflows

### Adding a New Project
1. `mkdir -p projects/project_name`
2. Create model extending `BaseLanguageModel`:
   ```python
   from common.models import BaseLanguageModel
   class MyModel(BaseLanguageModel):
       def forward(self, input_ids, labels=None):
           # Implementation
           return {'logits': logits, 'loss': loss}
   ```
3. Create `train.py` using `Evaluator` and `save_experiment()`
4. Create `evaluate.py` for testing
5. Create `config.yaml` for hyperparameters
6. Download dataset: `python tools/download_hf_dataset.py --name dataset_name`
7. Run training and track experiments

See `projects/custom_transformer/` for a complete working example.

### Adding Reusable Code to Common
1. Move code to appropriate `common/` subdirectory
2. Update `common/[submodule]/__init__.py` to export it
3. Update imports in projects
4. Document with docstrings

### Environment Details
- **Python**: 3.13.7 (requires >= 3.10)
- **GPU**: NVIDIA RTX 5060 Ti, ~15.4 GiB VRAM
- **CUDA**: 12.8
- **PyTorch**: 2.9.1+cu128
- **Package Manager**: `uv` (10-100x faster than pip)

## Important Notes

- **Never commit** `assets/`, checkpoints (`.pt`, `.pth`, `.ckpt`), or experiment logs
- **Always activate** `.venv` before running any Python commands
- **Use `uv`** instead of `pip` for installing packages
- **Use symlinks** to share datasets across projects
- **Per-project Docker**: Each project can have its own `docker/` subdirectory (not root-level)
- **Line length**: 100 characters (black/ruff configured)
