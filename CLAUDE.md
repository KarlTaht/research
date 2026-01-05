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
research-infra env
# or: python -m common.cli.infra env
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

### CLI Commands

The research CLI provides unified access to all tools. Launch the TUI or use commands directly:

```bash
# Launch interactive TUI (Ctrl+P for command palette)
research

# Data operations
research-data download dataset --name squad
research-data download model --repo-id gpt2
research-data analyze --dataset tinystories
research-data pretokenize --dataset tinystories --tokenizer gpt2
research-data fineweb sample --tokens 10000000
research-data fineweb index --status
research-data fineweb query --top-domains 50
research-data fineweb extract --corpus automotive

# Experiment tracking
research-exp list
research-exp summary
research-exp best perplexity --minimize --top 5
research-exp query "SELECT * FROM experiments WHERE perplexity < 20"
research-exp compare exp_001 exp_002

# Infrastructure
research-infra env
research-infra lambda
```

Alternative: Use `python -m common.cli.<module>` directly:
```bash
python -m common.cli.data download dataset --name squad
python -m common.cli.experiments list
python -m common.cli.infra env
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

- **`common/cli/`**: Unified CLI with Textual TUI
  - `data.py`: Data operations (download, analyze, pretokenize, fineweb)
  - `experiments.py`: Experiment tracking (list, query, compare)
  - `infra.py`: Infrastructure (env check, cloud availability)
  - `tui/`: Textual-based interactive TUI
  - `_internal/`: Individual command implementations

- **`tools/`**: Thin wrapper scripts (for backwards compatibility)
  - `test_env.py`: Environment verification
  - `lambda_availability.py`: Lambda Labs GPU availability

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

1. **CLI commands** (`research-data download`): For manual downloads
2. **Programmatic API** (`common/data/hf_utils.py`): For use in training scripts

Both use the same underlying functions from `common.data`:

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
research-data download dataset --name wmt14

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
6. Download dataset: `research-data download dataset --name dataset_name`
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

## Plan Execution Workflow

When executing a plan (after exiting plan mode), ALWAYS complete these steps before considering the task done:

### 1. Run Tests
After implementing changes, run the relevant test suite:
```bash
# For the main common package
pytest common/

# For specific projects
pytest projects/<project_name>/

# Or run all tests
pytest
```

### 2. Address Test Failures
If tests fail:
- Fix the failing tests or the code causing failures
- Re-run tests until they pass
- Do not skip this step - all tests must pass before committing

### 3. Run Linting/Formatting
Ensure code quality:
```bash
black .
ruff check . --fix
mypy common/  # if type hints were modified
```

### 4. Prompt to Commit and Push
Once tests pass and code is clean, **always ask the user** if they'd like to commit the changes:
```
"Tests pass and linting is clean. Would you like me to commit and push these changes?"
```

If the user agrees:
```bash
git add <relevant files>
git commit -m "<descriptive message>"
git push
```

**Important**: Do not ask for permission to run tests or linting - these are expected steps. However, always prompt before committing. Only skip asking about test failures or merge conflicts if they require user input to resolve.
