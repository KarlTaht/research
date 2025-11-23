# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

ML research monorepo at `~/research/` for implementing papers and conducting original research. Uses centralized assets with symlinks, shared `common/` package for reusable code, and `uv` for fast package management.

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

### Running Experiments
```bash
# From paper directory
cd papers/attention_is_all_you_need
python train.py --config config.yaml

# From repo root
python -m papers.attention_is_all_you_need.train
```

## Architecture

### Monorepo Structure
- **`common/`**: Shared Python package for reusable code across all papers/projects
  - `models/`: Model architectures
  - `training/`: Training loops, optimizers, schedulers
  - `data/`: Dataset loaders, HuggingFace utilities (`hf_utils.py`)
  - `utils/`: Logging, checkpointing, metrics
  - `visualization/`: Plotting and analysis

- **`assets/`**: Centralized storage (GITIGNORED)
  - `datasets/`: Raw and processed datasets
  - `models/`: Pretrained models and checkpoints
  - `outputs/`: Experiment outputs

- **`papers/[paper-name]/`**: One directory per paper implementation
  - Symlinks to `assets/datasets/` for data (e.g., `data -> ../../assets/datasets/squad`)
  - Contains `train.py`, `evaluate.py`, `config.yaml`

- **`projects/[project-name]/`**: Original research projects
  - Similar structure to papers but with more flexibility

- **`tools/`**: Standalone CLI utilities
  - `download_hf_dataset.py`: CLI for downloading HF datasets
  - `download_hf_model.py`: CLI for downloading HF models
  - `test_env.py`: Environment verification

- **`exploratory/`**: Jupyter notebooks and one-off experiments

### Import Patterns

Always use absolute imports from `common/`:

```python
# CORRECT
from common.data import download_dataset, get_datasets_dir, download_model
from common.models import ResNet, Transformer
from common.training import Trainer
from common.utils import save_checkpoint, setup_logger
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
3. Symlinked from papers/projects to avoid duplication

```bash
# Download once
python tools/download_hf_dataset.py --name wmt14

# Use in multiple papers via symlinks
cd papers/transformer_paper
ln -s ../../assets/datasets/wmt14 data

cd ../another_paper
ln -s ../../assets/datasets/wmt14 data
```

## Key Workflows

### Adding a New Paper Implementation
1. `mkdir -p papers/paper_name`
2. Create `train.py`, `evaluate.py`, `config.yaml`, `README.md`
3. Download dataset: `python tools/download_hf_dataset.py --name dataset_name`
4. Create symlink: `ln -s ../../assets/datasets/dataset_name papers/paper_name/data`
5. Import from `common/` and implement

### Adding Reusable Code to Common
1. Move code to appropriate `common/` subdirectory
2. Update `common/[submodule]/__init__.py` to export it
3. Update imports in papers/projects
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
- **Use symlinks** to share datasets across papers/projects
- **Per-project Docker**: Each paper/project can have its own `docker/` subdirectory (not root-level)
- **Line length**: 100 characters (black/ruff configured)
