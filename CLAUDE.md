# Claude Context: ML Research Monorepo

This document provides context about this repository for AI assistants like Claude.

## Repository Purpose

This is a **machine learning research monorepo** located at `~/research/` designed for:
- Implementing and reproducing research papers
- Conducting original ML research
- Sharing common code and utilities across projects
- Managing datasets and models centrally

## Key Design Decisions

### 1. Monorepo Architecture
- **Rationale**: Reduce development friction, share code across papers/projects, single dependency environment
- **Trade-off**: Repository can grow large, but managed through aggressive .gitignore
- **Pattern**: Common in ML research (Meta FAIR, OpenAI research)

### 2. Centralized Assets with Symlinks
- **Assets location**: `~/research/assets/{datasets,models,outputs}`
- **Usage**: Papers/projects symlink to centralized assets to avoid duplication
- **Why**: Datasets and models are large; storing once saves disk space
- Example: `papers/my_paper/data -> ../../assets/datasets/imagenet`

### 3. Common Package for Shared Code
- **Location**: `common/` is a Python package
- **Structure**:
  - `common/models/` - Reusable architectures
  - `common/training/` - Training loops, optimizers
  - `common/data/` - Dataset loaders, HuggingFace utilities
  - `common/utils/` - Logging, metrics, checkpointing
  - `common/visualization/` - Plotting, analysis
- **Import pattern**: `from common.data import download_dataset`

### 4. Per-Project Docker (Not Root-Level)
- **Rationale**: Different papers may need different environments
- **Pattern**: Each paper/project can have its own `docker/` subdirectory
- **Flexibility**: Use Docker when needed, not mandatory

### 5. Using `uv` Instead of `pip`
- **Why**: `uv` is 10-100x faster than pip, written in Rust
- **Commands**: Use `uv venv` and `uv pip install` instead of standard pip
- **Location**: Virtual environment at `~/research/.venv`

## Directory Structure

```
~/research/
├── assets/              # GITIGNORED - Large files stored here
│   ├── datasets/        # Centralized datasets
│   ├── models/          # Pretrained models, checkpoints
│   └── outputs/         # Experiment outputs
│
├── common/              # Shared Python package
│   ├── __init__.py      # Makes it importable
│   ├── models/
│   ├── training/
│   ├── data/            # Includes HuggingFace utilities
│   ├── utils/
│   └── visualization/
│
├── papers/              # Paper implementations
│   └── [paper-name]/
│       ├── README.md
│       ├── train.py
│       ├── evaluate.py
│       ├── config.yaml
│       ├── docker/      # Optional
│       └── data -> ../../assets/datasets/X  # Symlink!
│
├── projects/            # Original research
│   └── [project-name]/
│       ├── README.md
│       ├── experiments/
│       ├── notebooks/
│       └── docker/      # Optional
│
├── tools/               # Standalone CLI utilities
│   ├── download_hf_dataset.py  # CLI for HF datasets
│   ├── download_hf_model.py    # CLI for HF models
│   └── test_env.py             # Environment testing
│
└── exploratory/         # One-off experiments, notebooks
    └── [date-or-topic]/
```

## Important Files

### Configuration
- **pyproject.toml**: All dependencies for entire monorepo (PyTorch, HF, etc.)
- **.gitignore**: Comprehensive - ignores assets/, checkpoints, outputs, notebooks checkpoints

### Documentation
- **README.md**: Main user documentation, setup instructions, workflows
- **HUGGINGFACE_GUIDE.md**: How to download datasets/models from HuggingFace
- **CLAUDE.md**: This file - context for AI assistants

### Scripts
- **tools/download_hf_dataset.py**: CLI for downloading HF datasets to `assets/datasets/`
- **tools/download_hf_model.py**: CLI for downloading HF models to `assets/models/`
- **tools/test_env.py**: Tests all dependencies, PyTorch, CUDA, imports

### Core Utilities
- **common/data/hf_utils.py**: Programmatic HF download functions (used by CLI scripts)
- **common/data/__init__.py**: Exports `download_dataset`, `download_model`, helper functions

## Environment Setup

### Hardware
- **GPU**: NVIDIA GeForce RTX 5060 Ti
- **VRAM**: ~15.4 GiB actual (16,582,574,080 bytes)
- **CUDA**: 12.8
- **cuDNN**: Available

### Software
- **Python**: 3.13.7
- **Package manager**: `uv` (fast alternative to pip)
- **Virtual env**: `.venv/` in repository root
- **Dependencies**: PyTorch 2.9.1+cu128, transformers, datasets, wandb, tensorboard, etc.

### Activation
```bash
cd ~/research
source .venv/bin/activate
```

## Typical Workflows

### Adding a New Paper Implementation

1. Create directory: `papers/paper_name/`
2. Add files: `README.md`, `train.py`, `evaluate.py`, `config.yaml`
3. Import from common: `from common.training import Trainer`
4. Download dataset: `python tools/download_hf_dataset.py --name squad`
5. Create symlink: `ln -s ../../assets/datasets/squad data`
6. Implement using shared utilities
7. Run experiments, document results

### Adding Reusable Code

1. Identify code that could be shared across projects
2. Move to appropriate `common/` subdirectory
3. Update `common/[submodule]/__init__.py` if needed
4. Update imports in papers/projects
5. Document in docstrings

### Managing Large Files

- **Datasets**: Download to `assets/datasets/`, symlink from papers/projects
- **Models**: Download to `assets/models/`, load in code
- **Outputs**: Save to `assets/outputs/` or paper-specific `outputs/`
- **Never commit**: All assets/ contents are gitignored

## Import Patterns

### Good: Use absolute imports from common
```python
from common.data import download_dataset, get_datasets_dir
from common.models import ResNet, Transformer
from common.training import Trainer
from common.utils import save_checkpoint
```

### Bad: Relative imports or missing common
```python
# Don't do this
from ..common.data import download_dataset
import sys; sys.path.append('../..')
```

## HuggingFace Integration

The repository has **dual-use HuggingFace utilities**:

### CLI Usage
```bash
python tools/download_hf_dataset.py --name squad
python tools/download_hf_model.py --repo-id bert-base-uncased
```

### Programmatic Usage
```python
from common.data import download_dataset, download_model

dataset = download_dataset('squad')  # Goes to assets/datasets/
model = download_model('bert-base-uncased')  # Goes to assets/models/
```

Both approaches use the same underlying functions in `common/data/hf_utils.py`.

## Git Workflow

### What's Committed
- Source code (`.py`, `.md`, `.yaml`, etc.)
- Configuration (`pyproject.toml`, `.gitignore`)
- Documentation
- Common package code

### What's Ignored (Never Committed)
- `assets/` directory (all datasets, models, outputs)
- `.venv/` virtual environment
- `*.pt`, `*.pth`, `*.ckpt` checkpoint files
- `__pycache__/`, `.ipynb_checkpoints/`
- Experiment logs (`wandb/`, `runs/`, `tensorboard_logs/`)

### Git Status
- Repository is initialized (`git init` completed)
- No remote configured yet
- Ready for initial commit

## Testing the Environment

Run comprehensive environment tests:
```bash
source .venv/bin/activate
python tools/test_env.py
```

Tests verify:
- Python version (>= 3.10)
- Core ML packages (NumPy, Pandas, SciPy, scikit-learn)
- PyTorch + CUDA availability
- HuggingFace ecosystem
- Visualization tools
- Experiment tracking
- Development tools
- Common package imports

All tests currently passing.

## Development Tools

### Code Quality
- **black**: Code formatting (configured in pyproject.toml, line-length=100)
- **ruff**: Fast linting (configured in pyproject.toml)
- **mypy**: Type checking (configured, but not strictly enforced)
- **pytest**: Testing framework (when tests are added)
- **pre-commit**: Optional git hooks

### Usage
```bash
# Format code
black .

# Lint
ruff check .

# Type check
mypy common/

# Run tests (when available)
pytest
```

## Related Setup

The user previously had an ML project at `~/Development/Setup/ml-project` which served as initial exploration. This new `~/research/` structure is a fresh start with a different, more research-focused organization.

## Helper Commands for Claude

When working in this repository, remember:

```bash
# Always activate venv first
source ~/research/.venv/bin/activate

# Install new dependencies
uv pip install package-name

# Test environment
python ~/research/tools/test_env.py

# Download HF dataset
python ~/research/tools/download_hf_dataset.py --name [dataset]

# Download HF model
python ~/research/tools/download_hf_model.py --repo-id [model]
```

## User Preferences

Based on interactions:
- Prefers monorepo for reduced friction
- Likes centralized assets with symlinks
- Values comprehensive documentation
- Uses `uv` for faster package management
- Wants both CLI and programmatic interfaces for tools
- Prefers per-project Docker over root-level

## Next Steps (Potential)

The repository is now set up and ready for:
1. Adding first paper implementation
2. Building out `common/` utilities as needed
3. Creating reusable model architectures
4. Setting up experiment tracking (wandb/mlflow)
5. Adding benchmarking utilities
6. Creating result visualization tools

## Notes

- This is a new machine/setup, migrated from `~/Development/Setup/ml-research`
- Environment is fully configured and tested
- All dependencies installed (196 packages)
- GPU support verified and working
- Ready for research work
