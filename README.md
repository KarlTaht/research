# ML Research Monorepo

A monorepo for machine learning research, paper implementations, and original research projects.

## Directory Structure

```
~/research/
├── pyproject.toml              # Dependency management for entire monorepo
├── .gitignore                  # Comprehensive ignore rules
│
├── assets/                     # Centralized storage (gitignored)
│   ├── datasets/               # Raw & processed datasets
│   ├── models/                 # Pretrained models & checkpoints
│   └── outputs/                # Experiment outputs, plots, results
│
├── common/                     # Shared codebase (Python package)
│   ├── models/                 # Reusable model architectures
│   ├── training/               # Training loops, optimizers, schedulers
│   ├── data/                   # Dataset loaders, transforms, augmentation
│   ├── utils/                  # Logging, metrics, checkpointing, tracking
│   └── visualization/          # Plotting, analysis, figure generation
│
├── papers/                     # Paper implementations
│   └── [paper-name]/           # One directory per paper
│       ├── README.md
│       ├── train.py
│       ├── evaluate.py
│       ├── config.yaml
│       ├── docker/             # Optional Docker setup
│       └── data -> ../../assets/datasets/[name]  # Symlink
│
├── projects/                   # Original research projects
│   └── [project-name]/
│       ├── README.md
│       ├── experiments/
│       ├── notebooks/
│       └── docker/
│
├── tools/                      # Standalone utilities
│   ├── download_hf_dataset.py  # CLI for downloading HF datasets
│   ├── download_hf_model.py    # CLI for downloading HF models
│   ├── benchmark.py            # (to be added)
│   └── visualize_results.py    # (to be added)
│
└── exploratory/                # Exploratory analysis & one-off experiments
    └── [date-or-topic]/
        └── *.ipynb
```

## Setup

### 1. Install uv (if not already installed)

`uv` is a fast Python package installer and resolver, written in Rust. It's much faster than pip.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### 2. Install Dependencies

```bash
cd ~/research

# Create virtual environment with uv
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core dependencies (much faster than pip!)
uv pip install -e .

# Install with dev tools
uv pip install -e ".[dev]"

# Install with all extras (CV, NLP, dev tools)
uv pip install -e ".[all]"
```

### 3. Test Your Environment

Run the environment test script to verify all dependencies are properly installed:

```bash
# Make sure venv is activated
source .venv/bin/activate

# Run tests
python tools/test_env.py
```

This will test:
- Python version
- Core ML packages (NumPy, Pandas, SciPy, scikit-learn)
- PyTorch & CUDA availability
- HuggingFace ecosystem
- Visualization tools
- Experiment tracking
- Development tools
- Your `common` package imports

### 4. Install Pre-commit Hooks (Optional)

```bash
source .venv/bin/activate
pre-commit install
```

## Usage

### Importing from Common Package

The `common/` directory is a Python package that can be imported from anywhere in the monorepo:

```python
# Import models
from common.models import ResNet, Transformer, AttentionLayer

# Import training utilities
from common.training import Trainer, get_optimizer, get_scheduler

# Import data utilities
from common.data import ImageDataset, TextDataset, create_dataloader

# Import utilities
from common.utils import save_checkpoint, load_checkpoint, log_metrics
from common.utils import setup_logger, get_device

# Import visualization
from common.visualization import plot_training_curves, plot_confusion_matrix
```

### Adding a New Paper Implementation

1. Create a directory under `papers/`:
   ```bash
   mkdir -p papers/attention_is_all_you_need
   cd papers/attention_is_all_you_need
   ```

2. Create necessary files:
   ```bash
   touch README.md train.py evaluate.py config.yaml
   ```

3. Symlink to dataset (if needed):
   ```bash
   ln -s ../../assets/datasets/wmt14 data
   ```

4. Implement using common utilities:
   ```python
   # train.py
   from common.models import Transformer
   from common.training import Trainer
   from common.data import create_dataloader
   from common.utils import save_checkpoint

   # Your implementation here
   ```

### Adding a New Research Project

1. Create project directory:
   ```bash
   mkdir -p projects/my_research_project
   cd projects/my_research_project
   ```

2. Structure:
   ```
   my_research_project/
   ├── README.md              # Project overview & results
   ├── experiments/
   │   ├── baseline.yaml
   │   ├── ablation_1.yaml
   │   └── final_model.yaml
   ├── notebooks/
   │   └── analysis.ipynb
   ├── train.py
   ├── evaluate.py
   └── docker/                # If needed
       ├── Dockerfile
       └── docker-compose.yml
   ```

### Managing Assets

Assets (datasets, models, outputs) are centralized and gitignored.

#### Downloading from HuggingFace

Use the built-in HuggingFace utilities (see [HUGGINGFACE_GUIDE.md](HUGGINGFACE_GUIDE.md) for full details):

```bash
# Download dataset via CLI
python tools/download_hf_dataset.py --name squad
python tools/download_hf_dataset.py --name wmt14 --split train

# Download model via CLI
python tools/download_hf_model.py --repo-id bert-base-uncased
python tools/download_hf_model.py --repo-id openai/clip-vit-base-patch32

# Or use programmatically in your code
```

```python
from common.data import download_dataset, download_model

# Download in training scripts
dataset = download_dataset('squad')
model_path = download_model('bert-base-uncased')
```

#### Managing Downloaded Assets

```bash
# Symlink from a paper/project
cd papers/attention_is_all_you_need
ln -s ../../assets/datasets/wmt14 data

# Download from other sources
cd ~/research/assets/models
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
```

### Running Experiments

```bash
# Activate environment
source .venv/bin/activate

# Run from paper directory
cd papers/attention_is_all_you_need
python train.py --config config.yaml

# Or run from root with module syntax
cd ~/research
python -m papers.attention_is_all_you_need.train --config papers/attention_is_all_you_need/config.yaml
```

### Using Docker (Per-Project)

Each paper/project can have its own Docker setup:

```dockerfile
# papers/my_paper/docker/Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /research
COPY . .

RUN pip install -e .
CMD ["python", "papers/my_paper/train.py"]
```

```yaml
# papers/my_paper/docker/docker-compose.yml
version: '3.8'
services:
  training:
    build:
      context: ../..
      dockerfile: papers/my_paper/docker/Dockerfile
    volumes:
      - ../../assets:/research/assets
      - ../../common:/research/common
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

## Workflow Examples

### Typical Research Workflow

1. **Exploratory phase**: Create notebook in `exploratory/2025-01-15-idea/`
2. **Extract reusable code**: Move common functionality to `common/`
3. **Create project**: Set up in `projects/my_idea/`
4. **Run experiments**: Use `common/` utilities for training/eval
5. **Analyze results**: Use `common/visualization` for plots

### Reproducing a Paper

1. **Create paper directory**: `papers/paper_name/`
2. **Implement using common**: Reuse models/training from `common/`
3. **Download assets**: Store in `assets/`, symlink to paper
4. **Run experiments**: Train and evaluate
5. **Document results**: Update paper's README with findings

## Best Practices

1. **Never commit large files**: Use .gitignore, store in `assets/`
2. **Share code via common/**: Don't duplicate utility functions
3. **Use config files**: YAML configs for reproducible experiments
4. **Document everything**: READMEs for each paper/project
5. **Symlink datasets**: Avoid duplicating large files across projects
6. **Version control code only**: Models, datasets, outputs are gitignored

## Development

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy common/

# Run tests (when you add them)
pytest
```

### Adding to Common Package

When you find reusable code in a paper/project:

1. Move it to appropriate `common/` subdirectory
2. Add to `common/[submodule]/__init__.py` if needed
3. Update imports in your paper/project
4. Document in docstrings

## Tips

- **Import path**: Always import from root (e.g., `from common.models import ...`)
- **Relative imports**: Avoid them; use absolute imports from `common/`
- **Config management**: Use `omegaconf` or `yaml` for experiment configs
- **Experiment tracking**: Use wandb or tensorboard (already in dependencies)
- **GPU management**: Use `common/utils` for device handling

## Next Steps

1. Add your first paper implementation or project
2. Build out `common/` with reusable utilities as you go
3. Create `tools/` scripts for common tasks (dataset downloads, benchmarking)
4. Set up experiment tracking (wandb, mlflow, or tensorboard)
