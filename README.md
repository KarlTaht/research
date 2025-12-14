# ML Research Monorepo

A monorepo for machine learning research and original research projects.

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
├── projects/                   # Research projects
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

## Current Projects

| Project | Description |
|---------|-------------|
| **custom_transformer** | Decoder-only transformer with manual backpropagation (no autograd) for educational purposes |
| **embedded_attention** | Chunk-based conversational memory system using RAG with DuckDB vector storage |
| **continual_learning** | Experiments to build intuition for catastrophic forgetting in transformers |
| **research_agent** | Research assistant that reads papers, maintains hypotheses, and synthesizes findings |
| **research_manager_agent** | Agent for navigating and managing this ML research monorepo |

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
# Symlink from a project
cd projects/my_project
ln -s ../../assets/datasets/wmt14 data

# Download from other sources
cd ~/research/assets/models
wget https://download.pytorch.org/models/resnet50-0676ba61.pth
```

## Dataset Preparation

This section describes the full workflow for preparing domain-specific datasets from large corpora like FineWeb. The example below creates automotive and food corpora for continual learning experiments.

### 1. Build a Domain Index

For large datasets, first create an index to enable fast domain filtering:

```bash
cd projects/continual_learning

# Build index from FineWeb (extracts domain, token counts, file locations)
python build_domain_index.py
```

This creates a Parquet index at `assets/datasets/.../domain_index.parquet` with columns:
- `etld_plus_one`: Domain (e.g., "bmw.com", "allrecipes.com")
- `token_count`: Number of tokens in document
- `shard_file`: Source Arrow file
- `row_idx`: Row index within shard

### 2. Query the Index to Find Categories

Use SQL queries to explore available domains and estimate corpus sizes:

```bash
# Interactive exploration
python query_domain_index.py

# Example queries:
# - Find automotive domains: WHERE etld_plus_one LIKE '%auto%'
# - Estimate size: SELECT SUM(token_count) FROM idx WHERE ...
# - List top domains: SELECT etld_plus_one, COUNT(*) GROUP BY 1 ORDER BY 2 DESC
```

### 3. Extract Domain-Specific Corpora

Once you've identified target domains, extract them to JSONL files:

```bash
# Preview what will be extracted
python extract_corpus.py --preview

# Extract both corpora (default ~125M tokens each, ~500MB)
python extract_corpus.py

# Extract specific corpus with custom size
python extract_corpus.py --corpus automotive --target-tokens 50000000
```

Output structure:
```
data/
├── corpus_automotive/
│   ├── train.jsonl      # {"text": "..."}
│   ├── val.jsonl
│   └── metadata.json
└── corpus_food/
    ├── train.jsonl
    ├── val.jsonl
    └── metadata.json
```

### 4. Analyze Token Distributions

Understand vocabulary usage to choose optimal tokenizer size:

```bash
# Analyze a single corpus
python tools/analyze_tokens.py --jsonl data/corpus_automotive/train.jsonl

# Compare multiple corpora
python tools/analyze_tokens.py \
  --jsonl data/corpus_automotive/train.jsonl data/corpus_food/train.jsonl \
  --compare

# Get tokenizer size recommendations
python tools/analyze_tokens.py --jsonl data/corpus_automotive/train.jsonl --recommendations
```

Key metrics to look for:
- **Coverage milestones**: Tokens needed for 90%/95%/99% coverage
- **Vocabulary utilization**: What fraction of GPT-2's 50k tokens are actually used
- **Edge cases**: Single chars, punctuation, numbers, rare tokens

### 5. Train a Custom Tokenizer

Train a BPE tokenizer optimized for your corpus:

```bash
# Train on single corpus
python tools/analyze_tokens.py \
  --jsonl data/corpus_automotive/train.jsonl \
  --train-tokenizer 16384

# Train on combined corpora (recommended for multi-domain experiments)
python tools/analyze_tokens.py \
  --jsonl data/corpus_automotive/train.jsonl data/corpus_food/train.jsonl \
  --train-tokenizer 32768
```

Tokenizer saved to: `assets/models/tokenizers/combined_bpe_32768/`

**Choosing vocab size:**
- Use coverage analysis to find where diminishing returns start (~99% coverage)
- Powers of 2 are optimal for GPU efficiency (4096, 8192, 16384, 32768)
- Smaller vocab = faster training, larger embeddings proportion

### 6. Pre-tokenize for Fast Training

Cache tokenized data to eliminate startup overhead:

```bash
# Pre-tokenize with your custom tokenizer
python tools/pretokenize_dataset.py \
  --jsonl data/corpus_automotive/train.jsonl \
  --val-jsonl data/corpus_automotive/val.jsonl \
  --tokenizer combined_bpe_32768 \
  --max-length 1024

# Repeat for other corpora
python tools/pretokenize_dataset.py \
  --jsonl data/corpus_food/train.jsonl \
  --val-jsonl data/corpus_food/val.jsonl \
  --tokenizer combined_bpe_32768 \
  --max-length 1024
```

Cache location: `data/corpus_automotive/train_tokenized/combined_bpe_32768_v32768_len1024/`

**Benefits:**
- First run: ~70s to tokenize 180k documents
- Subsequent runs: ~1s to load from cache
- Cache is tokenizer-specific (different tokenizer = different cache)

### Complete Example: FineWeb → Training-Ready Data

```bash
cd projects/continual_learning

# 1. Build index (one-time, ~30 min for 100B token sample)
python build_domain_index.py

# 2. Explore domains
python query_domain_index.py
# → Identified: automotive (~800M tokens), food (~350M tokens)

# 3. Extract corpora (~500MB each)
python extract_corpus.py

# 4. Analyze and choose tokenizer size
python tools/analyze_tokens.py \
  --jsonl data/corpus_automotive/train.jsonl data/corpus_food/train.jsonl \
  --compare --recommendations
# → 99% coverage needs ~34k tokens, chose 32768

# 5. Train combined tokenizer
python tools/analyze_tokens.py \
  --jsonl data/corpus_automotive/train.jsonl data/corpus_food/train.jsonl \
  --train-tokenizer 32768

# 6. Pre-tokenize both corpora
for corpus in automotive food; do
  python tools/pretokenize_dataset.py \
    --jsonl data/corpus_${corpus}/train.jsonl \
    --val-jsonl data/corpus_${corpus}/val.jsonl \
    --tokenizer combined_bpe_32768 \
    --max-length 1024
done

# Ready for training!
```

### Running Experiments

```bash
# Activate environment
source .venv/bin/activate

# Run from project directory
cd projects/custom_transformer
python train.py --config configs/tinystories.yaml

# Or run from root with module syntax
cd ~/research
python -m projects.custom_transformer.train --config projects/custom_transformer/configs/tinystories.yaml
```

### Using Docker (Per-Project)

Each project can have its own Docker setup:

```dockerfile
# projects/my_project/docker/Dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /research
COPY . .

RUN pip install -e .
CMD ["python", "projects/my_project/train.py"]
```

```yaml
# projects/my_project/docker/docker-compose.yml
version: '3.8'
services:
  training:
    build:
      context: ../..
      dockerfile: projects/my_project/docker/Dockerfile
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
