# assets/ - Centralized Asset Storage

This directory stores large files (datasets, models, outputs) that are **not committed to git**. Assets are downloaded on-demand and shared across all papers/projects via symlinks.

## Directory Structure

```
assets/
├── datasets/           # HuggingFace datasets (downloaded via tools/)
│   ├── roneneldan/
│   │   └── TinyStories/
│   ├── nampdn-ai/
│   │   ├── tiny-textbooks/
│   │   ├── tiny-strange-textbooks/
│   │   └── tiny-codes/
│   └── ...
│
├── models/             # Pretrained models and custom tokenizers
│   ├── tokenizers/     # Custom trained tokenizers
│   │   └── tinystories_bpe_4096/
│   └── *.pt            # Model checkpoints
│
└── outputs/            # Experiment results
    └── experiments/    # Parquet files for experiment tracking
        └── *.parquet
```

## Downloading Assets

### Datasets
```bash
# Download a HuggingFace dataset
python tools/download_hf_dataset.py --name roneneldan/TinyStories
python tools/download_hf_dataset.py --name nampdn-ai/tiny-textbooks

# Or programmatically
from common.data import download_dataset
download_dataset('roneneldan/TinyStories')
```

### Models
```bash
# Download a HuggingFace model
python tools/download_hf_model.py --repo-id gpt2

# Or programmatically
from common.data import download_model
download_model('gpt2')
```

### Custom Tokenizers
```bash
# Train a dataset-specific tokenizer
python tools/analyze_tokens.py --dataset tinystories --train-tokenizer 4096
```

Tokenizers are saved to `assets/models/tokenizers/<dataset>_bpe_<vocab_size>/`

## Using Assets in Projects

### Via Symlinks (Recommended)
```bash
cd papers/my_paper
ln -s ../../assets/datasets/roneneldan/TinyStories data
```

### Via Registry
```python
from common.data import load_training_data
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

train_loader, val_loader = load_training_data(
    'tinystories',  # Uses registry to find path
    tokenizer,
    max_length=256,
    batch_size=16,
)
```

### Custom Tokenizers

Custom tokenizers are stored in `models/tokenizers/` with naming convention `<dataset>_bpe_<vocab_size>`.

Each tokenizer directory contains:
- `tokenizer.json` - Vocabulary and merge rules
- `tokenizer_config.json` - Configuration (model type, special tokens)
- `special_tokens_map.json` - Special token definitions

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'assets/models/tokenizers/tinystories_bpe_4096'
)
```

Train a new tokenizer:
```bash
python tools/analyze_tokens.py --dataset <name> --train-tokenizer <vocab_size>
```

## Git Configuration

The `assets/` directory is configured in `.gitignore` to prevent committing large files. Only `.md` files (this CLAUDE.md and README.md stubs) are tracked to preserve directory structure visibility.

## Experiment Storage

Experiment results are stored as Parquet files in `outputs/experiments/`:

```python
from common.utils import save_experiment, query_experiments

# Save results
save_experiment('exp_001', results_df, metadata={'model': 'custom-transformer'})

# Query with SQL
best = query_experiments("""
    SELECT experiment_name, MIN(perplexity) as best_ppl
    FROM experiments
    GROUP BY experiment_name
    ORDER BY best_ppl
    LIMIT 5
""")
```

Query via CLI:
```bash
python tools/query_experiments.py --list
python tools/query_experiments.py --best perplexity --top 5
```
