# common/data - Data Utilities

This module provides data loading, dataset management, and tokenization utilities.

## Files

### hf_utils.py
HuggingFace integration utilities for downloading and managing assets.

```python
from common.data import download_dataset, download_model, get_datasets_dir

# Download a dataset to assets/datasets/
dataset = download_dataset('roneneldan/TinyStories')

# Download a model to assets/models/
model_path = download_model('gpt2')

# Get paths
datasets_dir = get_datasets_dir()  # assets/datasets/
models_dir = get_models_dir()      # assets/models/
```

### dataset_registry.py
Centralized registry of supported datasets with unified loading interface.

**Registered Datasets:**
| Name | HF Path | Text Column | Description |
|------|---------|-------------|-------------|
| `tinystories` | roneneldan/TinyStories | text | Simple children's stories |
| `tiny-textbooks` | nampdn-ai/tiny-textbooks | textbook | Educational textbooks |
| `tiny-strange-textbooks` | nampdn-ai/tiny-strange-textbooks | textbook | Diverse textbooks |
| `tiny-codes` | nampdn-ai/tiny-codes | code | Code snippets |

```python
from common.data import load_training_data, list_datasets
from transformers import GPT2TokenizerFast

# List available datasets
print(list_datasets())

# Load train/val dataloaders
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

train_loader, val_loader = load_training_data(
    'tinystories',
    tokenizer,
    max_length=256,
    batch_size=16,
    subset_size=10000,  # Optional: limit training examples
)
```

### token_analyzer.py
Analyzes token distributions to optimize tokenizer vocabulary sizes for small-scale experiments.

**Key Features:**
- **CDF Analysis**: Computes coverage curve (N tokens → X% of dataset)
- **Edge Case Detection**: Identifies single chars, punctuation, numbers, hyphens, rare tokens
- **Recommendations**: Suggests vocab sizes for 90%, 95%, 99%, 99.9% coverage
- **Custom Tokenizer Training**: Trains dataset-specific BPE tokenizers

```python
from common.data import TokenAnalyzer, train_custom_tokenizer, analyze_and_recommend
from transformers import GPT2TokenizerFast

# Quick analysis with recommendations
report, recommendations = analyze_and_recommend('tinystories', subset_size=10000)
print(report.summary())

# Detailed analysis
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
analyzer = TokenAnalyzer(tokenizer)
report = analyzer.analyze_dataset('tinystories')

# Get recommendations
for rec in analyzer.recommend_vocab_size(report):
    print(rec.summary())

# Train a custom BPE tokenizer
tokenizer_path = train_custom_tokenizer(
    'tinystories',
    vocab_size=4096,
    subset_size=50000,
)

# Use the trained tokenizer
from transformers import PreTrainedTokenizerFast
custom_tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
```

## CLI Tools

### tools/analyze_tokens.py

```bash
# Basic analysis
python tools/analyze_tokens.py --dataset tinystories

# With subset for speed
python tools/analyze_tokens.py --dataset tinystories --subset 10000

# Show recommendations
python tools/analyze_tokens.py --dataset tinystories --recommendations

# Compare multiple datasets
python tools/analyze_tokens.py --dataset tinystories tiny-textbooks --compare

# Train custom tokenizer
python tools/analyze_tokens.py --dataset tinystories --train-tokenizer 4096

# Export CDF data for plotting
python tools/analyze_tokens.py --dataset tinystories --export-cdf cdf.csv
```

## Key Insights

From analyzing TinyStories with GPT-2 tokenizer:

| Coverage | Tokens Needed | vs GPT-2 (50,257) |
|----------|---------------|-------------------|
| 90% | 950 | 98% reduction |
| 95% | 2,000 | 96% reduction |
| 99% | 4,500 | 91% reduction |
| 99.9% | 11,000 | 78% reduction |

For small-scale transformer experiments, a 4,096-token vocabulary provides excellent coverage while dramatically reducing embedding parameters.
