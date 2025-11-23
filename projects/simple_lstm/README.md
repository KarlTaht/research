# Simple LSTM Language Model

A minimal but complete example demonstrating the ML research infrastructure with an LSTM-based language model trained on TinyStories.

## What This Demonstrates

This project showcases the full research workflow:

1. **Base Model Class**: LSTM extends `BaseLanguageModel` from `common/models/`
2. **Evaluation Framework**: Uses `Evaluator` from `common/training/`
3. **Experiment Tracking**: Results saved to Parquet files and queryable with DuckDB
4. **Dataset Integration**: Loads TinyStories dataset with HuggingFace tokenization
5. **Complete Pipeline**: Training → Evaluation → Results storage

## Model Architecture

- **Type**: LSTM (Long Short-Term Memory)
- **Tokenization**: GPT-2 BPE tokenizer (subword level)
- **Dataset**: TinyStories (simple narratives)

**Architecture**:
- Embedding layer (256-dim)
- 2-layer LSTM (512 hidden units)
- Linear projection to vocabulary
- ~5.6M parameters

## Quick Start

### 1. Prerequisites

Ensure you have:
- TinyStories dataset downloaded: `~/research/assets/datasets/roneneldan/TinyStories`
- Dependencies installed: `source .venv/bin/activate && uv pip install -e ".[all]"`

If you don't have TinyStories yet:
```bash
cd ~/research
source .venv/bin/activate
python tools/download_hf_dataset.py --name roneneldan/TinyStories
```

### 2. Train the Model

```bash
cd ~/research/projects/simple_lstm
source ../../.venv/bin/activate
python train.py
```

This will:
- Load 10,000 training examples (configurable in `config.yaml`)
- Train for 10 epochs (~5-10 minutes on CPU, faster on GPU)
- Save checkpoints to `checkpoints/`
- Save training metrics to experiment database

**Output**:
- Checkpoints: `checkpoints/lstm_epoch_*.pt`, `checkpoints/lstm_final.pt`
- Experiment results: `~/research/assets/outputs/experiments/simple_lstm_tinystories.parquet`

### 3. Evaluate the Model

```bash
python evaluate.py
```

This will:
- Load the final checkpoint
- Evaluate on validation set
- Generate sample text outputs
- Save evaluation metrics to experiment database

## Configuration

Edit `config.yaml` to modify:

```yaml
model:
  embedding_dim: 256      # Token embedding dimension
  hidden_dim: 512         # LSTM hidden state dimension
  num_layers: 2           # Number of LSTM layers
  dropout: 0.2            # Dropout probability

data:
  max_length: 128         # Maximum sequence length
  use_subset: true        # Use subset for quick testing
  subset_size: 10000      # Number of training examples

training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 10
  save_every: 5           # Save checkpoint frequency
```

## Querying Results

After training, query your experiment results:

```bash
# View all simple_lstm experiments
cd ~/research
python tools/query_experiments.py --sql "SELECT * FROM experiments WHERE experiment_name LIKE 'simple_lstm%'"

# Get best epoch by perplexity
python tools/query_experiments.py --best val_perplexity --top 5

# Compare with other experiments
python tools/query_experiments.py --compare simple_lstm_tinystories simple_lstm_eval
```

Or programmatically:

```python
from common.utils import query_experiments, load_experiment

# Load training results
results = load_experiment('simple_lstm_tinystories')
print(results[['epoch', 'train_loss', 'val_perplexity']])

# Query across all experiments
best = query_experiments("""
    SELECT experiment_name, MIN(val_perplexity) as best_perplexity
    FROM experiments
    GROUP BY experiment_name
    ORDER BY best_perplexity
""")
```

## Project Structure

```
projects/simple_lstm/
├── README.md           # This file
├── config.yaml         # Model and training configuration
├── model.py            # LSTM implementation (extends BaseLanguageModel)
├── train.py            # Training script
├── evaluate.py         # Evaluation script
└── checkpoints/        # Model checkpoints (created during training)
```

## Expected Results

With default settings (10K examples, 10 epochs):
- **Training time**: ~5-10 minutes (CPU), ~2-3 minutes (GPU)
- **Final validation perplexity**: ~15-25 (varies by run)
- **Model size**: ~5.6M parameters
- **Checkpoint size**: ~90 MB

## Extending This Example

This minimal example can be extended:

1. **Larger dataset**: Set `use_subset: false` in config.yaml to use full TinyStories (2.1M examples)
2. **More epochs**: Increase `num_epochs` for better performance
3. **Bigger model**: Increase `hidden_dim` or `num_layers`
4. **Different dataset**: Modify `train.py` to load tiny-textbooks or fineweb
5. **Advanced evaluation**: Add BLEU score, generation diversity metrics

## Reusing the Infrastructure

The `common/` package provides reusable components:

**Base Model** (`common/models/base.py`):
```python
from common.models import BaseLanguageModel

class MyModel(BaseLanguageModel):
    def forward(self, input_ids, labels=None):
        # Your implementation
        pass
```

**Evaluator** (`common/training/evaluator.py`):
```python
from common.training import Evaluator

evaluator = Evaluator(model, device)
metrics = evaluator.evaluate(dataloader)
```

**Experiment Storage** (`common/utils/experiment_storage.py`):
```python
from common.utils import save_experiment

save_experiment('my_experiment', results_df, metadata={...})
```

## Troubleshooting

**Dataset not found**:
```bash
python ~/research/tools/download_hf_dataset.py --name roneneldan/TinyStories
```

**CUDA out of memory**:
- Reduce `batch_size` in config.yaml
- Or train on CPU (slower but works)

**Import errors**:
```bash
source ~/research/.venv/bin/activate
cd ~/research
uv pip install -e ".[all]"
```

## Next Steps

- Try training on tiny-textbooks for educational content
- Implement a Transformer model using the same infrastructure
- Add more evaluation metrics (BLEU, diversity)
- Experiment with different hyperparameters
- Compare LSTM vs. other architectures

---

This example demonstrates a complete end-to-end ML research workflow. All components are reusable for future projects!
