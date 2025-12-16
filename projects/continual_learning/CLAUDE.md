# Continual Learning Project

Experiment to study catastrophic forgetting in transformers using domain-specific corpora (automotive, food) extracted from FineWeb.

## Quick Reference

```bash
# Train from scratch
python train.py --config configs/auto_236m.yaml

# Resume training (edit num_epochs in config first if needed)
python train.py --config configs/auto_236m.yaml --resume checkpoints/automotive_236m/latest.pt

# Interactive chat with checkpoint
python validate.py --checkpoint checkpoints/automotive_236m/best.pt --config configs/auto_236m.yaml --chat

# Single prompt generation
python validate.py --checkpoint checkpoints/automotive_236m/best.pt --config configs/auto_236m.yaml --prompt "The engine"
```

## Project Structure

```
continual_learning/
├── train.py              # Main training script
├── validate.py           # Interactive testing / generation
├── config_summary.py     # Config analysis utility
├── configs/              # Training configurations
│   ├── auto_236m.yaml    # 236M param model on automotive
│   ├── auto_70m.yaml     # 70M param model on automotive
│   ├── food_150m.yaml    # 150M param model on food
│   ├── food_80m.yaml     # 80M param model on food
│   └── tinystories.yaml  # Small validation config
├── models/
│   ├── __init__.py
│   └── torch_transformer.py  # TorchTransformer implementation
├── data/                 # Symlinks to assets/datasets/
│   ├── corpus_automotive -> ../../../assets/datasets/.../corpus_automotive
│   └── corpus_food -> ../../../assets/datasets/.../corpus_food
├── checkpoints/          # Saved model checkpoints
│   └── {experiment_name}/
│       ├── best.pt
│       ├── latest.pt
│       └── checkpoint_epoch{N}_step{S}.pt
├── logs/                 # Training logs
└── plan.md               # Detailed experiment plan
```

## Model Architecture

`TorchTransformer` - Modern decoder-only transformer:

| Component | Implementation |
|-----------|----------------|
| Normalization | RMSNorm (pre-norm) |
| Positions | Learned absolute embeddings |
| Attention | Causal multi-head attention |
| FFN | SwiGLU activation |
| Output | Weight-tied with embeddings |
| Precision | bfloat16 by default |

### Model Configurations

| Config | d_model | Blocks | Heads | FFN | Context | ~Params |
|--------|---------|--------|-------|-----|---------|---------|
| tinystories | 256 | 6 | 4 | 1024 | 256 | 5.8M |
| auto_70m | 512 | 8 | 8 | 2048 | 1024 | 70M |
| auto_236m | 1024 | 16 | 16 | 4096 | 1024 | 236M |

## Data

### Prepared Corpora

| Corpus | Train Docs | Val Docs | ~Tokens | Source |
|--------|------------|----------|---------|--------|
| Automotive | 179,897 | 19,988 | 111M | FineWeb (cars, vehicles) |
| Food | 199,280 | 22,142 | 112M | FineWeb (recipes, cooking) |

### Tokenizer

Combined BPE tokenizer trained on both corpora:
- **Location:** `assets/models/tokenizers/combined_bpe_32768/`
- **Vocab size:** 32,768
- **Coverage:** ~98% on both domains

```python
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    'assets/models/tokenizers/combined_bpe_32768'
)
```

### Pre-tokenized Caches

Located at `data/corpus_{domain}/train_tokenized/combined_bpe_32768_v32768_len1024/`

Enables ~1s training startup vs ~70s without cache.

## Configuration Format

```yaml
experiment_name: "automotive_236m"

model:
  d_model: 1024
  n_blocks: 16
  n_heads: 16
  d_ffn: 4096
  max_seq_len: 1024
  dtype: bfloat16

data:
  corpus_dir: "data/corpus_automotive"
  tokenizer: "combined_bpe_32768"
  max_length: 1024

training:
  batch_size: 4
  gradient_accumulation_steps: 16  # effective = 64
  learning_rate: 0.0003
  min_learning_rate: 0.00003
  lr_decay: cosine
  warmup_ratio: 0.05
  weight_decay: 0.01
  num_epochs: 3
  max_grad_norm: 1.0
  checkpoint_interval_minutes: 30

evaluation:
  generation_prompts:
    - "The engine"
  max_generation_length: 100
  temperature: 0.8
```

## Important Notes

### Resuming Training

When resuming, the checkpoint stores the **completed** epoch number. If checkpoint says `epoch: 3` and config says `num_epochs: 3`, training will immediately exit (no epochs to run).

**To continue training:** Edit `num_epochs` in the config to a higher value before resuming.

### Checkpoint Files

- `latest.pt` - Most recent checkpoint (time-based or epoch-end)
- `best.pt` - Lowest validation loss
- `checkpoint_epoch{N}_step{S}.pt` - Periodic saves (pruned to keep recent only)

### Data Loading

Two modes supported:
1. **Pre-tokenized (fast):** Set `corpus_dir` in config - loads from cache in ~1s
2. **HuggingFace datasets:** Set `dataset` in config - uses registry with on-demand tokenization

## Experiment Workflow

### Phase 1: Baseline Training

```bash
# Train on automotive corpus
python train.py --config configs/auto_236m.yaml

# Train on food corpus
python train.py --config configs/food_150m.yaml
```

### Phase 2: Sequential Training (Forgetting)

```bash
# Train A, then continue on B (measure forgetting of A)
python train.py --config configs/food_150m.yaml \
    --resume checkpoints/automotive_236m/best.pt
```

### Phase 3: Evaluation

```bash
# Interactive testing
python validate.py --checkpoint checkpoints/automotive_236m/best.pt \
    --config configs/auto_236m.yaml --chat
```

## Continual Learning Baselines

| Baseline | Training | Purpose |
|----------|----------|---------|
| Sequential A→B | Train(A) → Train(B) | Measure forgetting of A |
| Sequential B→A | Train(B) → Train(A) | Measure forgetting of B |
| Combined | Train(A+B) | Upper bound (no forgetting) |

## Next Steps (from plan.md)

1. Complete model validation on larger dataset
2. QA test generation with Claude Haiku (discriminative evaluation)
3. Run continual learning baseline experiments
4. Intervention experiments (replay, EWC, layer freezing)

## Related Files

- `common/models/base.py` - BaseLanguageModel abstract class
- `common/training/` - Training utilities, CheckpointManager
- `common/utils/training_logger.py` - TrainingLogger for experiment tracking
- `common/data/` - Dataset loading, tokenization utilities
