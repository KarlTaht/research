# CustomTransformer Project

## Overview

This project tests and trains the educational CustomTransformer - a decoder-only transformer with **manual backpropagation** (no PyTorch autograd).

**Implementation Status**: Complete (see PLAN.md for details)

## Project Structure

```
projects/custom_transformer/
├── tests/
│   ├── test_loss_decreases.py   # Functional tests (pytest) - 10 tests
│   └── forward_test.py          # Manual test script
├── configs/
│   ├── tinystories.yaml                  # GPT-2 tokenizer (50k vocab)
│   ├── tinystories_custom_tokenizer.yaml # Custom BPE tokenizer (4k vocab)
│   └── tiny-textbooks-small.yaml
├── train.py            # Training script with checkpointing, logging, evaluation
├── evaluate.py         # Evaluation and generation script
├── config_summary.py   # Model parameter summary tool
├── PLAN.md             # Implementation plan (COMPLETE)
└── CLAUDE.md           # This file
```

## Key Commands

```bash
# View model configuration and parameter count
python config_summary.py configs/tinystories.yaml

# Run functional tests
python -m pytest tests/test_loss_decreases.py -v -s --override-ini="addopts="

# Run manual forward test
python tests/forward_test.py

# Train on TinyStories with GPT-2 tokenizer
python train.py --config configs/tinystories.yaml

# Train with custom tokenizer (smaller vocab = faster training, smaller model)
python train.py --config configs/tinystories_custom_tokenizer.yaml

# Train on tiny-textbooks
python train.py --config configs/tiny-textbooks-small.yaml

# Resume training from checkpoint
python train.py --config configs/tinystories.yaml --resume checkpoints/latest.pt

# Skip Claude coherence evaluation (faster)
python train.py --config configs/tinystories.yaml --no-coherence-eval

# Evaluate a checkpoint
python evaluate.py --checkpoint checkpoints/best.pt --config configs/tinystories.yaml

# Evaluate with text generation
python evaluate.py --checkpoint checkpoints/best.pt --config configs/tinystories.yaml --generate

# Quick evaluation (limited batches)
python evaluate.py --checkpoint checkpoints/best.pt --config configs/tinystories.yaml --max-batches 50
```

## Model Architecture

The CustomTransformer uses **manual backpropagation** for educational purposes:

- **Forward**: `model.forward(tokens)` → logits
- **Backward**: `model.backward(loss_gradient)` → computes gradients
- **Update**: `model.update_parameters(lr)` → SGD update

This differs from standard PyTorch models that use autograd.

## Wrapper Class

`CustomTransformerWrapper` adapts the model to a BaseLanguageModel-like interface:

```python
from common.models.custom_transfromer import CustomTransformerWrapper

model = CustomTransformerWrapper(
    vocab_size=4096,   # Custom tokenizer size (or 50257 for GPT-2)
    d_model=256,
    n_blocks=4,
    n_heads=4,
    d_ffn=512,
    max_seq_len=256,
    dtype=torch.bfloat16,  # Optional, defaults to bfloat16
)

# For inference (returns dict like BaseLanguageModel)
outputs = model.forward(input_ids, labels=labels)
loss = outputs['loss']

# For training (handles manual backprop internally)
result = model.train_step(input_ids, labels, learning_rate=0.001, max_grad_norm=1.0)
# result = {'loss': float, 'status': 'ok'|'nan_detected'|'nan_gradient', 'grad_norm': float}

# Text generation
generated = model.generate(input_ids, max_length=100, temperature=0.8, top_k=50)

# Checkpointing
model.save_checkpoint('checkpoint.pt', epoch=5, train_loss=1.23)
model.load_checkpoint('checkpoint.pt')

# Or create from checkpoint directly
model = CustomTransformerWrapper.from_checkpoint('checkpoint.pt')
```

## Configuration

Edit config YAML files to adjust:

- **model**: d_model, n_blocks, n_heads, d_ffn, max_seq_len
- **data**: dataset, tokenizer, max_length, subset_size, val_subset_size
- **training**: batch_size, learning_rate, num_epochs, max_grad_norm, log_every, eval_every
- **evaluation**: generation_prompts, max_generation_length, temperature, top_k, evaluate_coherence

### Tokenizer Options

The training scripts support both GPT-2 and custom tokenizers:

```yaml
data:
  # Option 1: GPT-2 tokenizer (default if omitted)
  tokenizer: "gpt2"  # vocab_size=50,257

  # Option 2: Custom tokenizer (path relative to assets/models/tokenizers/)
  tokenizer: "tinystories_bpe_4096"  # vocab_size=4,096
```

**Why use a custom tokenizer?**
- 92% smaller embedding matrices (4k vs 50k vocab)
- Faster training due to smaller output projection
- Dataset-specific tokens provide better coverage
- The `tinystories_bpe_4096` tokenizer achieves 99%+ coverage on TinyStories

**Custom tokenizer location**: `assets/models/tokenizers/tinystories_bpe_4096/`

### Supported Datasets (via registry)

- `tinystories` - Simple children's stories (recommended for testing)
- `tiny-textbooks` - Educational textbooks
- `tiny-strange-textbooks` - Diverse textbooks
- `tiny-codes` - Code snippets

## Training Features

The training script (`train.py`) includes:

1. **Multi-dataset support** via `common.data` registry
2. **Custom tokenizer support** - GPT-2 or dataset-specific BPE tokenizers
3. **Checkpoint management** with automatic saving and resume capability
4. **Training logger** with TFLOP estimation
5. **Advanced evaluator** with:
   - Perplexity computation
   - Text generation samples
   - Optional Claude Haiku coherence scoring
6. **Numerical stability**:
   - bfloat16 with float32 precision mixing for loss
   - Gradient clipping
   - NaN detection with skip-update recovery

## Notes

- Uses **bfloat16** by default with **float32** precision mixing for numerical stability
- Tests use **float32** to avoid NaN issues with high learning rates
- The wrapper's `train_step()` replaces the typical `loss.backward()` + `optimizer.step()` pattern
- Checkpoints include model config for easy restoration
- Custom tokenizers must match the vocab_size used during training

## Related Files

- Model implementation: `common/models/custom_transfromer/CustomTransformer.py`
- Wrapper: `common/models/custom_transfromer/wrapper.py`
- Config class: `common/models/custom_transfromer/config.py`
- Model docs: `common/models/custom_transfromer/CLAUDE.md`
- Checkpoint manager: `common/training/checkpoint_manager.py`
- Advanced evaluator: `common/training/advanced_evaluator.py`
- Training logger: `common/utils/training_logger.py`
- Custom tokenizer: `assets/models/tokenizers/tinystories_bpe_4096/`
