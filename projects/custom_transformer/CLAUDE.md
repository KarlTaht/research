# CustomTransformer Project

## Overview

This project tests and trains the educational CustomTransformer - a decoder-only transformer with **manual backpropagation** (no PyTorch autograd).

## Project Structure

```
projects/custom_transformer/
├── model/              -> symlink to common/models/custom_transfromer/
├── tests/
│   ├── test_loss_decreases.py   # Functional tests (pytest)
│   └── forward_test.py          # Manual test script
├── configs/
│   └── tiny-textbooks-small.yaml
├── train.py            # Training script
├── evaluate.py         # Evaluation and generation script
├── config_summary.py   # Model parameter summary tool
├── PLAN.md             # Implementation plan
└── CLAUDE.md           # This file
```

## Key Commands

```bash
# View model configuration and parameter count
python config_summary.py configs/tiny-textbooks-small.yaml

# Run functional tests
python -m pytest tests/test_loss_decreases.py -v -s --override-ini="addopts="

# Run manual forward test
python tests/forward_test.py

# Train on tiny-textbooks
python train.py --config configs/tiny-textbooks-small.yaml

# Evaluate a checkpoint
python evaluate.py --checkpoint checkpoints/custom_transformer_epoch_5.pt --config configs/tiny-textbooks-small.yaml

# Evaluate with text generation
python evaluate.py --checkpoint checkpoints/custom_transformer_epoch_5.pt --config configs/tiny-textbooks-small.yaml --generate
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
    vocab_size=50257,  # GPT-2 tokenizer size
    d_model=256,
    n_blocks=4,
    n_heads=4,
    d_ffn=512,
    max_seq_len=256,
)

# For inference (returns dict like BaseLanguageModel)
outputs = model.forward(input_ids, labels=labels)
loss = outputs['loss']

# For training (handles manual backprop internally)
result = model.train_step(input_ids, labels, learning_rate=0.001)
```

## Configuration

Edit `config.yaml` to adjust:

- **model**: d_model, n_blocks, n_heads, d_ffn, max_seq_len
- **data**: dataset, text_column, max_length, subset_size
- **training**: batch_size, learning_rate, num_epochs

## Notes

- Uses **float32** in tests for numerical stability (bfloat16 can cause NaN with high learning rates)
- The wrapper's `train_step()` replaces the typical `loss.backward()` + `optimizer.step()` pattern
- Checkpoints save raw tensors (not nn.Parameter state_dict)

## Related Files

- Model implementation: `common/models/custom_transfromer/CustomTransformer.py`
- Wrapper: `common/models/custom_transfromer/wrapper.py`
- Config class: `common/models/custom_transfromer/config.py`
- Model docs: `common/models/custom_transfromer/CLAUDE.md`
