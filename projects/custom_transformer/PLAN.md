# CustomTransformer Testing Infrastructure Plan

## Overview

This plan details how to test the educational `CustomTransformer` (which uses manual backpropagation) by adapting it to work with the existing `BaseLanguageModel` infrastructure, creating functional tests, and setting up training on the tiny-textbooks dataset.

## User Requirements

1. **Make vocab_size configurable** - Modify CustomTransformer to accept vocab_size as a config parameter
2. **Create a thin wrapper class** - Adapts CustomTransformer to BaseLanguageModel interface while keeping manual backprop internally
3. **Functional test** - Validate loss decreases on a minimal dataset (overfitting is acceptable)
4. **Training script + config** - For nampdn-ai/tiny-textbooks dataset
5. **Eval framework** - Build evaluation for the dataset's train/test split

---

## Design Decisions

### 1. How should the wrapper handle manual backprop?

The wrapper will intercept the training loop entirely. It cannot use `loss.backward()` because `CustomTransformer` uses raw tensors (not `nn.Parameter`) and has its own `backward()` method.

**Strategy**: Create a `CustomTransformerWrapper` that:
- Does NOT extend `nn.Module` (no parameters registered)
- `forward()` returns `{'logits', 'loss'}` as expected by BaseLanguageModel interface
- Provides a custom `train_step()` method that handles the manual backprop cycle
- Computes loss internally using cross-entropy on the logits

### 2. What tokenizer to use?

Use GPT-2 tokenizer (vocab_size=50257) and configure CustomTransformer to match.

### 3. How to structure the functional test?

Use pytest with standalone scripts, following the pattern in `tests/test_transformer_training.py`.

### 4. Should the wrapper support `generate()`?

Yes - it will work for inference by calling `forward()` iteratively.

---

## Implementation Phases

### Phase 1: Make vocab_size Configurable

**File**: `common/models/custom_transfromer/CustomTransformer.py`

Modify `__init__` to read from config with defaults:
```python
def __init__(self, config):
    self.vocab_size = config.get('vocab_size', 128)
    self.max_seq_len = config.get('max_seq_len', 128)
    self.n_blocks = config.get('n_blocks', 8)
    self.n_heads = config.get('n_heads', 4)
    self.d_model = config.get('d_model', 128)
    self.d_ffn = config.get('d_ffn', 128)
    # ... rest unchanged
```

Create `CustomTransformerConfig` class with `get()`, `get_device()`, `get_dtype()` methods.

---

### Phase 2: Create Wrapper Class

**New File**: `common/models/custom_transfromer/wrapper.py`

Key methods:
- `forward(input_ids, labels=None)` -> `{'logits', 'loss'}`
- `train_step(input_ids, labels, learning_rate)` -> `{'loss': float}`
- `generate(input_ids, max_length, temperature, top_k, eos_token_id)`
- `save_checkpoint(path)`, `get_model_info()`, `count_parameters()`

The `train_step()` method handles the complete cycle:
1. Forward pass to get logits
2. Compute cross-entropy loss
3. Compute loss gradient: `(softmax(logits) - one_hot(targets)) / (batch * seq)`
4. Call `model.backward(loss_gradient)`
5. Call `model.update_parameters(learning_rate)`

---

### Phase 3: Functional Test

**New File**: `projects/custom_transformer/tests/test_loss_decreases.py`

Tests:
- `test_forward_pass` - Correct output shapes
- `test_forward_with_labels` - Loss computation works
- `test_train_step` - Single step completes
- `test_loss_decreases_on_fixed_batch` - **CRITICAL**: Validates manual backprop works
- `test_no_nan_gradients` - No NaN/Inf in gradients

The key test overfits a fixed batch for 50 steps and asserts `final_loss < initial_loss * 0.5`.

---

### Phase 4: Training Script + Config

**New Files**:
- `projects/custom_transformer/train.py`
- `projects/custom_transformer/config.yaml`

**config.yaml**:
```yaml
model:
  d_model: 256
  n_blocks: 4
  n_heads: 4
  d_ffn: 512
  max_seq_len: 256

data:
  dataset: "nampdn-ai/tiny-textbooks"
  text_column: "textbook"
  max_length: 256
  use_subset: true
  subset_size: 5000

training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 5
  eval_every: 500
  save_every: 1
  log_every: 100
```

**train.py** structure:
1. Load config from YAML
2. Load GPT-2 tokenizer
3. Load/download tiny-textbooks dataset to `assets/datasets/`
4. Tokenize with labels = input_ids (language modeling)
5. Create DataLoaders
6. Initialize CustomTransformerWrapper with `vocab_size=len(tokenizer)`
7. Training loop using `model.train_step()`
8. Periodic evaluation
9. Save checkpoints and experiment results

---

### Phase 5: Evaluation Script

**New File**: `projects/custom_transformer/evaluate.py`

Features:
- Load checkpoint
- Evaluate on test split with perplexity
- Generate sample text with `--generate` flag

---

## Files Summary

### Files to Modify

| File | Change |
|------|--------|
| `common/models/custom_transfromer/CustomTransformer.py` | Make vocab_size, max_seq_len, n_blocks, etc. configurable |

### Files to Create

| File | Purpose |
|------|---------|
| `common/models/custom_transfromer/config.py` | `CustomTransformerConfig` class |
| `common/models/custom_transfromer/wrapper.py` | `CustomTransformerWrapper` class |
| `common/models/custom_transfromer/__init__.py` | Export wrapper and config |
| `projects/custom_transformer/tests/__init__.py` | Empty init for pytest |
| `projects/custom_transformer/tests/test_loss_decreases.py` | Functional tests |
| `projects/custom_transformer/train.py` | Training script |
| `projects/custom_transformer/evaluate.py` | Evaluation script |
| `projects/custom_transformer/config.yaml` | Configuration |

---

## Implementation Order

1. Modify `CustomTransformer.py` for configurable parameters
2. Create `config.py` and `wrapper.py`
3. Create functional test `test_loss_decreases.py`
4. Create `train.py` and `config.yaml`
5. Create `evaluate.py`
6. Update `__init__.py` exports

---

## Potential Challenges

1. **Memory**: With vocab_size=50257, embedding matrices become large (~6.4M params). May need smaller batch size or float16.

2. **Training Speed**: Manual backprop is slower than PyTorch autograd. Consider using smaller data subset.

3. **Dataset Access**: tiny-textbooks may require HuggingFace login.

4. **Numerical Stability**: Large vocab with bfloat16 may have precision issues. Monitor for NaN losses.

---

## Critical File Paths

- `common/models/custom_transfromer/CustomTransformer.py` - Core model to modify
- `common/models/base.py` - Interface pattern to follow
- `projects/simple_lstm/train.py` - Training script pattern
- `tests/test_transformer_training.py` - Test pattern
- `common/data/hf_utils.py` - Dataset download utilities
