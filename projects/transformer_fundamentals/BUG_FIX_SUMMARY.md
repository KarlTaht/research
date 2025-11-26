# Bug Fix Summary: Training Loss vs Evaluation Quality Mismatch

## Problem

Training showed very low loss (0.084) but evaluation produced complete garbage:
- Generated outputs: "Jay Jay Jay..." or "45 45 45..."
- Expected: Meaningful dialogue summaries

## Root Cause: Off-by-One Error in Loss Calculation

The model was training to predict the **current token** instead of the **next token**.

### How It Worked (WRONG):

```python
# Training example:
input = [dialogue tokens] [SEP] [400, 500, 600]  # summary tokens
tgt_ids = [400, 500, 600]  # fed to decoder

# Decoder processing:
# Position 0: sees token 400, outputs logits_0
# Position 1: sees tokens 400, 500, outputs logits_1
# Position 2: sees tokens 400, 500, 600, outputs logits_2

# OLD BUGGY LOSS CALCULATION:
labels = [400, 500, 600]
loss = compare(logits_0, 400)  # ❌ logits_0 should predict 500, not 400!
     + compare(logits_1, 500)  # ❌ logits_1 should predict 600, not 500!
     + compare(logits_2, 600)  # ❌ logits_2 should predict EOS, not 600!
```

**Result**: Model learned to "copy" the token it just saw, achieving artificially low loss without learning actual sequence generation.

### How It Works Now (CORRECT):

```python
# NEW FIXED LOSS CALCULATION:
# Shift logits and labels to align next-token prediction
shift_logits = logits[:, :-1, :]  # Remove last position
shift_labels = labels[:, 1:]      # Remove first position

loss = compare(shift_logits_0, 500)  # ✅ logits_0 predicts 500
     + compare(shift_logits_1, 600)  # ✅ logits_1 predicts 600
     + compare(shift_logits_2, EOS)  # ✅ logits_2 predicts EOS
```

**Result**: Model now learns actual next-token prediction, enabling proper generation.

## Changes Made

### 1. Fixed Loss Calculation (`common/models/reference_transformer/model.py:275-288`)

```python
# CRITICAL: Shift labels to align with autoregressive prediction
# logits[:, i] should predict labels[:, i+1] (next token prediction)
shift_logits = logits[:, :-1, :].contiguous()  # Remove last position
shift_labels = tgt_labels[:, 1:].contiguous()  # Remove first position

loss = self.criterion(
    shift_logits.reshape(-1, self.vocab_size),
    shift_labels.reshape(-1)
)
```

### 2. Fixed Model Config Saving (`common/models/reference_transformer/model.py:71-83`)

Model now properly saves all hyperparameters to checkpoint for easy loading.

### 3. Fixed Checkpoint Loading (`projects/transformer_fundamentals/evaluate.py:36-81`)

- Loads both old (empty config) and new (full config) checkpoints
- Infers missing hyperparameters from state_dict when needed
- Extracts: `d_model`, `d_ff`, `n_heads`, `n_encoder_layers`, `n_decoder_layers`, `max_seq_len`

### 4. Fixed Generation (`projects/transformer_fundamentals/evaluate.py:92-162`)

Seeds decoder with BOS token to avoid empty tensor errors.

## Impact

**Old checkpoints are now invalid** - they were trained with the buggy loss calculation and won't generate properly even with the fix applied at evaluation time.

**Action Required**: Re-train the model from scratch with the fixed code.

## Testing

Created comprehensive test suite (`test_evaluate.py`) covering:
- ✅ Checkpoint loading (new and old format)
- ✅ Config inference from state_dict
- ✅ Model forward pass
- ✅ Generation without crashes
- ✅ State dict compatibility

All 6 tests pass.

## Next Steps

1. **Re-train the model** with the bug fix
2. Monitor that loss increases initially (expected, as task is now harder)
3. Verify generation quality improves as training progresses
4. Expected loss will be higher (~1-3 range) but outputs will be meaningful
