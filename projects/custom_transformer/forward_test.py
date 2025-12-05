#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models.custom_transfromer.CustomTransformer import CustomTransformer

# Minimal config mock (or use your real config class)
class MockConfig:
    def get_device(self): return None
    def get_dtype(self): return None

model = CustomTransformer(MockConfig())

# Fake batch: 2 sequences, 16 tokens each
batch_size, seq_len = 2, 16
tokens = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=model.device)

# Create fake targets (next token prediction - shift by 1)
targets = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=model.device)

print("=" * 50)
print("FORWARD PASS")
print("=" * 50)

# Forward pass
logits = model.forward(tokens)

print(f"Input shape:  {tokens.shape}")       # [2, 16]
print(f"Output shape: {logits.shape}")       # [2, 16, vocab_size]
print(f"Output dtype: {logits.dtype}")       # bfloat16
print(f"Any NaNs:     {torch.isnan(logits).any().item()}")

print("\n" + "=" * 50)
print("LOSS COMPUTATION")
print("=" * 50)

# Compute cross-entropy loss
logits_flat = logits.view(-1, model.vocab_size)  # [batch*seq, vocab]
targets_flat = targets.view(-1)                   # [batch*seq]
loss = F.cross_entropy(logits_flat, targets_flat)

print(f"Loss: {loss.item():.4f}")
print(f"Expected random loss: ~{torch.log(torch.tensor(model.vocab_size)).item():.4f}")

print("\n" + "=" * 50)
print("BACKWARD PASS")
print("=" * 50)

# Compute loss gradient (cross-entropy gradient = probs - one_hot)
probs = F.softmax(logits, dim=-1)
one_hot = F.one_hot(targets, num_classes=model.vocab_size).float().to(logits.dtype)
loss_gradient = (probs - one_hot) / (batch_size * seq_len)  # normalize by batch

print(f"Loss gradient shape: {loss_gradient.shape}")  # [batch, seq, vocab]

# Run backward pass
try:
    model.backward(loss_gradient)
    print("Backward pass completed!")

    # Check gradient norms
    print("\n" + "=" * 50)
    print("GRADIENT NORMS")
    print("=" * 50)

    # Check for NaN gradients
    has_nan = False
    for key in model.cache.gradients:
        grad = model.cache.gradients[key]
        if torch.isnan(grad).any():
            print(f"  WARNING: NaN in gradient '{key}'")
            has_nan = True

    if not has_nan:
        print("\nNo NaN gradients detected!")

except Exception as e:
    print(f"Backward pass failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("PARAMETER UPDATE TEST")
print("=" * 50)

# Store original parameter values
orig_W1 = model.W1[0].clone()

# Update parameters
learning_rate = 0.001
try:
    model.update_parameters(learning_rate)
    print(f"Parameters updated with lr={learning_rate}")

    # Verify weights changed
    weight_diff = (model.W1[0] - orig_W1).abs().mean().item()
    print(f"Mean W1[0] change: {weight_diff:.8f}")

    if weight_diff > 0:
        print("Parameters successfully updated!")
    else:
        print("WARNING: Parameters did not change")

except Exception as e:
    print(f"Parameter update failed: {e}")
    import traceback
    traceback.print_exc()
