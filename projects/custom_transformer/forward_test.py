#!/usr/bin/env python3

import torch
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
tokens = torch.randint(0, 128, (2, 16), device=model.device)

# Forward pass
logits = model.forward(tokens)

print(f"Input shape:  {tokens.shape}")      # [2, 16]
print(f"Output shape: {logits.shape}")       # Should be [2, 16, 128]
print(f"Output dtype: {logits.dtype}")       # Should be bfloat16
print(f"Any NaNs:     {torch.isnan(logits).any().item()}")
