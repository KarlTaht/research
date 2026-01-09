# Routed Transformer Grokking Research

## Overview

This document summarizes findings from experiments investigating whether **routed networks** (learned routing through parallel FFN heads) can accelerate grokking or enable learning with less training data.

**Baseline reference**: Standard transformer with wd=0.5-1.0, p=113, lr=3e-4, 30/70 train/test split.

---

## Key Findings

### 1. Routing Regularization is Required for Grokking

Without regularization, the routed transformer **fails to grok** entirely:

| Configuration | Test Accuracy | Result |
|--------------|---------------|--------|
| Routed, no regularization | ~29% | No grokking |
| Routed, entropy reg (λ=0.10) | ~100% | Grokking |
| Routed, sparsity reg (λ=0.10) | ~100% | Grokking |

**Why?** Without regularization, routing weights remain near-uniform (high entropy), preventing head specialization. The network cannot learn to route different computations to different heads.

### 2. λ_routing Has a Threshold Effect

Not all regularization strengths work. Lower values fail to induce grokking:

| λ_routing | Test Accuracy | Grokking? |
|-----------|---------------|-----------|
| 0.01 | ~29% | No |
| 0.03 | ~29% | No |
| 0.05 | ~29% | No |
| 0.10 | ~100% | Yes |

This suggests the routing gate needs **strong pressure** to specialize. Weak regularization is insufficient.

### 3. Routed Transformer is Slower Than Standard Transformer

On single-operation modular arithmetic (p=113, 30/70 split):

| Model | Epochs to 95% Test Acc | Relative Speed |
|-------|------------------------|----------------|
| Standard Transformer | ~4,600 | 1x (baseline) |
| Routed (entropy λ=0.10) | ~17,800 | ~0.26x (4x slower) |
| Routed (sparsity λ=0.10) | ~17,800 | ~0.26x (4x slower) |

The routed architecture adds complexity without speed benefit for single-operation tasks.

### 4. Routing Weights Must Be Excluded from Weight Decay

Similar to embeddings and biases, routing gate weights should **not** have weight decay applied:

```python
# No weight decay for: biases, LayerNorm, embeddings, routing gates
if (
    'bias' in name
    or 'ln' in name.lower()
    or 'embedding' in name
    or 'gate' in name          # Routing gate weights
    or 'state_proj' in name    # Routing state projection
    or 'input_proj' in name    # Routing input projection
):
    no_decay_params.append(param)
```

**Why?** High weight decay pushes routing gate weights toward zero, causing the softmax to output uniform distributions. This prevents any head from specializing.

### 5. Multi-Operation Learning: Routing Fails Completely

When training on **both** addition and multiplication simultaneously (p=113, 30k epochs):

| Model | Best Test Acc | Final Test Acc | Train Acc | Result |
|-------|---------------|----------------|-----------|--------|
| Standard Transformer | 82.72% | 82.72% | 100% | Progressing toward grokking |
| Routed (entropy λ=0.10) | 25.39% | 17.59% | 100% | Complete failure |

**Key observations:**
- Multi-op is harder than single-op (neither fully grokked in 30k epochs)
- Standard transformer is making progress (82.72% and climbing)
- Routed transformer **completely failed** - worse than random for p=113 (~0.88%)
- The routing mechanism appears to **hurt** rather than help multi-task learning

**Hypothesis for failure:** The routing gate may be learning to distinguish operations but routing them to heads that haven't specialized properly. The added complexity interferes with the grokking dynamics rather than enabling operation-specific circuits.

---

## Architecture

### RoutedGrokTransformer

```
RoutedGrokTransformer
├── Token Embedding: vocab_size → d_model
├── Positional Embedding: seq_len → d_model
├── N × TransformerBlock
│   ├── Pre-LayerNorm → MultiHeadAttention → Residual
│   └── Pre-LayerNorm → RoutedFFN → Residual
└── LayerNorm → Unembedding (d_model → p classes)
```

### RoutedFFN (Key Innovation)

Instead of a single FFN, uses N parallel FFN heads with learned routing:

```
Input x ───┬───► FFN_1 ───┐
           ├───► FFN_2 ───┤
           ├───► FFN_3 ───┼───► Weighted Sum ───► Output
           └───► FFN_4 ───┘
                   ▲
           Routing Gate(x)
           (softmax weights)
```

Each FFN head is a standard 2-layer MLP: `d_model → 4*d_model → d_model`

The routing gate learns to assign different inputs to different heads based on content.

---

## Experimental Setup

### Single-Operation Experiments

- **Task**: Modular addition (a + b) mod p
- **Prime**: p = 113 (12,769 total examples)
- **Split**: 30% train / 70% test
- **Model**: 4 layers, 4 attention heads, d_model=128
- **Training**: AdamW, lr=3e-4, wd=1.0, warmup=500 epochs

### Multi-Operation Experiments (In Progress)

- **Task**: Both addition AND multiplication
- **Dataset**: 2 × p² examples (one per operation)
- **Hypothesis**: Routing may help by specializing heads per operation
- **Vocab**: p residues + add_token + mul_token + eq_token

---

## Open Questions

1. ~~**Multi-task advantage?**~~ **ANSWERED: No.** Routed transformer fails completely on multi-op while standard transformer progresses normally.

2. **Why does routing hurt multi-task?** Is the routing gate interfering with gradient flow? Is λ=0.10 wrong for multi-op?

3. **Optimal λ_routing?** Is there a sweet spot between λ=0.05 (no grokking) and λ=0.10 (slow grokking)?

4. **Alternative regularizers?** Would other regularization schemes (Gini coefficient, consistency) work better?

5. **Smaller train splits?** Can routing enable grokking with less training data (20/80 or 10/90)?

6. **Is routing fundamentally incompatible with grokking?** The dynamics of delayed generalization may conflict with learned routing.

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/models.py` | RoutedGrokTransformer, RoutedFFN implementations |
| `src/trainer.py` | Training loop with routing weight decay exclusion |
| `src/data.py` | MultiOpSequenceDataset for multi-operation learning |
| `src/losses.py` | Routing regularizers (entropy, sparsity, gini) |
| `configs/multi_op.yaml` | Multi-operation experiment config |

---

## Conclusions

1. **Routing alone is not enough** - regularization is required to force specialization
2. **Strong regularization needed** - λ=0.10 works, lower values fail
3. **Single-task: no advantage** - routed transformer is 4x slower than standard
4. **Multi-task: routing hurts** - routed transformer fails completely (17% vs 83%)

**Overall assessment:** The routed transformer architecture does not appear beneficial for grokking tasks. While it can eventually grok single-operation tasks with proper regularization, it is significantly slower than standard transformers. For multi-operation tasks, routing actively interferes with learning.

The hypothesis that routing would enable operation-specific head specialization was **not supported** by experiments. The additional complexity of the routing mechanism appears to conflict with the dynamics that enable grokking (delayed generalization through weight decay and implicit regularization).
