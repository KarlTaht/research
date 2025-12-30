# CLAUDE.md - Least Action Learning

## Project Overview

Research project investigating **grokking** (delayed generalization) in neural networks, with a focus on routing mechanisms that learn efficient computation paths. The project implements both baseline MLPs and transformers for modular arithmetic tasks.

## Key Concepts

### Grokking
The phenomenon where neural networks first memorize training data (high train acc, low test acc), then suddenly generalize after extended training. Key factors:
- **Weight decay**: Critical for grokking (typically 1.0-2.0 for this task)
- **Full-batch training**: Standard for grokking experiments
- **Modular arithmetic**: Clean mathematical structure enables analysis

### Modular Arithmetic Task
Given inputs `(a, b)` where `a, b ∈ {0, ..., p-1}` for prime `p`, predict `(a ⊕ b) mod p` where `⊕` is addition or multiplication.

## Project Structure

```
projects/least_action_learning/
├── configs/                    # YAML experiment configurations
│   ├── baseline.yaml          # MLP baseline (p=113)
│   ├── validate.yaml          # Quick validation (p=17, MLP)
│   ├── transformer.yaml       # Transformer (p=113)
│   └── validate_transformer.yaml  # Quick transformer validation (p=17)
├── scripts/
│   └── train.py               # Main training entry point
├── src/
│   ├── __init__.py            # Package exports
│   ├── data.py                # Dataset classes
│   ├── models.py              # Model architectures
│   ├── trainer.py             # Training loop
│   ├── losses.py              # Loss functions and regularizers
│   ├── metrics.py             # Metric tracking
│   ├── routing.py             # Routing gate implementation
│   └── visualize.py           # Plotting utilities
└── outputs/                   # Experiment results (gitignored)
```

## Commands

```bash
# Activate environment first
source .venv/bin/activate

# Quick validation run (p=17, ~5-10k epochs to grok)
python projects/least_action_learning/scripts/train.py \
    --config projects/least_action_learning/configs/validate.yaml

# Transformer validation
python projects/least_action_learning/scripts/train.py \
    --config projects/least_action_learning/configs/validate_transformer.yaml

# Full experiment (p=113)
python projects/least_action_learning/scripts/train.py \
    --config projects/least_action_learning/configs/baseline.yaml

# Resume training
python projects/least_action_learning/scripts/train.py \
    --config configs/baseline.yaml --resume

# Extend training by 50k epochs
python projects/least_action_learning/scripts/train.py \
    --config configs/baseline.yaml --resume --extra-epochs 50000
```

## Model Types

| Type | Description | Config `model_type` |
|------|-------------|---------------------|
| BaselineMLP | Standard MLP with GELU activations | `"baseline"` |
| RoutedNetwork | MLP with learned routing through parallel heads | `"routed"` |
| GrokTransformer | Decoder-only transformer matching original paper | `"transformer"` |

### Transformer Architecture
- Input format: `[a, op, b, =]` token sequence (4 tokens)
- Vocabulary: `p + 2` tokens (residues 0..p-1, plus op and equals tokens)
- Causal attention masking
- Predicts result class from last position hidden state
- Pre-norm (LayerNorm before attention/FFN)

## Configuration System

Configs are YAML files loaded by `TrainerConfig`. Key parameters:

```yaml
# Data
p: 17                    # Prime modulus
operation: "add"         # "add" or "multiply"
train_frac: 0.5          # Fraction for training

# Model
model_type: "transformer"  # "baseline", "routed", "transformer"
hidden_dim: 128          # d_model for transformer
n_layers: 2              # Number of layers/blocks
n_heads: 4               # Attention heads (transformer) or routing heads

# Training
epochs: 100000
lr: 0.001
weight_decay: 1.0        # Critical for grokking!
optimizer: "adamw"
grad_clip: 1.0           # Gradient clipping (null to disable)

# Regularization (routed networks only)
routing_regularizer: "entropy"  # null, "entropy", "sparsity", "gini"
lambda_routing: 0.01
lambda_spectral: 0.0
```

## Key Files

### src/data.py
- `ModularArithmeticDataset`: One-hot encoded inputs for MLPs
- `SequenceArithmeticDataset`: Token sequences for transformers
- Both provide `get_train()`, `get_test()`, `get_all()` methods

### src/models.py
- `BaselineMLP`: Standard feedforward network
- `RoutedNetwork`: Network with learned routing gates
- `GrokTransformer`: Decoder-only transformer
- `create_model()`: Factory function for instantiation

### src/trainer.py
- `TrainerConfig`: Dataclass for all hyperparameters
- `Trainer`: Handles training loop, evaluation, checkpointing
- Supports resume via `--resume` flag
- Logs: epoch, train/test accuracy, loss, weight norm, spectral smoothness

### src/losses.py
- `LeastActionLoss`: Combined loss with optional regularizers
- `spectral_smoothness()`: Measures output function smoothness via FFT
- Routing regularizers: entropy, sparsity, gini, consistency

### src/visualize.py
- `plot_training_curves()`: Loss and accuracy over time
- `plot_spectral_analysis()`: FFT analysis of output function
- `plot_routing_heatmap()`: Dominant routing head per input
- `save_all_visualizations()`: Generate all plots to output dir

## Output Format

Training logs show:
```
Epoch  12300 | Train: 100.0% | Test:  47.1% | Loss: 1.23e-04, wnorm=45.2, smooth=0.234
```

- **Train/Test**: Accuracy as percentage
- **Loss**: Cross-entropy in scientific notation
- **wnorm**: Total L2 weight norm (tracks regularization effect)
- **smooth**: Spectral smoothness (logged every 1000 epochs)

## Troubleshooting

### Grokking not occurring
- Increase `weight_decay` (try 1.0-2.0)
- Ensure full-batch training (default)
- Check `train_frac` isn't too high (0.3-0.5 typical)

### Transformer dtype errors on MPS
- Ensure token tensors are `torch.long` (fixed in data.py)
- Visualization functions need `is_transformer=True` flag (auto-detected)

### Resuming from wrong epoch
- Checkpoints now store explicit `epoch` field
- Old checkpoints fall back to `best_epoch` (may be inaccurate)
