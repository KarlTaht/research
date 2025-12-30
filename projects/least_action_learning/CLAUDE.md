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
Given inputs `(a, b)` where `a, b ∈ {0, ..., p-1}` for prime `p`, predict `(a + b) mod p` where `+` is addition or multiplication.

## Project Structure

```
projects/least_action_learning/
├── configs/                    # YAML experiment configurations
│   ├── baseline.yaml          # MLP baseline (p=113)
│   ├── validate.yaml          # Quick validation (p=17, MLP)
│   ├── transformer.yaml       # Transformer (p=113)
│   ├── validate_transformer.yaml  # Quick transformer validation (p=17)
│   └── transformer_sweep.yaml # Hyperparameter sweep (lr x wd x p)
├── scripts/
│   ├── train.py               # Main training entry point
│   └── run_sweep.py           # Run multiple experiments from sweep config
├── src/
│   ├── __init__.py            # Package exports
│   ├── data.py                # Dataset classes
│   ├── models.py              # Model architectures
│   ├── trainer.py             # Training loop
│   ├── losses.py              # Loss functions, regularizers, curvature metrics
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

# Transformer validation (p=17, 20k epochs)
python projects/least_action_learning/scripts/train.py \
    --config projects/least_action_learning/configs/validate_transformer.yaml

# Full experiment (p=113)
python projects/least_action_learning/scripts/train.py \
    --config projects/least_action_learning/configs/baseline.yaml

# Run hyperparameter sweep (18 experiments)
python projects/least_action_learning/scripts/run_sweep.py \
    --config projects/least_action_learning/configs/transformer_sweep.yaml

# Sweep with dry-run (show commands without running)
python scripts/run_sweep.py --config configs/transformer_sweep.yaml --dry-run

# Run single experiment from sweep
python scripts/run_sweep.py --config configs/transformer_sweep.yaml \
    --experiment p17_lr1e-3_wd1.0

# Resume training
python scripts/train.py --config configs/baseline.yaml --resume

# Extend training by 50k epochs
python scripts/train.py --config configs/baseline.yaml --resume --extra-epochs 50000
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
weight_decay: 1.5        # Critical for grokking (1.0-2.0 typical)
optimizer: "adamw"
beta1: 0.9               # AdamW first moment decay
beta2: 0.98              # AdamW second moment (0.98 more stable than 0.999)
eps: 1.0e-8              # AdamW epsilon (use 1.0e-8 not 1e-8 in YAML)
grad_clip: 1.0           # Gradient clipping (null to disable)
warmup_epochs: 500       # Linear LR warmup from 0 to lr

# Regularization (routed networks only)
routing_regularizer: "entropy"  # null, "entropy", "sparsity", "gini"
lambda_routing: 0.01
lambda_spectral: 0.0
```

### YAML Scientific Notation
Use `1.0e-8` instead of `1e-8` in YAML files. YAML 1.1 requires a decimal point for scientific notation to be parsed as float.

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
- All models implement `get_representation()` for pre-output hidden states
- Transformers also have `forward_from_embeddings()` and `get_embeddings()` for gradient analysis

### src/trainer.py
- `TrainerConfig`: Dataclass for all hyperparameters
- `Trainer`: Handles training loop, evaluation, checkpointing
- **Weight decay groups**: Excludes embeddings, biases, LayerNorm from weight decay
- **LR warmup**: Linear warmup from 0 to `lr` over `warmup_epochs`
- Supports resume via `--resume` flag

### src/losses.py
- `LeastActionLoss`: Combined loss with optional regularizers
- `spectral_smoothness()`: Measures output function smoothness via FFT
- `compute_jacobian_norm()`: Input sensitivity via random projections
- `compute_hessian_trace()`: Curvature via Hutchinson estimator
- Routing regularizers: entropy, sparsity, gini, consistency

### src/metrics.py
- `TrainingMetrics`: Dataclass for all tracked metrics
- `MetricsHistory`: Stores metrics over training, saves to Parquet
- `compute_representation_norm()`: Hidden state magnitude before unembedding

### src/visualize.py
- `plot_training_curves()`: Loss and accuracy over time
- `plot_spectral_analysis()`: FFT analysis of output function
- `plot_routing_heatmap()`: Dominant routing head per input
- `save_all_visualizations()`: Generate all plots to output dir

## Output Format

Training logs show (every `log_every` epochs):
```
Epoch   12300 | Train: 100.0% | Test:  47.1% | Loss: 1.23e-04, rnorm=45.2, smooth=0.234, jac=1.50e+03, |hess|=2.30e+02
```

- **Train/Test**: Accuracy as percentage
- **Loss**: Cross-entropy in scientific notation
- **rnorm**: Representation norm (hidden state before unembedding)
- **smooth**: Spectral smoothness (lower = smoother output function)
- **jac**: Jacobian norm (input sensitivity)
- **|hess|**: Absolute Hessian trace (curvature)

All metrics are saved to `history.parquet` for later analysis.

## Hyperparameter Sweep

The sweep config (`transformer_sweep.yaml`) runs 18 experiments:
- Problem sizes: p=17 (faster grokking) and p=113 (slower)
- Learning rates: 3e-4, 1e-3, 3e-3
- Weight decay: 0.5, 1.0, 2.0

```bash
# Run full sweep (sequential)
python scripts/run_sweep.py --config configs/transformer_sweep.yaml

# Resume from experiment index 5
python scripts/run_sweep.py --config configs/transformer_sweep.yaml --start-from 5
```

## Troubleshooting

### Grokking not occurring
- Increase `weight_decay` (try 1.0-2.0)
- Ensure full-batch training (default)
- Check `train_frac` isn't too high (0.3-0.5 typical)
- Use `warmup_epochs` to stabilize early training

### Training instability (accuracy blips)
- Add `warmup_epochs: 500` for linear LR warmup
- Use `beta2: 0.98` instead of 0.999 (shorter optimizer memory)
- Ensure weight decay excludes embeddings/biases (handled automatically)

### YAML parsing errors
- Use `1.0e-8` not `1e-8` for scientific notation
- TrainerConfig has `__post_init__` safeguard for string-to-float conversion

### Transformer dtype errors on MPS
- Ensure token tensors are `torch.long` (fixed in data.py)
- Visualization functions need `is_transformer=True` flag (auto-detected)

### Resuming from wrong epoch
- Checkpoints now store explicit `epoch` field
- Old checkpoints fall back to `best_epoch` (may be inaccurate)
