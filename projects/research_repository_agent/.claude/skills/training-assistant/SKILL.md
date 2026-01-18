---
name: training-assistant
description: Help with ML training tasks. Use when user wants to run training, debug errors, estimate resources, or resume from checkpoints.
---

# Training Assistant

Help users with ML training tasks, from running experiments to debugging errors.

## When to Use

- "How do I train X?"
- "Run training for the transformer"
- "Resume from checkpoint"
- "I got this error: ..."
- "How much GPU memory do I need?"

## Available Tools

| Tool | Purpose |
|------|---------|
| `run_command` | Generate and execute commands |
| `explain_error` | Context-aware error explanation |
| `suggest_cleanup` | Identify organizational issues |

## Common Tasks

### 1. Generate Training Commands

```
User: "Train the custom transformer on tinystories"

1. Find the project and config
2. Generate command: python train.py --config configs/tinystories.yaml
3. Show the command before executing
4. Require confirmation for long-running tasks
```

### 2. Resume Training

```
User: "Resume training from checkpoint"

1. Find the latest checkpoint with find_checkpoint()
2. Generate command with --resume flag
3. Show checkpoint path and metrics
```

### 3. Explain Errors

```
User: "I got CUDA out of memory"

1. Identify error type
2. Suggest solutions (reduce batch size, gradient checkpointing)
3. Show modified config or command
```

## Error Patterns

See [ERROR_PATTERNS.md](ERROR_PATTERNS.md) for common errors and fixes.

## Safety Rules

- **Always show commands before executing**
- **Require confirmation for training runs**
- **Never execute dangerous commands** (rm -rf, sudo, etc.)
- **Check GPU availability before suggesting training**

## Resource Estimation

Rough GPU memory estimates for transformers:
- Parameters × 4 bytes (fp32) or × 2 bytes (fp16)
- Activations: batch_size × seq_len × d_model × n_layers × 4 bytes
- Optimizer states: 2× model size (Adam)

Example: 10M param model, batch_size=32, seq_len=512
- Model: ~40MB (fp32)
- Activations: ~500MB (rough)
- Optimizer: ~80MB
- Total: ~1GB minimum

## Commands Reference

```bash
# Training
python train.py --config configs/default.yaml
python train.py --config configs/default.yaml --resume

# Evaluation
python evaluate.py --checkpoint path/to/checkpoint.pt

# GPU Check
nvidia-smi

# Environment
python -c "import torch; print(torch.cuda.is_available())"
```
