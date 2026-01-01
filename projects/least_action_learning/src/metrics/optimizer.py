"""Adam optimizer dynamics metrics.

These metrics analyze the internal state of the Adam optimizer to understand
training dynamics during grokking:
- Effective Learning Rate: Tracks the adaptive learning rate scaling per parameter
- Adam Ratio: Signal-to-noise ratio of gradient vs. variance estimate
- Update vs. Decay: Balance between gradient update and weight regularization
"""

import torch
from torch.optim import Optimizer
from typing import Optional
from dataclasses import dataclass


@dataclass
class AdamMetrics:
    """Container for Adam optimizer dynamics metrics."""

    effective_lr_mean: float  # Mean of sqrt(v_t) across all params
    effective_lr_max: float  # Max of sqrt(v_t) (hottest learning)
    adam_ratio_mean: float  # Mean of |m_t| / (sqrt(v_t) + eps)
    adam_ratio_max: float  # Max ratio (most confident direction)
    update_decay_ratio: float  # ||grad update|| / ||weight decay||


def compute_adam_metrics(
    optimizer: Optimizer,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> Optional[AdamMetrics]:
    """
    Extract metrics from Adam/AdamW optimizer state.

    Analyzes the internal state of Adam to understand training dynamics:

    1. **Effective Learning Rate** (sqrt(v_t)):
       The term 1/(sqrt(v_t) + eps) acts as a per-parameter LR multiplier.
       We track sqrt(v_t) - larger values mean SMALLER effective learning rate.

    2. **Adam Ratio** (|m_t| / (sqrt(v_t) + eps)):
       This is the core of Adam's update step. High ratio = confident direction.
       Acts as signal-to-noise ratio for gradient estimates.

    3. **Update vs. Decay Ratio**:
       In AdamW: delta_w = lr * m/(sqrt(v)+eps) + lr * lambda * w
       We compute ||gradient_update|| / ||weight_decay_update||
       High ratio = learning dominates. Low ratio = forgetting dominates.

    Args:
        optimizer: Adam or AdamW optimizer with populated state
        weight_decay: Weight decay coefficient (lambda) for decay ratio
        eps: Adam epsilon for numerical stability

    Returns:
        AdamMetrics if optimizer has state, None if state not yet populated
    """
    if not hasattr(optimizer, "state") or len(optimizer.state) == 0:
        return None

    all_sqrt_v = []
    all_ratios = []
    grad_update_norm_sq = 0.0
    decay_norm_sq = 0.0

    for param_group in optimizer.param_groups:
        group_wd = param_group.get("weight_decay", weight_decay)

        for p in param_group["params"]:
            # Look up state by parameter (state persists even after zero_grad)
            state = optimizer.state.get(p, {})
            if "exp_avg" not in state or "exp_avg_sq" not in state:
                continue

            m_t = state["exp_avg"]  # First moment (gradient EMA)
            v_t = state["exp_avg_sq"]  # Second moment (squared gradient EMA)

            # sqrt(v_t + eps) - effective per-parameter LR scaling denominator
            sqrt_v = torch.sqrt(v_t + eps)
            all_sqrt_v.append(sqrt_v.flatten())

            # |m_t| / sqrt(v_t + eps) - signal-to-noise ratio
            ratio = torch.abs(m_t) / sqrt_v
            all_ratios.append(ratio.flatten())

            # Gradient update term: m_t / sqrt(v_t + eps)
            # This is what Adam actually uses for updates (before LR scaling)
            update = m_t / sqrt_v
            grad_update_norm_sq += (update**2).sum().item()

            # Weight decay term: weight_decay * p
            if group_wd > 0:
                decay_norm_sq += (group_wd * p.data**2).sum().item()

    if not all_sqrt_v:
        return None

    # Concatenate all statistics
    sqrt_v_all = torch.cat(all_sqrt_v)
    ratios_all = torch.cat(all_ratios)

    # Compute aggregate metrics
    effective_lr_mean = sqrt_v_all.mean().item()
    effective_lr_max = sqrt_v_all.max().item()
    adam_ratio_mean = ratios_all.mean().item()
    adam_ratio_max = ratios_all.max().item()

    # Update vs decay ratio: ||gradient update|| / ||weight decay||
    if decay_norm_sq > 1e-12:
        update_decay_ratio = (grad_update_norm_sq**0.5) / (decay_norm_sq**0.5)
    else:
        update_decay_ratio = float("inf")

    return AdamMetrics(
        effective_lr_mean=effective_lr_mean,
        effective_lr_max=effective_lr_max,
        adam_ratio_mean=adam_ratio_mean,
        adam_ratio_max=adam_ratio_max,
        update_decay_ratio=update_decay_ratio,
    )
