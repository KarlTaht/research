"""Loss functions and regularizers for least action learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, Optional


def entropy_regularizer(routing_weights: list[Tensor], eps: float = 1e-8) -> Tensor:
    """
    Entropy regularizer to encourage decisive routing.

    Lower entropy = routing concentrates on fewer heads.
    Minimizing this encourages the network to make clear routing choices.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer
        eps: Small constant for numerical stability

    Returns:
        Mean entropy across all layers and samples (scalar tensor)
    """
    total_entropy = 0.0

    for weights in routing_weights:
        # Clamp for numerical stability
        weights = weights.clamp(min=eps, max=1 - eps)
        # Entropy: -sum(w * log(w))
        entropy = -(weights * weights.log()).sum(dim=-1)
        total_entropy = total_entropy + entropy.mean()

    return total_entropy / len(routing_weights)


def sparsity_regularizer(routing_weights: list[Tensor]) -> Tensor:
    """
    L1 sparsity regularizer to encourage few active heads.

    Since weights sum to 1 (softmax), this effectively encourages
    concentration on a single head (minimum L1 = 1 when one-hot).

    Actually for softmax outputs summing to 1, L1 norm is always 1.
    So we use a different formulation: penalize deviation from one-hot.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer

    Returns:
        Mean sparsity penalty across all layers
    """
    total_penalty = 0.0

    for weights in routing_weights:
        # Encourage one-hot: maximize the max weight
        # Penalty = 1 - max(weights) → 0 when one-hot, higher when spread
        max_weight = weights.max(dim=-1).values
        penalty = 1.0 - max_weight
        total_penalty = total_penalty + penalty.mean()

    return total_penalty / len(routing_weights)


def gini_regularizer(routing_weights: list[Tensor]) -> Tensor:
    """
    Gini coefficient regularizer for sparsity.

    Gini = 0 means uniform, Gini = 1 means maximally unequal.
    We return 1 - Gini so that minimizing this encourages sparsity.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer

    Returns:
        Mean (1 - Gini) across all layers (lower = sparser)
    """
    total = 0.0

    for weights in routing_weights:
        batch_size, n_heads = weights.shape

        # Sort weights
        sorted_weights, _ = weights.sort(dim=-1)

        # Gini coefficient calculation
        # G = (2 * sum(i * w_i) / (n * sum(w_i))) - (n + 1) / n
        indices = torch.arange(1, n_heads + 1, device=weights.device).float()
        numerator = (indices * sorted_weights).sum(dim=-1)
        denominator = n_heads * sorted_weights.sum(dim=-1)

        gini = (2 * numerator / (denominator + 1e-8)) - (n_heads + 1) / n_heads
        total = total + (1.0 - gini).mean()

    return total / len(routing_weights)


def consistency_regularizer(
    routing_weights: list[Tensor],
    inputs: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """
    Encourage similar inputs to have similar routing.

    Computes pairwise input similarity and penalizes routing
    differences for similar inputs.

    Args:
        routing_weights: List of [batch, n_heads] tensors per layer
        inputs: Input tensor [batch, input_dim]
        temperature: Temperature for similarity computation

    Returns:
        Consistency penalty (lower = more consistent routing for similar inputs)
    """
    # Compute input similarity matrix
    inputs_norm = F.normalize(inputs, dim=-1)
    similarity = torch.mm(inputs_norm, inputs_norm.t()) / temperature
    similarity = F.softmax(similarity, dim=-1)

    total_penalty = 0.0

    for weights in routing_weights:
        # For each sample, expected routing of similar samples
        expected_routing = torch.mm(similarity, weights)

        # KL divergence between actual and expected routing
        kl = F.kl_div(
            weights.log().clamp(min=-100),
            expected_routing,
            reduction='batchmean',
        )
        total_penalty = total_penalty + kl

    return total_penalty / len(routing_weights)


def make_low_freq_mask(p: int, K: int, device: Optional[torch.device] = None) -> Tensor:
    """
    Create a mask for low-frequency components in 2D FFT.

    Args:
        p: Grid size (p x p FFT)
        K: Frequency cutoff (include frequencies with |k| <= K)
        device: Device to create tensor on

    Returns:
        Boolean mask [p, p] where True = low frequency
    """
    # Create frequency grid
    freqs = torch.fft.fftfreq(p, device=device)

    # 2D frequency magnitudes (with wraparound for FFT)
    fx, fy = torch.meshgrid(freqs, freqs, indexing='ij')

    # Distance from origin (in frequency space)
    freq_magnitude = torch.sqrt(fx**2 + fy**2)

    # Cutoff in normalized frequency
    cutoff = K / p

    return freq_magnitude <= cutoff


def spectral_smoothness(
    model: nn.Module,
    p: int,
    K: int,
    device: torch.device,
    is_transformer: bool = False,
) -> float:
    """
    Compute spectral smoothness of the model's output function.

    Evaluates the model on all p² inputs and computes what fraction
    of the output's spectral energy is in low frequencies.

    Args:
        model: Model to evaluate (should return logits as first output)
        p: Prime modulus (determines input space size)
        K: Frequency cutoff for "low frequency"
        device: Device to run on
        is_transformer: If True, create sequence inputs [a, op, b, =] instead of one-hot

    Returns:
        Smoothness score in [0, 1], higher = smoother
    """
    model.eval()

    with torch.no_grad():
        # Create all p² inputs
        a_vals = torch.arange(p, device=device)
        b_vals = torch.arange(p, device=device)
        aa, bb = torch.meshgrid(a_vals, b_vals, indexing='ij')
        pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)
        batch_size = p * p

        if is_transformer:
            # Create sequence inputs: [a, op, b, =]
            op_token = p      # operation token
            eq_token = p + 1  # equals token
            inputs = torch.stack([
                pairs[:, 0].long(),
                torch.full((batch_size,), op_token, dtype=torch.long, device=device),
                pairs[:, 1].long(),
                torch.full((batch_size,), eq_token, dtype=torch.long, device=device),
            ], dim=1)
        else:
            # One-hot encode for MLP
            inputs = torch.zeros(batch_size, 2 * p, device=device)
            inputs[torch.arange(batch_size), pairs[:, 0]] = 1.0
            inputs[torch.arange(batch_size), p + pairs[:, 1]] = 1.0

        # Get model predictions
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Get predicted class (or use softmax for soft version)
        predicted = logits.argmax(dim=-1).float()

        # Reshape to p x p grid
        predicted = predicted.view(p, p)

        # Compute 2D FFT
        spectrum = torch.fft.fft2(predicted)
        power = (spectrum.abs() ** 2)

        # Create low-frequency mask
        mask = make_low_freq_mask(p, K, device=device)

        # Compute ratio
        low_energy = (power * mask).sum()
        total_energy = power.sum()

        smoothness = (low_energy / (total_energy + 1e-8)).item()

    model.train()
    return smoothness


def spectral_smoothness_loss(
    model: nn.Module,
    p: int,
    K: int,
    device: torch.device,
) -> Tensor:
    """
    Differentiable spectral smoothness loss using soft predictions.

    Unlike spectral_smoothness(), this uses softmax probabilities
    instead of hard predictions, allowing gradient flow.

    Note: This is expensive - computes forward pass on all p² inputs.
    Should be computed periodically, not every step.
    """
    # Create all inputs (same as above)
    a_vals = torch.arange(p, device=device)
    b_vals = torch.arange(p, device=device)
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing='ij')
    pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)

    batch_size = p * p
    inputs = torch.zeros(batch_size, 2 * p, device=device)
    inputs[torch.arange(batch_size), pairs[:, 0]] = 1.0
    inputs[torch.arange(batch_size), p + pairs[:, 1]] = 1.0

    # Forward pass (with gradients)
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    # Use expected prediction value (soft)
    probs = F.softmax(logits, dim=-1)
    class_indices = torch.arange(p, device=device).float()
    expected_class = (probs * class_indices).sum(dim=-1)

    # Reshape and compute FFT
    expected_class = expected_class.view(p, p)
    spectrum = torch.fft.fft2(expected_class)
    power = spectrum.abs() ** 2

    # Low-frequency energy ratio
    mask = make_low_freq_mask(p, K, device=device).float()
    low_energy = (power * mask).sum()
    total_energy = power.sum()

    smoothness = low_energy / (total_energy + 1e-8)

    # Return 1 - smoothness as loss (minimize to increase smoothness)
    return 1.0 - smoothness


class LeastActionLoss(nn.Module):
    """
    Combined loss for least action learning.

    L = L_task + λ_routing * L_routing + λ_spectral * L_spectral

    Args:
        routing_regularizer: One of "entropy", "sparsity", "gini", "consistency", or None
        lambda_routing: Weight for routing regularizer
        lambda_spectral: Weight for spectral smoothness (0 = disabled)
        spectral_k: Frequency cutoff for spectral smoothness
        spectral_interval: Compute spectral loss every N steps (expensive)
    """

    def __init__(
        self,
        routing_regularizer: Optional[str] = "entropy",
        lambda_routing: float = 0.01,
        lambda_spectral: float = 0.0,
        spectral_k: Optional[int] = None,
        spectral_interval: int = 100,
    ):
        super().__init__()
        self.routing_regularizer = routing_regularizer
        self.lambda_routing = lambda_routing
        self.lambda_spectral = lambda_spectral
        self.spectral_k = spectral_k
        self.spectral_interval = spectral_interval

        # Select regularizer function
        self.reg_fn: Optional[Callable] = None
        if routing_regularizer == "entropy":
            self.reg_fn = entropy_regularizer
        elif routing_regularizer == "sparsity":
            self.reg_fn = sparsity_regularizer
        elif routing_regularizer == "gini":
            self.reg_fn = gini_regularizer
        elif routing_regularizer is not None:
            raise ValueError(f"Unknown regularizer: {routing_regularizer}")

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        routing_weights: Optional[list[Tensor]] = None,
        inputs: Optional[Tensor] = None,
        model: Optional[nn.Module] = None,
        p: Optional[int] = None,
        step: int = 0,
    ) -> tuple[Tensor, dict]:
        """
        Compute total loss.

        Args:
            logits: Model output logits
            targets: Target class indices
            routing_weights: List of routing weight tensors (for routing reg)
            inputs: Input tensor (for consistency reg)
            model: Full model (for spectral reg)
            p: Prime modulus (for spectral reg)
            step: Current training step (for spectral interval)

        Returns:
            total_loss: Combined loss tensor
            loss_dict: Dictionary of individual loss components
        """
        # Task loss
        task_loss = F.cross_entropy(logits, targets)
        loss_dict = {"task_loss": task_loss.item()}

        total_loss = task_loss

        # Routing regularizer
        if self.reg_fn is not None and routing_weights is not None and self.lambda_routing > 0:
            if self.routing_regularizer == "consistency" and inputs is not None:
                routing_loss = consistency_regularizer(routing_weights, inputs)
            else:
                routing_loss = self.reg_fn(routing_weights)

            total_loss = total_loss + self.lambda_routing * routing_loss
            loss_dict["routing_loss"] = routing_loss.item()

        # Spectral smoothness (periodic)
        if (
            self.lambda_spectral > 0
            and model is not None
            and p is not None
            and step % self.spectral_interval == 0
        ):
            k = self.spectral_k if self.spectral_k is not None else p // 4
            spectral_loss = spectral_smoothness_loss(
                model, p, k, logits.device
            )
            total_loss = total_loss + self.lambda_spectral * spectral_loss
            loss_dict["spectral_loss"] = spectral_loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict
