"""Curvature metrics: spectral smoothness, Jacobian norms, Hessian traces.

This module provides two categories of curvature metrics:

1. **Input-sensitivity metrics** (compute_jacobian_norm, compute_hessian_trace):
   Measure how the model output changes with respect to input perturbations.
   - Jacobian: ||∇_x f(x)||² - first-order sensitivity
   - Hessian: Tr(∇²_x f(x)) - second-order curvature

2. **Weight-curvature metrics** (compute_gradient_norm, compute_weight_hessian_trace, compute_fisher_trace):
   Measure the loss landscape with respect to model weights.
   - Gradient: ||∇_w L|| - slope of loss surface, drives learning
   - Weight Hessian: Tr(∇²_w L) - curvature of loss surface
   - Fisher: Tr(∇L · ∇Lᵀ) - empirical Fisher information, proxy for Hessian
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Callable


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

    Evaluates the model on all p^2 inputs and computes what fraction
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
        # Create all p^2 inputs
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


def compute_jacobian_norm(
    model: nn.Module,
    inputs: Tensor,
    num_samples: int = 10,
    is_transformer: bool = False,
) -> float:
    """
    Compute mean squared Jacobian norm ||grad_x f(x)||^2 via random projections.

    This measures first-order smoothness: how sensitive the output is to
    input perturbations. Lower values indicate smoother functions.

    Uses random projection for efficiency: instead of computing full Jacobian,
    compute E[||J^T v||^2] for random v, which equals ||J||^2_F / output_dim.

    For transformers, computes Jacobian with respect to embeddings (continuous)
    rather than token IDs (discrete).

    Args:
        model: Model to evaluate
        inputs: Input tensor [batch, input_dim] for MLP, [batch, seq_len] for transformer
        num_samples: Number of random projections for estimation
        is_transformer: If True, compute w.r.t. embeddings

    Returns:
        Mean squared Jacobian Frobenius norm across batch
    """
    model.eval()

    if is_transformer:
        # For transformers, get embeddings and compute gradients w.r.t. them
        with torch.no_grad():
            embeddings = model.get_embeddings(inputs)
        embeddings = embeddings.detach().requires_grad_(True)
        logits = model.forward_from_embeddings(embeddings)
        grad_input = embeddings
    else:
        inputs = inputs.detach().requires_grad_(True)
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        grad_input = inputs

    batch_size, output_dim = logits.shape
    total_jac_norm_sq = 0.0

    for _ in range(num_samples):
        # Random projection vector
        v = torch.randn(batch_size, output_dim, device=logits.device)
        v = v / v.norm(dim=-1, keepdim=True)

        # Compute J^T v via backward pass
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=grad_input,
            grad_outputs=v,
            create_graph=False,
            retain_graph=True,
        )[0]

        # ||J^T v||^2 for each sample in batch (flatten if needed)
        grads_flat = grads.view(batch_size, -1)
        jac_norm_sq = (grads_flat ** 2).sum(dim=-1)  # [batch]
        total_jac_norm_sq += jac_norm_sq.mean().item()

    model.train()
    # Average over samples, scale by output_dim to get Frobenius norm
    return (total_jac_norm_sq / num_samples) * output_dim


def compute_hessian_trace(
    model: nn.Module,
    inputs: Tensor,
    num_hutchinson_samples: int = 5,
    is_transformer: bool = False,
) -> float:
    """
    Compute Hessian trace via Hutchinson's stochastic trace estimator.

    This measures second-order curvature: Tr(grad^2_x f(x)). Higher values
    indicate more curved (less smooth) function landscapes.

    Uses Hutchinson's trick: Tr(H) = E[v^T H v] for random v with E[vv^T] = I.

    For transformers, computes Hessian with respect to embeddings.

    Args:
        model: Model to evaluate
        inputs: Input tensor [batch, input_dim] for MLP, [batch, seq_len] for transformer
        num_hutchinson_samples: Number of random vectors for trace estimation
        is_transformer: If True, compute w.r.t. embeddings

    Returns:
        Mean Hessian trace across batch
    """
    model.eval()

    if is_transformer:
        # For transformers, get embeddings and compute gradients w.r.t. them
        with torch.no_grad():
            embeddings = model.get_embeddings(inputs)
        grad_input = embeddings.detach().requires_grad_(True)
        logits = model.forward_from_embeddings(grad_input)
    else:
        grad_input = inputs.detach().requires_grad_(True)
        outputs = model(grad_input)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

    batch_size = grad_input.shape[0]
    total_trace = 0.0

    for _ in range(num_hutchinson_samples):
        # Random Rademacher vector (+1 or -1)
        v = torch.randint(0, 2, grad_input.shape, device=grad_input.device).float() * 2 - 1

        # First derivative: compute gradient of sum of logits w.r.t. inputs
        # We use the predicted class logit for each sample
        predicted_class = logits.argmax(dim=-1)
        selected_logits = logits.gather(1, predicted_class.unsqueeze(1)).squeeze()

        # Gradient
        grads = torch.autograd.grad(
            outputs=selected_logits.sum(),
            inputs=grad_input,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Second derivative: v^T H v = v^T grad(grad f . v)
        grad_v_product = (grads * v).sum()

        # Compute gradient of (grad . v) w.r.t. inputs
        hvp = torch.autograd.grad(
            outputs=grad_v_product,
            inputs=grad_input,
            create_graph=False,
            retain_graph=True,
        )[0]

        # v^T H v (flatten and sum)
        v_flat = v.view(batch_size, -1)
        hvp_flat = hvp.view(batch_size, -1)
        trace_estimate = (v_flat * hvp_flat).sum(dim=-1)  # [batch]
        total_trace += trace_estimate.mean().item()

    model.train()
    return total_trace / num_hutchinson_samples


def compute_per_layer_jacobian_norms(
    model: nn.Module,
    inputs: Tensor,
    num_samples: int = 5,
    is_transformer: bool = False,
) -> list[float]:
    """
    Compute Jacobian norm at each layer's output.

    For each layer, computes ||d(layer_output)/d(input)||_F via random projections.

    Args:
        model: Model to evaluate
        inputs: Input tensor [batch, input_dim] for MLP, [batch, seq_len] for transformer
        num_samples: Number of random projections for estimation
        is_transformer: If True, compute w.r.t. embeddings

    Returns:
        List of Jacobian norms, one per layer
    """
    model.eval()
    layer_norms = []

    # Get the input to differentiate against
    if is_transformer:
        with torch.no_grad():
            embeddings = model.get_embeddings(inputs)
        grad_input = embeddings.detach().requires_grad_(True)
    else:
        grad_input = inputs.detach().requires_grad_(True)

    # Get intermediate representations
    intermediates = []
    if hasattr(model, 'get_per_layer_representations'):
        with torch.enable_grad():
            # Re-run forward to get representations with gradients
            if is_transformer:
                # Need to forward from embeddings
                intermediates = model.get_per_layer_representations_from_embeddings(grad_input)
            else:
                intermediates = model.get_per_layer_representations(grad_input)
    elif hasattr(model, 'layers'):
        # RoutedNetwork - manually track
        if hasattr(model, 'embed'):
            x = model.embed(grad_input) if not is_transformer else grad_input
        else:
            x = grad_input

        for layer in model.layers:
            x, _ = layer(x)
            intermediates.append(x)
    elif hasattr(model, 'net'):
        # BaselineMLP
        x = grad_input
        for i, module in enumerate(model.net):
            x = module(x)
            if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
                intermediates.append(x)
            elif isinstance(module, torch.nn.Linear) and module == model.net[-1]:
                intermediates.append(x)

    if not intermediates:
        model.train()
        return []

    # Compute Jacobian norm for each intermediate
    for rep in intermediates:
        batch_size = rep.shape[0]
        rep_flat = rep.view(batch_size, -1)
        output_dim = rep_flat.shape[1]

        total_jac_norm_sq = 0.0
        for _ in range(num_samples):
            v = torch.randn_like(rep_flat)
            v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

            try:
                grads = torch.autograd.grad(
                    outputs=rep_flat,
                    inputs=grad_input,
                    grad_outputs=v,
                    create_graph=False,
                    retain_graph=True,
                )[0]

                grads_flat = grads.view(batch_size, -1)
                jac_norm_sq = (grads_flat ** 2).sum(dim=-1)
                total_jac_norm_sq += jac_norm_sq.mean().item()
            except RuntimeError:
                # Gradient not available (e.g., disconnected graph)
                total_jac_norm_sq += 0.0

        layer_norms.append((total_jac_norm_sq / num_samples) * output_dim)

    model.train()
    return layer_norms


def compute_per_layer_hessian_traces(
    model: nn.Module,
    inputs: Tensor,
    num_hutchinson_samples: int = 3,
    is_transformer: bool = False,
) -> list[float]:
    """
    Compute Hessian trace at each layer's output via Hutchinson estimator.

    This is expensive - use sparingly. Computes Tr(d^2(layer_output)/d(input)^2).

    Args:
        model: Model to evaluate
        inputs: Input tensor [batch, input_dim] for MLP, [batch, seq_len] for transformer
        num_hutchinson_samples: Number of random vectors for trace estimation
        is_transformer: If True, compute w.r.t. embeddings

    Returns:
        List of Hessian traces, one per layer
    """
    model.eval()
    layer_traces = []

    # Get the input to differentiate against
    if is_transformer:
        with torch.no_grad():
            embeddings = model.get_embeddings(inputs)
        grad_input = embeddings.detach().requires_grad_(True)
    else:
        grad_input = inputs.detach().requires_grad_(True)

    # Get intermediate representations with gradients
    intermediates = []
    if hasattr(model, 'layers'):
        if hasattr(model, 'embed'):
            x = model.embed(grad_input) if not is_transformer else grad_input
        else:
            x = grad_input

        for layer in model.layers:
            x, _ = layer(x)
            intermediates.append(x)
    elif hasattr(model, 'net'):
        x = grad_input
        for module in model.net:
            x = module(x)
            if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU)):
                intermediates.append(x)
            elif isinstance(module, torch.nn.Linear) and module == model.net[-1]:
                intermediates.append(x)

    if not intermediates:
        model.train()
        return []

    # Compute Hessian trace for each intermediate
    for rep in intermediates:
        batch_size = rep.shape[0]
        rep_flat = rep.view(batch_size, -1)

        # Use mean of representation as scalar objective
        scalar_output = rep_flat.mean(dim=-1).sum()

        total_trace = 0.0
        for _ in range(num_hutchinson_samples):
            v = torch.randint(0, 2, grad_input.shape, device=grad_input.device).float() * 2 - 1

            try:
                # First gradient
                grads = torch.autograd.grad(
                    outputs=scalar_output,
                    inputs=grad_input,
                    create_graph=True,
                    retain_graph=True,
                )[0]

                # v^T H v via second gradient
                grad_v_product = (grads * v).sum()
                hvp = torch.autograd.grad(
                    outputs=grad_v_product,
                    inputs=grad_input,
                    create_graph=False,
                    retain_graph=True,
                )[0]

                v_flat = v.view(batch_size, -1)
                hvp_flat = hvp.view(batch_size, -1)
                trace_estimate = (v_flat * hvp_flat).sum(dim=-1)
                total_trace += trace_estimate.mean().item()
            except RuntimeError:
                total_trace += 0.0

        layer_traces.append(total_trace / num_hutchinson_samples)

    model.train()
    return layer_traces


# ─── Weight-Based Curvature Metrics ──────────────────────────────────────────────


def compute_gradient_norm(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
) -> float:
    """
    Compute ||∇_w L|| - gradient norm w.r.t. model weights.

    This measures the "slope" of the loss surface at the current point in
    weight space. High gradient norm indicates steep loss landscape.

    Args:
        model: The model (must be in train mode or have requires_grad=True on params)
        loss_fn: Loss function taking (logits, targets) -> scalar loss
        inputs: Input batch [batch, ...] (one-hot for MLP, token IDs for transformer)
        targets: Target batch [batch]

    Returns:
        L2 norm of gradient across all parameters
    """
    # Ensure model is ready for gradient computation
    model.zero_grad()

    # Forward pass
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        logits = outputs[0]  # Handle routed networks returning (logits, metrics)
    else:
        logits = outputs

    # Compute loss and backprop
    loss = loss_fn(logits, targets)
    loss.backward()

    # Compute total gradient norm
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2

    return total_norm_sq ** 0.5


def compute_weight_hessian_trace(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    num_hutchinson_samples: int = 10,
) -> float:
    """
    Compute Tr(∇²_w L) - trace of Hessian w.r.t. weights via Hutchinson estimator.

    This measures the curvature of the loss surface. During grokking, the Hessian
    spectrum is hypothesized to flatten (trace decreases), indicating the model
    finds flatter minima with better generalization.

    Uses Hutchinson's trick: Tr(H) = E[v^T H v] for random v ~ N(0, I).

    Args:
        model: The model
        loss_fn: Loss function taking (logits, targets) -> scalar loss
        inputs: Input batch
        targets: Target batch
        num_hutchinson_samples: Number of random vectors for estimation (more = accurate but slower)

    Returns:
        Estimated trace of Hessian w.r.t. weights
    """
    trace_sum = 0.0

    for _ in range(num_hutchinson_samples):
        # Forward pass with gradient tracking
        model.zero_grad()
        outputs = model(inputs)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        loss = loss_fn(logits, targets)

        # Get gradients with computation graph retained
        # allow_unused=True handles routed networks where some heads may not be used
        params = [p for p in model.parameters() if p.requires_grad]
        grads = torch.autograd.grad(
            loss, params, create_graph=True, retain_graph=True, allow_unused=True
        )
        # Replace None gradients (unused params) with zeros
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

        # Random vector for Hutchinson (same shape as concatenated gradients)
        v = [torch.randn_like(g) for g in grads]

        # Compute v^T H v = v^T (∂/∂w)(∇_w L)
        # First: ∇_w L · v (scalar)
        grad_v = sum((g * vi).sum() for g, vi in zip(grads, v))

        # Second: differentiate again to get Hv
        hvp = torch.autograd.grad(grad_v, params, retain_graph=True, allow_unused=True)
        # Replace None with zeros
        hvp = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp, params)]

        # v^T H v = sum(v * Hv)
        trace_estimate = sum((vi * hvi).sum().item() for vi, hvi in zip(v, hvp))
        trace_sum += trace_estimate

    return trace_sum / num_hutchinson_samples


def compute_fisher_trace(
    model: nn.Module,
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    inputs: Tensor,
    targets: Tensor,
    max_samples: int = 32,
) -> float:
    """
    Compute Tr(F) where F = E[∇L ∇L^T] is the empirical Fisher information matrix.

    The Fisher information measures the curvature of the log-likelihood and serves
    as an approximation to the Hessian. For a batch:

        Tr(F) ≈ (1/N) Σᵢ ||∇_w L(xᵢ, yᵢ)||²

    This is computationally cheaper than the true Hessian trace.

    Args:
        model: The model
        loss_fn: Loss function (typically cross-entropy, should work per-sample)
        inputs: Input batch
        targets: Target batch
        max_samples: Maximum number of samples to use (for efficiency)

    Returns:
        Trace of empirical Fisher information matrix
    """
    batch_size = min(inputs.shape[0], max_samples)
    fisher_trace = 0.0

    # Compute per-sample gradient norms (diagonal Fisher approximation)
    for i in range(batch_size):
        model.zero_grad()

        # Single-sample forward
        sample_input = inputs[i : i + 1]
        sample_target = targets[i : i + 1]

        outputs = model(sample_input)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        loss = loss_fn(logits, sample_target)
        loss.backward()

        # ||∇_w L_i||²
        grad_norm_sq = sum(
            p.grad.data.norm(2).item() ** 2
            for p in model.parameters()
            if p.grad is not None
        )
        fisher_trace += grad_norm_sq

    return fisher_trace / batch_size
