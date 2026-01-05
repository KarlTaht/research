"""Fourier/frequency analysis for mechanistic interpretability of grokking.

Implements analyses from Nanda et al. "Progress measures for grokking via
mechanistic interpretability" (ICLR 2023).

For modular arithmetic (a + b) mod p, the model learns to use a sparse set
of Fourier frequencies (cos/sin at omega_k = 2*pi*k/p) and trigonometric
identities to compute the result.

Key analyses:
- Embedding spectrum: DFT of token embeddings to see which frequencies are encoded
- Logit spectrum: 2D DFT over all (a,b) inputs to identify key cos(w*(a+b)) components
- Ablations: Restricted (keep key freqs) and excluded (remove key freqs) loss

Example usage:
    from src.analysis import ExperimentLoader
    from src.analysis.frequency import analyze_checkpoint_frequency

    loader = ExperimentLoader()
    exp = loader.load('p17_lr3e-4_wd1.0')

    results = analyze_checkpoint_frequency(exp)
    print(f"Key frequencies: {results['logit_spectrum'].key_frequencies}")
    print(f"Total FVE: {results['logit_spectrum'].fve_total:.1%}")
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from .loader import ExperimentData, ExperimentLoader


@dataclass
class EmbeddingSpectrum:
    """Fourier spectrum of token embeddings.

    Attributes:
        freq_power: [p] total power at each frequency k (sum over d_model)
        key_frequencies: List of dominant frequency indices (sorted by power)
        cos_coeffs: [p, d_model] cosine Fourier coefficients
        sin_coeffs: [p, d_model] sine Fourier coefficients
        p: Prime modulus
    """

    freq_power: NDArray[np.float64]
    key_frequencies: list[int]
    cos_coeffs: NDArray[np.float64]
    sin_coeffs: NDArray[np.float64]
    p: int


@dataclass
class LogitSpectrum:
    """2D Fourier spectrum of model logits over all (a, b) input pairs.

    Attributes:
        logit_grid: [p, p, p] logits for all inputs (a, b) -> logits
        power_2d: [p, p] 2D Fourier power spectrum (summed over output classes)
        diag_power: [p] power along the (a+b) diagonal frequencies
        key_frequencies: Frequency indices with significant (a+b) component
        fve_per_freq: [p] fraction of variance explained by each frequency
        fve_total: Total FVE by all key frequencies combined
        p: Prime modulus
    """

    logit_grid: NDArray[np.float64]
    power_2d: NDArray[np.float64]
    diag_power: NDArray[np.float64]
    key_frequencies: list[int]
    fve_per_freq: NDArray[np.float64]
    fve_total: float
    p: int


@dataclass
class AblationResults:
    """Results from frequency ablation experiments.

    Restricted: Keep only key frequency components in logits.
    Excluded: Remove key frequency components from logits.

    If the model has learned the Fourier algorithm:
    - restricted_loss should be low (the key frequencies carry the signal)
    - excluded_loss should be high (removing them destroys performance)

    Attributes:
        full_loss: Cross-entropy loss with original logits
        full_accuracy: Accuracy with original logits
        restricted_loss: Loss keeping only key frequencies
        restricted_accuracy: Accuracy keeping only key frequencies
        excluded_loss: Loss removing key frequencies
        excluded_accuracy: Accuracy removing key frequencies
        key_frequencies: Frequencies used in ablation
    """

    full_loss: float
    full_accuracy: float
    restricted_loss: float
    restricted_accuracy: float
    excluded_loss: float
    excluded_accuracy: float
    key_frequencies: list[int]


def compute_fourier_basis(p: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute orthonormal Fourier basis for Z/pZ.

    Args:
        p: Prime modulus

    Returns:
        cos_basis: [p, p] where cos_basis[k, n] = cos(2*pi*k*n/p) / sqrt(p/2)
        sin_basis: [p, p] where sin_basis[k, n] = sin(2*pi*k*n/p) / sqrt(p/2)

    Note:
        - k=0 row of sin_basis is all zeros
        - For k > p/2, frequencies are conjugate pairs (can be ignored)
        - Normalization makes basis orthonormal for k in [1, (p-1)/2]
    """
    n = np.arange(p)
    k = np.arange(p)
    omega = 2 * np.pi / p

    # [p, p] matrices of cos(2*pi*k*n/p) and sin(2*pi*k*n/p)
    angles = np.outer(k, n) * omega  # [k, n]

    cos_basis = np.cos(angles)
    sin_basis = np.sin(angles)

    # Normalize for orthonormality (except DC component k=0)
    # sqrt(2/p) normalization makes <cos_k, cos_k> = 1 for k > 0
    norm = np.sqrt(2 / p)
    cos_basis[1:] *= norm
    sin_basis[1:] *= norm

    # DC component has different normalization
    cos_basis[0] *= 1 / np.sqrt(p)

    return cos_basis, sin_basis


def compute_embedding_spectrum(
    model: nn.Module,
    p: int,
    threshold: float = 0.15,
) -> EmbeddingSpectrum:
    """Compute Fourier spectrum of token embeddings.

    Applies DFT along the token dimension of the embedding matrix to see
    which frequencies the model encodes in its input representations.

    Args:
        model: GrokTransformer with token_embedding attribute
        p: Prime modulus (uses first p token embeddings)
        threshold: Fraction of max power to consider "key" (default 0.15)

    Returns:
        EmbeddingSpectrum with power and coefficient data

    Raises:
        ValueError: If model doesn't have token_embedding attribute
    """
    if not hasattr(model, "token_embedding"):
        raise ValueError("Model must have token_embedding attribute (GrokTransformer)")

    # Extract embedding weights for residue tokens [0, p-1]
    # Shape: [p, d_model]
    with torch.no_grad():
        embeddings = model.token_embedding.weight[:p].detach().cpu().numpy().astype(np.float64)

    # Compute Fourier basis
    cos_basis, sin_basis = compute_fourier_basis(p)

    # Project embeddings onto Fourier basis
    # cos_coeffs[k, d] = sum_n embedding[n, d] * cos_basis[k, n]
    cos_coeffs = cos_basis @ embeddings  # [p, d_model]
    sin_coeffs = sin_basis @ embeddings  # [p, d_model]

    # Compute power at each frequency (sum over d_model dimensions)
    # Power = cos_coeff^2 + sin_coeff^2 for each (k, d), then sum over d
    power_per_dim = cos_coeffs**2 + sin_coeffs**2  # [p, d_model]
    freq_power = power_per_dim.sum(axis=1)  # [p]

    # Identify key frequencies
    key_frequencies = identify_key_frequencies(freq_power, threshold=threshold)

    return EmbeddingSpectrum(
        freq_power=freq_power,
        key_frequencies=key_frequencies,
        cos_coeffs=cos_coeffs,
        sin_coeffs=sin_coeffs,
        p=p,
    )


def compute_logit_spectrum(
    model: nn.Module,
    p: int,
    operation: str = "add",
    device: Optional[torch.device] = None,
    batch_size: int = 512,
) -> LogitSpectrum:
    """Compute 2D Fourier spectrum of model logits over all inputs.

    Generates all p^2 input pairs (a, b), computes model logits for each,
    then applies 2D FFT to identify which Fourier components the model uses.

    The key insight is that for modular addition, the correct output depends
    on (a + b) mod p. In Fourier space, this manifests as power concentrated
    on the diagonal frequencies cos(w_k * (a+b)) and sin(w_k * (a+b)).

    Args:
        model: GrokTransformer in eval mode
        p: Prime modulus
        operation: "add" (default) or "multiply"
        device: Compute device (default: auto-detect)
        batch_size: Batch size for forward passes

    Returns:
        LogitSpectrum with 2D power spectrum and key frequency data
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Generate all p^2 input pairs
    # For transformer: input format is [a, op, b, =] as tokens
    a_vals = torch.arange(p, device=device)
    b_vals = torch.arange(p, device=device)

    # Create meshgrid of all (a, b) pairs
    aa, bb = torch.meshgrid(a_vals, b_vals, indexing="ij")
    aa_flat = aa.flatten()  # [p^2]
    bb_flat = bb.flatten()  # [p^2]

    # Token IDs for transformer input
    # Assuming vocab: 0..p-1 = residues, p = op, p+1 = equals
    op_token = p  # Addition/multiplication operator
    eq_token = p + 1  # Equals sign

    # Build input sequences: [a, op, b, =]
    # Shape: [p^2, 4]
    inputs = torch.stack(
        [
            aa_flat,
            torch.full_like(aa_flat, op_token),
            bb_flat,
            torch.full_like(aa_flat, eq_token),
        ],
        dim=1,
    )

    # Compute logits in batches
    logits_list = []
    with torch.no_grad():
        for i in range(0, p * p, batch_size):
            batch = inputs[i : i + batch_size]
            batch_logits = model(batch)  # [batch, p]
            logits_list.append(batch_logits.cpu())

    logits_flat = torch.cat(logits_list, dim=0).numpy().astype(np.float64)  # [p^2, p]

    # Reshape to grid: [p, p, p] where logit_grid[a, b, c] = logit for class c
    logit_grid = logits_flat.reshape(p, p, p)

    # Apply 2D FFT over (a, b) dimensions for each output class
    # fft_grid[k_a, k_b, c] = 2D Fourier coeff at frequencies (k_a, k_b) for class c
    fft_grid = np.fft.fft2(logit_grid, axes=(0, 1))

    # Compute power spectrum (magnitude squared, summed over output classes)
    power_2d = np.abs(fft_grid) ** 2
    power_2d_total = power_2d.sum(axis=2)  # [p, p]

    # Extract diagonal power: for modular addition, signal is at (k, k) for each k
    # This corresponds to cos(w_k * (a+b)) and sin(w_k * (a+b)) components
    diag_power = np.array([power_2d_total[k, k] for k in range(p)])

    # Also include anti-diagonal for completeness (k, -k mod p)
    # These are conjugate pairs that also contribute to (a+b) signal
    for k in range(1, p):
        diag_power[k] += power_2d_total[k, p - k]

    # Compute FVE for each frequency
    fve_per_freq = compute_fve_per_frequency(logit_grid, p)

    # Identify key frequencies from diagonal power
    key_frequencies = identify_key_frequencies(diag_power, threshold=0.15)

    # Compute total FVE for key frequencies
    if key_frequencies:
        fve_total = fve_per_freq[key_frequencies].sum()
    else:
        fve_total = 0.0

    return LogitSpectrum(
        logit_grid=logit_grid,
        power_2d=power_2d_total,
        diag_power=diag_power,
        key_frequencies=key_frequencies,
        fve_per_freq=fve_per_freq,
        fve_total=float(fve_total),
        p=p,
    )


def identify_key_frequencies(
    power: NDArray[np.float64],
    threshold: float = 0.15,
    top_k: Optional[int] = None,
    exclude_dc: bool = True,
) -> list[int]:
    """Identify frequencies with significant power.

    Args:
        power: [p] power spectrum
        threshold: Fraction of max power to consider significant (default 0.15)
        top_k: Alternative: just take top k frequencies (overrides threshold)
        exclude_dc: Whether to exclude k=0 (DC component)

    Returns:
        List of frequency indices sorted by power (descending)
    """
    p = len(power)

    # Only consider unique frequencies (k and p-k are conjugates)
    # For real signals, we only need k in [0, p//2]
    max_k = (p + 1) // 2

    if exclude_dc:
        candidates = list(range(1, max_k))
    else:
        candidates = list(range(max_k))

    if not candidates:
        return []

    candidate_power = [(k, power[k]) for k in candidates]

    if top_k is not None:
        # Take top k by power
        candidate_power.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in candidate_power[:top_k]]
    else:
        # Use threshold
        max_power = max(power[k] for k in candidates)
        threshold_value = threshold * max_power
        key_freqs = [k for k, pwr in candidate_power if pwr >= threshold_value]
        # Sort by power descending
        key_freqs.sort(key=lambda k: power[k], reverse=True)
        return key_freqs


def compute_fve_per_frequency(
    logit_grid: NDArray[np.float64],
    p: int,
) -> NDArray[np.float64]:
    """Compute fraction of variance explained by each diagonal frequency.

    For each frequency k, we compute how much of the logit variance is
    explained by the cos(w_k * (a+b)) and sin(w_k * (a+b)) components.

    Uses Parseval's theorem: sum of |FFT|^2 = N^2 * variance (for 2D FFT).
    FVE(k) = power_at_k / total_power.

    Args:
        logit_grid: [p, p, p] logits for all inputs
        p: Prime modulus

    Returns:
        [p] array of FVE values for each frequency (sum to ~1 if all freqs included)
    """
    fve = np.zeros(p)

    # Compute total power and per-frequency power across all output classes
    total_power = 0.0
    freq_power = np.zeros(p)

    for c in range(p):
        logits_c = logit_grid[:, :, c]  # [p, p]

        # Subtract mean for variance-based interpretation
        logits_centered = logits_c - logits_c.mean()

        # 2D FFT
        fft_c = np.fft.fft2(logits_centered)

        # Total power for this class (excluding DC since we centered)
        power_spectrum = np.abs(fft_c) ** 2
        total_power += power_spectrum.sum()

        # For each frequency k, the (a+b) component is at (k, k) and (k, p-k)
        for k in range(p):
            # Power at diagonal
            freq_power[k] += power_spectrum[k, k]
            # Power at anti-diagonal (conjugate pair)
            if k > 0 and k < p:
                freq_power[k] += power_spectrum[k, p - k]

    # Compute FVE as fraction of total power
    if total_power > 1e-10:
        fve = freq_power / total_power

    return fve


def create_restricted_logits(
    logit_grid: NDArray[np.float64],
    key_frequencies: list[int],
    p: int,
) -> NDArray[np.float64]:
    """Reconstruct logits keeping only key frequency components.

    Apply 2D FFT, zero out non-key frequencies, inverse FFT.

    Args:
        logit_grid: [p, p, p] original logits
        key_frequencies: Frequency indices to keep
        p: Prime modulus

    Returns:
        [p, p, p] reconstructed logits with only key frequencies
    """
    restricted = np.zeros_like(logit_grid)

    for c in range(p):
        fft_c = np.fft.fft2(logit_grid[:, :, c])

        # Create mask for key frequencies
        mask = np.zeros((p, p), dtype=bool)
        for k in key_frequencies:
            mask[k, k] = True  # Diagonal
            if k > 0:
                mask[k, p - k] = True  # Anti-diagonal
                mask[p - k, k] = True  # Conjugate
                mask[p - k, p - k] = True  # Conjugate

        # Also keep DC component (k=0)
        mask[0, 0] = True

        # Apply mask and inverse FFT
        fft_masked = fft_c * mask
        restricted[:, :, c] = np.real(np.fft.ifft2(fft_masked))

    return restricted


def create_excluded_logits(
    logit_grid: NDArray[np.float64],
    key_frequencies: list[int],
    p: int,
) -> NDArray[np.float64]:
    """Reconstruct logits removing key frequency components.

    Args:
        logit_grid: [p, p, p] original logits
        key_frequencies: Frequency indices to remove
        p: Prime modulus

    Returns:
        [p, p, p] reconstructed logits without key frequencies
    """
    excluded = np.zeros_like(logit_grid)

    for c in range(p):
        fft_c = np.fft.fft2(logit_grid[:, :, c])

        # Create mask to exclude key frequencies
        mask = np.ones((p, p), dtype=bool)
        for k in key_frequencies:
            mask[k, k] = False
            if k > 0:
                mask[k, p - k] = False
                mask[p - k, k] = False
                mask[p - k, p - k] = False

        # Apply mask and inverse FFT
        fft_masked = fft_c * mask
        excluded[:, :, c] = np.real(np.fft.ifft2(fft_masked))

    return excluded


def compute_ablation_metrics(
    logit_grid: NDArray[np.float64],
    key_frequencies: list[int],
    p: int,
    operation: str = "add",
) -> AblationResults:
    """Compute metrics for restricted and excluded logit ablations.

    Args:
        logit_grid: [p, p, p] logits for all (a, b) pairs
        key_frequencies: Frequencies to use for ablation
        p: Prime modulus
        operation: "add" or "multiply"

    Returns:
        AblationResults with full/restricted/excluded loss and accuracy
    """
    # Create labels grid
    a_vals = np.arange(p)
    b_vals = np.arange(p)
    aa, bb = np.meshgrid(a_vals, b_vals, indexing="ij")

    if operation == "add":
        labels = (aa + bb) % p
    else:
        labels = (aa * bb) % p

    def compute_metrics(logits: NDArray[np.float64]) -> tuple[float, float]:
        """Compute cross-entropy loss and accuracy from logits."""
        # Flatten for computation
        logits_flat = logits.reshape(-1, p)  # [p^2, p]
        labels_flat = labels.flatten()  # [p^2]

        # Softmax and cross-entropy
        logits_max = logits_flat.max(axis=1, keepdims=True)
        logits_stable = logits_flat - logits_max
        exp_logits = np.exp(logits_stable)
        softmax = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        # Cross-entropy: -log(p[correct class])
        correct_probs = softmax[np.arange(p * p), labels_flat]
        loss = -np.log(correct_probs + 1e-10).mean()

        # Accuracy
        predictions = logits_flat.argmax(axis=1)
        accuracy = (predictions == labels_flat).mean()

        return float(loss), float(accuracy)

    # Full logits
    full_loss, full_acc = compute_metrics(logit_grid)

    # Restricted (keep only key frequencies)
    restricted_logits = create_restricted_logits(logit_grid, key_frequencies, p)
    restricted_loss, restricted_acc = compute_metrics(restricted_logits)

    # Excluded (remove key frequencies)
    excluded_logits = create_excluded_logits(logit_grid, key_frequencies, p)
    excluded_loss, excluded_acc = compute_metrics(excluded_logits)

    return AblationResults(
        full_loss=full_loss,
        full_accuracy=full_acc,
        restricted_loss=restricted_loss,
        restricted_accuracy=restricted_acc,
        excluded_loss=excluded_loss,
        excluded_accuracy=excluded_acc,
        key_frequencies=key_frequencies,
    )


def analyze_checkpoint_frequency(
    experiment: ExperimentData,
    epoch: Optional[int] = None,
    threshold: float = 0.15,
    device: Optional[torch.device] = None,
) -> dict:
    """Full Fourier analysis of an experiment checkpoint.

    Loads the model from checkpoint, computes embedding spectrum, logit
    spectrum, and ablation metrics.

    Args:
        experiment: ExperimentData from ExperimentLoader
        epoch: Specific checkpoint epoch (None = best checkpoint)
        threshold: Threshold for key frequency detection
        device: Compute device (default: auto-detect)

    Returns:
        Dictionary with:
        - embedding_spectrum: EmbeddingSpectrum
        - logit_spectrum: LogitSpectrum
        - ablation_results: AblationResults
        - config: Experiment config dict
    """
    # Check model type
    model_type = experiment.config.get("model_type", "transformer")
    if model_type != "transformer":
        raise ValueError(f"Frequency analysis only supports transformer models, got {model_type}")

    # Get parameters from config
    p = experiment.p
    operation = experiment.config.get("operation", "add")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model from checkpoint
    loader = ExperimentLoader(experiment.output_dir.parent if experiment.output_dir else None)
    model = loader.reconstruct_model(experiment, epoch=epoch, device=device)
    model.eval()

    # Compute embedding spectrum
    embedding_spectrum = compute_embedding_spectrum(model, p, threshold=threshold)

    # Compute logit spectrum
    logit_spectrum = compute_logit_spectrum(model, p, operation=operation, device=device)

    # Use logit spectrum's key frequencies for ablation
    key_frequencies = logit_spectrum.key_frequencies

    # Compute ablation metrics
    ablation_results = compute_ablation_metrics(
        logit_spectrum.logit_grid, key_frequencies, p, operation=operation
    )

    return {
        "embedding_spectrum": embedding_spectrum,
        "logit_spectrum": logit_spectrum,
        "ablation_results": ablation_results,
        "config": experiment.config,
    }
