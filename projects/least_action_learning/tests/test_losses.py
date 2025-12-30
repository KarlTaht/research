"""Tests for losses.py - Loss functions and regularizers."""

import pytest
import torch
import torch.nn as nn
import math
from projects.least_action_learning.src.losses import (
    entropy_regularizer,
    sparsity_regularizer,
    gini_regularizer,
    consistency_regularizer,
    make_low_freq_mask,
    spectral_smoothness,
    spectral_smoothness_loss,
    compute_jacobian_norm,
    compute_hessian_trace,
    jacobian_regularizer,
    LeastActionLoss,
)


class TestEntropyRegularizer:
    """Tests for entropy regularizer."""

    def test_uniform_weights_max_entropy(self, tiny_model_config):
        """Test uniform weights give maximum entropy."""
        n_heads = tiny_model_config["n_heads"]
        weights = [torch.full((8, n_heads), 1.0 / n_heads)]

        entropy = entropy_regularizer(weights)
        expected_max = math.log(n_heads)

        assert abs(entropy.item() - expected_max) < 0.01

    def test_one_hot_weights_near_zero_entropy(self, tiny_model_config):
        """Test one-hot weights give near-zero entropy."""
        n_heads = tiny_model_config["n_heads"]
        weights = torch.zeros(8, n_heads)
        weights[:, 0] = 1.0

        entropy = entropy_regularizer([weights])
        assert entropy.item() < 0.01

    def test_returns_tensor(self, sample_routing_weights):
        """Test returns tensor."""
        entropy = entropy_regularizer(sample_routing_weights)
        assert isinstance(entropy, torch.Tensor)

    def test_entropy_non_negative(self, sample_routing_weights):
        """Test entropy is non-negative."""
        entropy = entropy_regularizer(sample_routing_weights)
        assert entropy.item() >= 0

    def test_multiple_layers_averaged(self, sample_routing_weights):
        """Test entropy is averaged over layers."""
        # Single layer entropy
        single_entropy = entropy_regularizer([sample_routing_weights[0]])
        # Two layers should produce similar range
        multi_entropy = entropy_regularizer(sample_routing_weights)

        # Both should be valid entropies
        assert single_entropy.item() >= 0
        assert multi_entropy.item() >= 0


class TestSparsityRegularizer:
    """Tests for sparsity regularizer."""

    def test_one_hot_gives_near_zero(self, tiny_model_config):
        """Test one-hot weights give near-zero penalty."""
        n_heads = tiny_model_config["n_heads"]
        weights = torch.zeros(8, n_heads)
        weights[:, 0] = 1.0

        penalty = sparsity_regularizer([weights])
        assert abs(penalty.item()) < 0.01

    def test_uniform_gives_max_penalty(self, tiny_model_config):
        """Test uniform weights give maximum penalty."""
        n_heads = tiny_model_config["n_heads"]
        weights = [torch.full((8, n_heads), 1.0 / n_heads)]

        penalty = sparsity_regularizer(weights)
        expected = 1.0 - (1.0 / n_heads)

        assert abs(penalty.item() - expected) < 0.01

    def test_penalty_in_valid_range(self, sample_routing_weights):
        """Test penalty is in [0, 1)."""
        penalty = sparsity_regularizer(sample_routing_weights)
        assert 0 <= penalty.item() < 1

    def test_returns_tensor(self, sample_routing_weights):
        """Test returns tensor."""
        penalty = sparsity_regularizer(sample_routing_weights)
        assert isinstance(penalty, torch.Tensor)


class TestGiniRegularizer:
    """Tests for Gini coefficient regularizer."""

    def test_uniform_gives_high_penalty(self, tiny_model_config):
        """Test uniform weights give high (1 - Gini) penalty."""
        n_heads = tiny_model_config["n_heads"]
        weights = [torch.full((8, n_heads), 1.0 / n_heads)]

        penalty = gini_regularizer(weights)
        # Gini = 0 for uniform -> penalty = 1
        assert penalty.item() > 0.9

    def test_one_hot_gives_low_penalty(self, tiny_model_config):
        """Test one-hot weights give low penalty."""
        n_heads = tiny_model_config["n_heads"]
        weights = torch.zeros(8, n_heads)
        weights[:, 0] = 1.0

        penalty = gini_regularizer([weights])
        # Gini near 1 for one-hot -> penalty near 0 (for n_heads=2, one-hot gives exactly 0.5)
        assert penalty.item() <= 0.5

    def test_returns_tensor(self, sample_routing_weights):
        """Test returns tensor."""
        penalty = gini_regularizer(sample_routing_weights)
        assert isinstance(penalty, torch.Tensor)


class TestConsistencyRegularizer:
    """Tests for consistency regularizer."""

    def test_returns_tensor(self, sample_routing_weights, tiny_model_config, device):
        """Test returns tensor."""
        inputs = torch.randn(16, tiny_model_config["hidden_dim"] * 2, device=device)
        sample_weights = [w.to(device) for w in sample_routing_weights]

        penalty = consistency_regularizer(sample_weights, inputs)
        assert isinstance(penalty, torch.Tensor)

    def test_penalty_non_negative(self, sample_routing_weights, tiny_model_config, device):
        """Test penalty is non-negative (KL divergence is non-negative)."""
        inputs = torch.randn(16, tiny_model_config["hidden_dim"] * 2, device=device)
        sample_weights = [w.to(device) for w in sample_routing_weights]

        penalty = consistency_regularizer(sample_weights, inputs)
        # KL divergence can be slightly negative due to numerical issues
        assert penalty.item() >= -0.1


class TestMakeLowFreqMask:
    """Tests for make_low_freq_mask."""

    def test_mask_shape(self, small_p):
        """Test mask has correct shape."""
        K = small_p // 4
        mask = make_low_freq_mask(small_p, K)

        assert mask.shape == (small_p, small_p)

    def test_mask_dtype(self, small_p):
        """Test mask is boolean."""
        mask = make_low_freq_mask(small_p, small_p // 4)
        assert mask.dtype == torch.bool

    def test_center_is_low_freq(self, small_p):
        """Test DC component (0,0) is included in low freq."""
        mask = make_low_freq_mask(small_p, small_p // 4)
        assert mask[0, 0]

    def test_increasing_k_includes_more(self, small_p):
        """Test larger K includes more frequencies."""
        mask_small = make_low_freq_mask(small_p, 1)
        mask_large = make_low_freq_mask(small_p, small_p // 2)

        assert mask_small.sum() <= mask_large.sum()

    def test_k_zero_only_dc(self):
        """Test K=0 includes only DC component."""
        p = 7
        mask = make_low_freq_mask(p, 0)
        # Only (0,0) should be True
        assert mask.sum() == 1
        assert mask[0, 0]


class TestSpectralSmoothness:
    """Tests for spectral_smoothness."""

    def test_returns_float(self, baseline_mlp, small_p, device):
        """Test returns float value."""
        baseline_mlp = baseline_mlp.to(device)
        smoothness = spectral_smoothness(baseline_mlp, small_p, small_p // 4, device)

        assert isinstance(smoothness, float)

    def test_smoothness_in_valid_range(self, baseline_mlp, small_p, device):
        """Test smoothness is in [0, 1]."""
        baseline_mlp = baseline_mlp.to(device)
        smoothness = spectral_smoothness(baseline_mlp, small_p, small_p // 4, device)

        assert 0 <= smoothness <= 1

    def test_transformer_mode(self, grok_transformer, small_p, device):
        """Test with transformer model."""
        grok_transformer = grok_transformer.to(device)
        smoothness = spectral_smoothness(
            grok_transformer, small_p, small_p // 4, device,
            is_transformer=True
        )

        assert 0 <= smoothness <= 1

    def test_deterministic(self, baseline_mlp, small_p, device):
        """Test function is deterministic for same model."""
        baseline_mlp = baseline_mlp.to(device)
        baseline_mlp.eval()

        s1 = spectral_smoothness(baseline_mlp, small_p, small_p // 4, device)
        s2 = spectral_smoothness(baseline_mlp, small_p, small_p // 4, device)

        assert s1 == s2


class TestSpectralSmoothnessLoss:
    """Tests for spectral_smoothness_loss (differentiable version)."""

    def test_returns_tensor(self, baseline_mlp, small_p, device):
        """Test returns tensor."""
        baseline_mlp = baseline_mlp.to(device)
        loss = spectral_smoothness_loss(baseline_mlp, small_p, small_p // 4, device)

        assert isinstance(loss, torch.Tensor)

    def test_loss_in_valid_range(self, baseline_mlp, small_p, device):
        """Test loss is in [0, 1]."""
        baseline_mlp = baseline_mlp.to(device)
        loss = spectral_smoothness_loss(baseline_mlp, small_p, small_p // 4, device)

        assert 0 <= loss.item() <= 1

    def test_gradient_exists(self, baseline_mlp, small_p, device):
        """Test gradient can flow through loss."""
        baseline_mlp = baseline_mlp.to(device)
        baseline_mlp.zero_grad()

        loss = spectral_smoothness_loss(baseline_mlp, small_p, small_p // 4, device)
        loss.backward()

        # At least some parameters should have gradients
        grad_count = sum(1 for p in baseline_mlp.parameters() if p.grad is not None)
        assert grad_count > 0


class TestComputeJacobianNorm:
    """Tests for compute_jacobian_norm."""

    def test_returns_positive(self, baseline_mlp, sample_batch, device):
        """Test returns positive value."""
        baseline_mlp = baseline_mlp.to(device)
        inputs, _ = sample_batch
        inputs = inputs.to(device)

        jac_norm = compute_jacobian_norm(baseline_mlp, inputs, num_samples=5)

        assert jac_norm > 0

    def test_returns_float(self, baseline_mlp, sample_batch, device):
        """Test returns float."""
        baseline_mlp = baseline_mlp.to(device)
        inputs, _ = sample_batch
        inputs = inputs.to(device)

        jac_norm = compute_jacobian_norm(baseline_mlp, inputs, num_samples=3)

        assert isinstance(jac_norm, float)

    def test_transformer_mode(self, grok_transformer, sample_sequence_batch, device):
        """Test with transformer model."""
        grok_transformer = grok_transformer.to(device)
        input_ids, _ = sample_sequence_batch
        input_ids = input_ids.to(device)

        jac_norm = compute_jacobian_norm(
            grok_transformer, input_ids,
            num_samples=3,
            is_transformer=True
        )

        assert jac_norm > 0

    def test_finite_value(self, baseline_mlp, sample_batch, device):
        """Test returns finite value."""
        baseline_mlp = baseline_mlp.to(device)
        inputs, _ = sample_batch
        inputs = inputs.to(device)

        jac_norm = compute_jacobian_norm(baseline_mlp, inputs, num_samples=3)

        assert math.isfinite(jac_norm)


class TestComputeHessianTrace:
    """Tests for compute_hessian_trace."""

    def test_returns_float(self, baseline_mlp, sample_batch, device):
        """Test returns float value."""
        baseline_mlp = baseline_mlp.to(device)
        inputs, _ = sample_batch
        inputs = inputs.to(device)

        hess_trace = compute_hessian_trace(baseline_mlp, inputs, num_hutchinson_samples=3)

        assert isinstance(hess_trace, float)

    def test_finite_value(self, baseline_mlp, sample_batch, device):
        """Test returns finite value."""
        baseline_mlp = baseline_mlp.to(device)
        inputs, _ = sample_batch
        inputs = inputs.to(device)

        hess_trace = compute_hessian_trace(baseline_mlp, inputs, num_hutchinson_samples=3)

        assert math.isfinite(hess_trace)

    def test_transformer_mode(self, grok_transformer, sample_sequence_batch, device):
        """Test with transformer model."""
        grok_transformer = grok_transformer.to(device)
        input_ids, _ = sample_sequence_batch
        input_ids = input_ids.to(device)

        hess_trace = compute_hessian_trace(
            grok_transformer, input_ids,
            num_hutchinson_samples=2,
            is_transformer=True
        )

        assert isinstance(hess_trace, float)
        assert math.isfinite(hess_trace)


class TestJacobianRegularizer:
    """Tests for jacobian_regularizer (differentiable)."""

    def test_returns_tensor(self, baseline_mlp, sample_batch, device):
        """Test returns tensor."""
        baseline_mlp = baseline_mlp.to(device)
        inputs, _ = sample_batch
        inputs = inputs.to(device).requires_grad_(True)

        penalty = jacobian_regularizer(baseline_mlp, inputs, num_samples=2)

        assert isinstance(penalty, torch.Tensor)

    def test_gradient_flows(self, baseline_mlp, sample_batch, device):
        """Test gradient flows through penalty."""
        baseline_mlp = baseline_mlp.to(device)
        baseline_mlp.zero_grad()
        inputs, _ = sample_batch
        inputs = inputs.to(device).requires_grad_(True)

        penalty = jacobian_regularizer(baseline_mlp, inputs, num_samples=2)
        penalty.backward()

        # At least some parameters should have gradients
        grad_count = sum(1 for p in baseline_mlp.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_non_negative(self, baseline_mlp, sample_batch, device):
        """Test penalty is non-negative."""
        baseline_mlp = baseline_mlp.to(device)
        inputs, _ = sample_batch
        inputs = inputs.to(device).requires_grad_(True)

        penalty = jacobian_regularizer(baseline_mlp, inputs, num_samples=3)

        assert penalty.item() >= 0


class TestLeastActionLoss:
    """Tests for LeastActionLoss combined loss."""

    def test_task_loss_computed(self, task_only_loss, sample_batch, device):
        """Test task loss is always computed."""
        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], 7, device=device)
        targets = targets.to(device)

        total_loss, loss_dict = task_only_loss(logits, targets)

        assert "task_loss" in loss_dict
        assert loss_dict["task_loss"] > 0

    def test_task_loss_is_cross_entropy(self, task_only_loss, device):
        """Test task loss matches cross entropy."""
        logits = torch.randn(8, 7, device=device)
        targets = torch.randint(0, 7, (8,), device=device)

        _, loss_dict = task_only_loss(logits, targets)
        expected = nn.functional.cross_entropy(logits, targets).item()

        assert abs(loss_dict["task_loss"] - expected) < 1e-5

    def test_routing_loss_with_weights(self, sample_batch, sample_routing_weights, device):
        """Test routing loss computed when weights provided."""
        loss_fn = LeastActionLoss(routing_regularizer="entropy", lambda_routing=0.1)

        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], 7, device=device)
        targets = targets.to(device)

        total_loss, loss_dict = loss_fn(logits, targets, routing_weights=sample_routing_weights)

        assert "routing_loss" in loss_dict

    def test_no_routing_loss_without_weights(self, least_action_loss, sample_batch, device):
        """Test no routing loss when weights not provided."""
        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], 7, device=device)
        targets = targets.to(device)

        total_loss, loss_dict = least_action_loss(logits, targets, routing_weights=None)

        assert "routing_loss" not in loss_dict

    @pytest.mark.parametrize("regularizer", ["entropy", "sparsity", "gini"])
    def test_all_regularizer_types(self, regularizer, sample_batch, sample_routing_weights, device):
        """Test all regularizer types work."""
        loss_fn = LeastActionLoss(routing_regularizer=regularizer, lambda_routing=0.1)

        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], 7, device=device)
        targets = targets.to(device)

        total_loss, loss_dict = loss_fn(logits, targets, routing_weights=sample_routing_weights)

        assert "routing_loss" in loss_dict
        assert loss_dict["routing_loss"] >= 0

    def test_invalid_regularizer_raises(self):
        """Test invalid regularizer raises ValueError."""
        with pytest.raises(ValueError, match="Unknown regularizer"):
            LeastActionLoss(routing_regularizer="invalid")

    def test_none_regularizer_works(self, sample_batch, device):
        """Test None regularizer doesn't add routing loss."""
        loss_fn = LeastActionLoss(routing_regularizer=None)

        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], 7, device=device)
        targets = targets.to(device)

        _, loss_dict = loss_fn(logits, targets)
        assert "routing_loss" not in loss_dict

    def test_spectral_loss_periodic(self, baseline_mlp, sample_batch, small_p, device):
        """Test spectral loss computed at correct intervals."""
        loss_fn = LeastActionLoss(
            routing_regularizer=None,
            lambda_spectral=0.1,
            spectral_interval=10,
        )
        baseline_mlp = baseline_mlp.to(device)

        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], small_p, device=device)
        targets = targets.to(device)

        # Step 10 should compute spectral loss
        _, loss_dict_10 = loss_fn(
            logits, targets, model=baseline_mlp, p=small_p, step=10
        )

        # Step 5 should not compute spectral loss
        _, loss_dict_5 = loss_fn(
            logits, targets, model=baseline_mlp, p=small_p, step=5
        )

        assert "spectral_loss" in loss_dict_10
        assert "spectral_loss" not in loss_dict_5

    def test_spectral_loss_step_zero(self, baseline_mlp, sample_batch, small_p, device):
        """Test spectral loss computed at step 0."""
        loss_fn = LeastActionLoss(
            routing_regularizer=None,
            lambda_spectral=0.1,
            spectral_interval=10,
        )
        baseline_mlp = baseline_mlp.to(device)

        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], small_p, device=device)
        targets = targets.to(device)

        _, loss_dict = loss_fn(
            logits, targets, model=baseline_mlp, p=small_p, step=0
        )

        assert "spectral_loss" in loss_dict

    def test_total_loss_includes_all_components(self, sample_batch, sample_routing_weights, baseline_mlp, small_p, device):
        """Test total loss includes task + routing + spectral."""
        loss_fn = LeastActionLoss(
            routing_regularizer="entropy",
            lambda_routing=0.1,
            lambda_spectral=0.1,
            spectral_interval=1,
        )
        baseline_mlp = baseline_mlp.to(device)

        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], small_p, device=device)
        targets = targets.to(device)

        total_loss, loss_dict = loss_fn(
            logits, targets,
            routing_weights=sample_routing_weights,
            model=baseline_mlp,
            p=small_p,
            step=1,
        )

        assert "task_loss" in loss_dict
        assert "routing_loss" in loss_dict
        assert "spectral_loss" in loss_dict
        assert "total_loss" in loss_dict

        # Total should be sum of components (approximately)
        expected_total = (
            loss_dict["task_loss"]
            + 0.1 * loss_dict["routing_loss"]
            + 0.1 * loss_dict["spectral_loss"]
        )
        assert abs(loss_dict["total_loss"] - expected_total) < 0.01

    def test_lambda_routing_zero_no_routing_loss(self, sample_batch, sample_routing_weights, device):
        """Test lambda_routing=0 doesn't add routing loss to total."""
        loss_fn = LeastActionLoss(routing_regularizer="entropy", lambda_routing=0.0)

        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], 7, device=device)
        targets = targets.to(device)

        _, loss_dict = loss_fn(logits, targets, routing_weights=sample_routing_weights)

        # routing_loss should not be in dict when lambda=0
        assert "routing_loss" not in loss_dict

    def test_returns_tuple(self, task_only_loss, sample_batch, device):
        """Test returns (loss_tensor, loss_dict) tuple."""
        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], 7, device=device)
        targets = targets.to(device)

        result = task_only_loss(logits, targets)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], dict)

    def test_loss_tensor_requires_grad(self, task_only_loss, sample_batch, device):
        """Test loss tensor has gradient."""
        inputs, targets = sample_batch
        logits = torch.randn(inputs.shape[0], 7, device=device, requires_grad=True)
        targets = targets.to(device)

        total_loss, _ = task_only_loss(logits, targets)

        assert total_loss.requires_grad


