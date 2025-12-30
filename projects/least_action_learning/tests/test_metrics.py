"""Tests for metrics module (RoutingMetrics, TrainingMetrics, MetricsHistory)."""

import pytest
import torch
import math
import pandas as pd
from typing import Optional

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from projects.least_action_learning.src.metrics import (
    RoutingMetrics,
    TrainingMetrics,
    MetricsHistory,
    compute_routing_entropy,
    compute_head_utilization,
    compute_routing_consistency,
    get_dominant_head_per_layer,
    get_routing_path_signature,
    compute_layer_weight_norms,
    compute_total_weight_norm,
    compute_representation_norm,
)


class TestRoutingMetrics:
    """Tests for RoutingMetrics dataclass."""

    def test_creation(self, sample_routing_weights, tiny_model_config):
        """Test RoutingMetrics creation."""
        n_heads = tiny_model_config["n_heads"]
        metrics = RoutingMetrics(
            layer_weights=sample_routing_weights,
            routing_entropy=0.5,
            head_utilization=torch.ones(n_heads) / n_heads,
        )

        assert len(metrics.layer_weights) == len(sample_routing_weights)
        assert metrics.routing_entropy == 0.5
        assert metrics.head_utilization.shape == (n_heads,)

    def test_to_dict(self, sample_routing_weights, tiny_model_config):
        """Test conversion to dictionary."""
        n_heads = tiny_model_config["n_heads"]
        n_layers = len(sample_routing_weights)
        metrics = RoutingMetrics(
            layer_weights=sample_routing_weights,
            routing_entropy=0.5,
            head_utilization=torch.ones(n_heads) / n_heads,
        )

        d = metrics.to_dict()

        assert "routing_entropy" in d
        assert "head_utilization" in d
        assert "n_layers" in d
        assert d["n_layers"] == n_layers
        assert d["routing_entropy"] == 0.5
        assert isinstance(d["head_utilization"], list)


class TestComputeRoutingEntropy:
    """Tests for compute_routing_entropy function."""

    def test_uniform_gives_max_entropy(self, tiny_model_config):
        """Test uniform weights give maximum entropy."""
        n_heads = tiny_model_config["n_heads"]
        batch_size = 16

        # Uniform weights
        weights = torch.full((batch_size, n_heads), 1.0 / n_heads)

        entropy = compute_routing_entropy(weights)
        max_entropy = math.log(n_heads)

        assert abs(entropy - max_entropy) < 0.1

    def test_one_hot_gives_near_zero_entropy(self, tiny_model_config):
        """Test one-hot weights give near-zero entropy."""
        n_heads = tiny_model_config["n_heads"]
        batch_size = 16

        # One-hot weights (all weight on first head)
        weights = torch.zeros(batch_size, n_heads)
        weights[:, 0] = 1.0

        entropy = compute_routing_entropy(weights)

        # Should be very close to 0 (but not exactly due to eps clamping)
        assert entropy < 0.1

    def test_returns_float(self, sample_routing_weights):
        """Test returns a float."""
        entropy = compute_routing_entropy(sample_routing_weights[0])
        assert isinstance(entropy, float)

    def test_entropy_non_negative(self, sample_routing_weights):
        """Test entropy is non-negative."""
        for weights in sample_routing_weights:
            entropy = compute_routing_entropy(weights)
            assert entropy >= 0


class TestComputeHeadUtilization:
    """Tests for compute_head_utilization function."""

    def test_output_shape(self, sample_routing_weights, tiny_model_config):
        """Test output shape matches n_heads."""
        n_heads = tiny_model_config["n_heads"]
        utilization = compute_head_utilization(sample_routing_weights)

        assert utilization.shape == (n_heads,)

    def test_uniform_weights_give_uniform_utilization(self, uniform_routing_weights, tiny_model_config):
        """Test uniform weights give uniform utilization."""
        n_heads = tiny_model_config["n_heads"]
        utilization = compute_head_utilization(uniform_routing_weights)
        expected = torch.full((n_heads,), 1.0 / n_heads)

        assert torch.allclose(utilization, expected, atol=1e-5)

    def test_utilization_sums_to_one(self, sample_routing_weights):
        """Test utilization sums to approximately 1."""
        utilization = compute_head_utilization(sample_routing_weights)
        total = utilization.sum().item()

        assert abs(total - 1.0) < 1e-5

    def test_utilization_in_valid_range(self, sample_routing_weights):
        """Test utilization values are in [0, 1]."""
        utilization = compute_head_utilization(sample_routing_weights)

        assert (utilization >= 0).all()
        assert (utilization <= 1).all()


class TestComputeRoutingConsistency:
    """Tests for compute_routing_consistency function."""

    def test_returns_float(self, sample_routing_weights, small_p):
        """Test returns a float."""
        batch_size = sample_routing_weights[0].shape[0]

        # Create dummy pairs and similarity mask
        pairs = torch.randint(0, small_p, (batch_size, 2))
        similar_mask = torch.eye(batch_size, dtype=torch.bool)

        consistency = compute_routing_consistency(sample_routing_weights, pairs, similar_mask)

        assert isinstance(consistency, float)

    def test_consistency_non_negative(self, sample_routing_weights, small_p):
        """Test consistency (variance) is non-negative."""
        batch_size = sample_routing_weights[0].shape[0]

        pairs = torch.randint(0, small_p, (batch_size, 2))
        # Mark some inputs as similar
        similar_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        similar_mask[:4, :4] = True  # First 4 inputs are similar

        consistency = compute_routing_consistency(sample_routing_weights, pairs, similar_mask)

        assert consistency >= 0


class TestGetDominantHeadPerLayer:
    """Tests for get_dominant_head_per_layer function."""

    def test_output_shape(self, sample_routing_weights, tiny_model_config):
        """Test output shape."""
        n_layers = len(sample_routing_weights)
        batch_size = sample_routing_weights[0].shape[0]

        dominant = get_dominant_head_per_layer(sample_routing_weights)

        assert dominant.shape == (n_layers, batch_size)

    def test_output_dtype(self, sample_routing_weights):
        """Test output is integer type (indices)."""
        dominant = get_dominant_head_per_layer(sample_routing_weights)
        assert dominant.dtype == torch.int64

    def test_one_hot_gives_correct_head(self, one_hot_routing_weights, tiny_model_config):
        """Test one-hot weights give correct dominant head."""
        dominant = get_dominant_head_per_layer(one_hot_routing_weights)

        # All one-hot weights are on head 0
        assert (dominant == 0).all()

    def test_indices_in_valid_range(self, sample_routing_weights, tiny_model_config):
        """Test dominant head indices are in valid range."""
        n_heads = tiny_model_config["n_heads"]
        dominant = get_dominant_head_per_layer(sample_routing_weights)

        assert (dominant >= 0).all()
        assert (dominant < n_heads).all()


class TestGetRoutingPathSignature:
    """Tests for get_routing_path_signature function."""

    def test_output_shape(self, sample_routing_weights, tiny_model_config):
        """Test output shape."""
        n_layers = len(sample_routing_weights)
        batch_size = sample_routing_weights[0].shape[0]

        signature = get_routing_path_signature(sample_routing_weights)

        assert signature.shape == (batch_size, n_layers)

    def test_output_dtype(self, sample_routing_weights):
        """Test output is integer type."""
        signature = get_routing_path_signature(sample_routing_weights)
        assert signature.dtype == torch.int64

    def test_transposed_from_dominant(self, sample_routing_weights):
        """Test signature is transposed dominant heads."""
        dominant = get_dominant_head_per_layer(sample_routing_weights)
        signature = get_routing_path_signature(sample_routing_weights)

        assert torch.allclose(dominant.T, signature)


class TestComputeLayerWeightNorms:
    """Tests for compute_layer_weight_norms function."""

    def test_returns_list(self, baseline_mlp):
        """Test returns a list."""
        norms = compute_layer_weight_norms(baseline_mlp)
        assert isinstance(norms, list)

    def test_all_norms_positive(self, baseline_mlp):
        """Test all norms are positive."""
        norms = compute_layer_weight_norms(baseline_mlp)
        for norm in norms:
            assert norm > 0

    def test_baseline_mlp_has_norms(self, baseline_mlp):
        """Test baseline MLP returns norms."""
        norms = compute_layer_weight_norms(baseline_mlp)
        assert len(norms) > 0

    def test_routed_network_has_norms(self, routed_network):
        """Test routed network returns norms."""
        norms = compute_layer_weight_norms(routed_network)
        assert len(norms) > 0

    def test_transformer_has_norms(self, grok_transformer):
        """Test transformer returns norms (via fallback)."""
        norms = compute_layer_weight_norms(grok_transformer)
        assert len(norms) > 0


class TestComputeTotalWeightNorm:
    """Tests for compute_total_weight_norm function."""

    def test_returns_float(self, baseline_mlp):
        """Test returns a float."""
        norm = compute_total_weight_norm(baseline_mlp)
        assert isinstance(norm, float)

    def test_norm_positive(self, baseline_mlp):
        """Test norm is positive."""
        norm = compute_total_weight_norm(baseline_mlp)
        assert norm > 0

    def test_norm_increases_with_scaled_weights(self, baseline_mlp):
        """Test norm increases when weights are scaled."""
        norm1 = compute_total_weight_norm(baseline_mlp)

        # Scale all weights by 2
        with torch.no_grad():
            for param in baseline_mlp.parameters():
                param.mul_(2.0)

        norm2 = compute_total_weight_norm(baseline_mlp)

        # Should be approximately 2x
        assert abs(norm2 / norm1 - 2.0) < 0.1

    def test_different_models_different_norms(self, baseline_mlp, routed_network):
        """Test different models have different norms."""
        norm1 = compute_total_weight_norm(baseline_mlp)
        norm2 = compute_total_weight_norm(routed_network)

        assert norm1 != norm2


class TestComputeRepresentationNorm:
    """Tests for compute_representation_norm function."""

    def test_returns_float(self, baseline_mlp, sample_batch, device):
        """Test returns a float."""
        inputs, _ = sample_batch
        baseline_mlp = baseline_mlp.to(device)
        inputs = inputs.to(device)

        norm = compute_representation_norm(baseline_mlp, inputs)
        assert isinstance(norm, float)

    def test_norm_positive(self, baseline_mlp, sample_batch, device):
        """Test norm is positive."""
        inputs, _ = sample_batch
        baseline_mlp = baseline_mlp.to(device)
        inputs = inputs.to(device)

        norm = compute_representation_norm(baseline_mlp, inputs)
        assert norm > 0

    def test_routed_network(self, routed_network, sample_batch, device):
        """Test works with routed network."""
        inputs, _ = sample_batch
        routed_network = routed_network.to(device)
        inputs = inputs.to(device)

        norm = compute_representation_norm(routed_network, inputs)
        assert norm > 0

    def test_transformer(self, grok_transformer, sample_sequence_batch, device):
        """Test works with transformer."""
        input_ids, _ = sample_sequence_batch
        grok_transformer = grok_transformer.to(device)
        input_ids = input_ids.to(device)

        norm = compute_representation_norm(grok_transformer, input_ids)
        assert norm > 0


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_creation(self, sample_training_metrics):
        """Test TrainingMetrics creation."""
        m = sample_training_metrics

        assert m.step == 100
        assert m.train_loss == 0.5
        assert m.train_acc == 0.8
        assert m.test_loss == 0.6
        assert m.test_acc == 0.7

    def test_to_dict_basic_fields(self, sample_training_metrics):
        """Test basic fields are in dict."""
        d = sample_training_metrics.to_dict()

        assert "step" in d
        assert "train_loss" in d
        assert "train_acc" in d
        assert "test_loss" in d
        assert "test_acc" in d
        assert "routing_entropy" in d

    def test_to_dict_head_utilization(self, sample_training_metrics):
        """Test head utilization is expanded in dict."""
        d = sample_training_metrics.to_dict()

        assert "head_0_utilization" in d
        assert "head_1_utilization" in d

    def test_to_dict_optional_fields(self, sample_training_metrics):
        """Test optional fields are in dict when set."""
        d = sample_training_metrics.to_dict()

        # These are set in the fixture
        assert "spectral_smoothness" in d
        assert "total_weight_norm" in d
        assert "representation_norm" in d
        assert "jacobian_norm" in d
        assert "hessian_trace" in d

    def test_to_dict_missing_optional_fields(self):
        """Test optional fields are not in dict when None."""
        m = TrainingMetrics(
            step=1,
            train_loss=0.1,
            train_acc=0.9,
            test_loss=0.2,
            test_acc=0.8,
            routing_entropy=0.5,
            head_utilization=[0.5, 0.5],
            # Optional fields left as None
        )
        d = m.to_dict()

        assert "spectral_smoothness" not in d
        assert "total_weight_norm" not in d
        assert "representation_norm" not in d
        assert "jacobian_norm" not in d
        assert "hessian_trace" not in d

    def test_to_dict_layer_weight_norms(self):
        """Test layer weight norms are expanded in dict."""
        m = TrainingMetrics(
            step=1,
            train_loss=0.1,
            train_acc=0.9,
            test_loss=0.2,
            test_acc=0.8,
            routing_entropy=0.5,
            head_utilization=[0.5, 0.5],
            layer_weight_norms=[1.0, 2.0, 3.0],
        )
        d = m.to_dict()

        assert "layer_0_weight_norm" in d
        assert "layer_1_weight_norm" in d
        assert "layer_2_weight_norm" in d


class TestMetricsHistory:
    """Tests for MetricsHistory class."""

    def test_empty_creation(self, metrics_history):
        """Test empty history creation."""
        assert len(metrics_history.history) == 0
        assert len(metrics_history.routing_weights_history) == 0

    def test_log_adds_metrics(self, metrics_history, sample_training_metrics):
        """Test log adds metrics to history."""
        metrics_history.log(sample_training_metrics)

        assert len(metrics_history.history) == 1
        assert metrics_history.history[0] == sample_training_metrics

    def test_log_with_routing_weights(self, metrics_history, sample_training_metrics, sample_routing_weights):
        """Test log with routing weights."""
        metrics_history.log(sample_training_metrics, routing_weights=sample_routing_weights)

        assert len(metrics_history.history) == 1
        assert len(metrics_history.routing_weights_history) == 1
        assert len(metrics_history.routing_weights_history[0]) == len(sample_routing_weights)

    def test_multiple_logs(self, metrics_history, sample_training_metrics):
        """Test multiple logs accumulate."""
        for i in range(5):
            m = TrainingMetrics(
                step=i,
                train_loss=0.5 - i * 0.1,
                train_acc=0.5 + i * 0.1,
                test_loss=0.6 - i * 0.1,
                test_acc=0.4 + i * 0.1,
                routing_entropy=0.5,
                head_utilization=[0.5, 0.5],
            )
            metrics_history.log(m)

        assert len(metrics_history.history) == 5

    def test_get_dataframe(self, metrics_history, sample_training_metrics):
        """Test conversion to DataFrame."""
        for i in range(3):
            m = TrainingMetrics(
                step=i * 100,
                train_loss=0.5,
                train_acc=0.5,
                test_loss=0.6,
                test_acc=0.4,
                routing_entropy=0.5,
                head_utilization=[0.5, 0.5],
            )
            metrics_history.log(m)

        df = metrics_history.get_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "step" in df.columns
        assert "train_loss" in df.columns

    def test_get_grokking_step_found(self, metrics_history):
        """Test finding grokking step when threshold is reached."""
        for i in range(10):
            m = TrainingMetrics(
                step=i * 100,
                train_loss=0.5,
                train_acc=0.5,
                test_loss=0.5,
                test_acc=0.1 * i + 0.1,  # Increases from 0.1 to 1.0
                routing_entropy=0.5,
                head_utilization=[0.5, 0.5],
            )
            metrics_history.log(m)

        # 95% threshold reached at step 900 (0.1*9 + 0.1 = 1.0)
        # Actually at i=9, test_acc=1.0 > 0.95
        # At i=8, test_acc=0.9 < 0.95
        # So at i=9, step=900, we cross threshold
        grok_step = metrics_history.get_grokking_step(accuracy_threshold=0.95)

        assert grok_step == 900

    def test_get_grokking_step_not_found(self, metrics_history):
        """Test grokking step returns None when threshold not reached."""
        for i in range(5):
            m = TrainingMetrics(
                step=i * 100,
                train_loss=0.5,
                train_acc=0.5,
                test_loss=0.5,
                test_acc=0.5,  # Never reaches 95%
                routing_entropy=0.5,
                head_utilization=[0.5, 0.5],
            )
            metrics_history.log(m)

        grok_step = metrics_history.get_grokking_step(accuracy_threshold=0.95)

        assert grok_step is None

    def test_get_grokking_step_custom_threshold(self, metrics_history):
        """Test grokking step with custom threshold."""
        for i in range(10):
            m = TrainingMetrics(
                step=i * 100,
                train_loss=0.5,
                train_acc=0.5,
                test_loss=0.5,
                test_acc=0.1 * (i + 1),  # 0.1, 0.2, ... 1.0
                routing_entropy=0.5,
                head_utilization=[0.5, 0.5],
            )
            metrics_history.log(m)

        # 50% threshold reached at step 500 (0.1*6 = 0.6 > 0.5)
        grok_step = metrics_history.get_grokking_step(accuracy_threshold=0.5)

        assert grok_step == 400  # i=4 gives 0.5, i=5 gives 0.6 > 0.5

    def test_routing_weights_cloned(self, metrics_history, sample_training_metrics, sample_routing_weights):
        """Test routing weights are cloned (not references)."""
        metrics_history.log(sample_training_metrics, routing_weights=sample_routing_weights)

        # Modify original
        sample_routing_weights[0].fill_(0.0)

        # Stored should be unchanged
        assert not torch.allclose(
            metrics_history.routing_weights_history[0][0],
            torch.zeros_like(metrics_history.routing_weights_history[0][0])
        )

    def test_routing_weights_on_cpu(self, metrics_history, sample_training_metrics, sample_routing_weights, device):
        """Test routing weights are moved to CPU."""
        # Move to device first
        weights_on_device = [w.to(device) for w in sample_routing_weights]
        metrics_history.log(sample_training_metrics, routing_weights=weights_on_device)

        # Check stored on CPU
        for w in metrics_history.routing_weights_history[0]:
            assert w.device == torch.device("cpu")
