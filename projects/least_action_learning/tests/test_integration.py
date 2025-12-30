"""Integration tests for least_action_learning project.

These tests verify that different modules work together correctly.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from projects.least_action_learning.src.data import (
    ModularArithmeticDataset,
    SequenceArithmeticDataset,
)
from projects.least_action_learning.src.models import (
    BaselineMLP,
    RoutedNetwork,
    GrokTransformer,
    create_model,
)
from projects.least_action_learning.src.losses import (
    LeastActionLoss,
    spectral_smoothness,
    compute_jacobian_norm,
    compute_hessian_trace,
)
from projects.least_action_learning.src.metrics import (
    TrainingMetrics,
    MetricsHistory,
    compute_routing_entropy,
    compute_head_utilization,
    compute_total_weight_norm,
    compute_representation_norm,
)
from projects.least_action_learning.src.routing import RoutingGate, RoutedLayer


class TestDataModelIntegration:
    """Tests for data + model integration."""

    def test_baseline_mlp_with_modular_dataset(self, small_p, device):
        """Test baseline MLP works with modular arithmetic dataset."""
        dataset = ModularArithmeticDataset(p=small_p, operation="add", train_frac=0.3)

        model = BaselineMLP(
            input_dim=dataset.input_dim,
            hidden_dim=32,
            output_dim=dataset.output_dim,
            n_layers=2,
        ).to(device)

        train = dataset.get_train()
        inputs = train.inputs.to(device)
        targets = train.targets.to(device)

        # Forward pass
        logits = model(inputs)

        # Check shapes
        assert logits.shape == (len(inputs), small_p)

        # Loss computation
        loss = nn.functional.cross_entropy(logits, targets)
        assert loss.item() > 0

    def test_routed_network_with_modular_dataset(self, small_p, device):
        """Test routed network works with modular arithmetic dataset."""
        dataset = ModularArithmeticDataset(p=small_p, operation="multiply", train_frac=0.3)

        model = RoutedNetwork(
            input_dim=dataset.input_dim,
            hidden_dim=32,
            output_dim=dataset.output_dim,
            n_layers=2,
            n_heads=4,
        ).to(device)

        train = dataset.get_train()
        inputs = train.inputs.to(device)
        targets = train.targets.to(device)

        # Forward pass returns (logits, routing_metrics)
        logits, metrics = model(inputs)

        # Check shapes
        assert logits.shape == (len(inputs), small_p)
        assert len(metrics.layer_weights) == 2  # n_layers
        assert metrics.layer_weights[0].shape == (len(inputs), 4)  # (batch, n_heads)

    def test_transformer_with_sequence_dataset(self, small_p, device):
        """Test transformer works with sequence dataset."""
        dataset = SequenceArithmeticDataset(p=small_p, operation="add", train_frac=0.5)

        model = GrokTransformer(
            vocab_size=dataset.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            output_dim=dataset.output_dim,
            max_seq_len=5,
        ).to(device)

        train = dataset.get_train()
        input_ids = train.input_ids.to(device)
        targets = train.targets.to(device)

        # Forward pass
        logits = model(input_ids)

        # Check shapes
        assert logits.shape == (len(input_ids), small_p)

        # Loss computation
        loss = nn.functional.cross_entropy(logits, targets)
        assert loss.item() > 0


class TestModelLossIntegration:
    """Tests for model + loss integration."""

    def test_baseline_with_least_action_loss(self, sample_batch, device):
        """Test baseline MLP with LeastActionLoss."""
        inputs, targets = sample_batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        model = BaselineMLP(
            input_dim=inputs.shape[-1],
            hidden_dim=32,
            output_dim=7,  # small_p
            n_layers=2,
        ).to(device)

        loss_fn = LeastActionLoss(
            routing_regularizer=None,  # Baseline has no routing
            lambda_routing=0.0,
        )

        logits = model(inputs)
        total_loss, loss_dict = loss_fn(logits, targets)

        assert "task_loss" in loss_dict
        assert total_loss.item() > 0
        # No routing loss when routing_weights is None
        assert loss_dict.get("routing_loss", 0.0) == 0.0

    def test_routed_with_entropy_regularizer(self, sample_batch, device):
        """Test routed network with entropy regularization."""
        inputs, targets = sample_batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        model = RoutedNetwork(
            input_dim=inputs.shape[-1],
            hidden_dim=32,
            output_dim=7,
            n_layers=2,
            n_heads=4,
        ).to(device)

        loss_fn = LeastActionLoss(
            routing_regularizer="entropy",
            lambda_routing=0.01,
        )

        logits, metrics = model(inputs)
        total_loss, loss_dict = loss_fn(
            logits, targets,
            routing_weights=metrics.layer_weights,
        )

        assert "task_loss" in loss_dict
        assert "routing_loss" in loss_dict
        assert loss_dict["routing_loss"] > 0  # Entropy regularizer adds loss


class TestModelMetricsIntegration:
    """Tests for model + metrics integration."""

    def test_compute_metrics_for_baseline(self, baseline_mlp, sample_batch, device):
        """Test computing metrics for baseline MLP."""
        inputs, _ = sample_batch
        baseline_mlp = baseline_mlp.to(device)
        inputs = inputs.to(device)

        # Weight norms
        total_norm = compute_total_weight_norm(baseline_mlp)
        assert total_norm > 0

        # Representation norm
        repr_norm = compute_representation_norm(baseline_mlp, inputs)
        assert repr_norm > 0

    def test_compute_metrics_for_routed(self, routed_network, sample_batch, device):
        """Test computing metrics for routed network."""
        inputs, _ = sample_batch
        routed_network = routed_network.to(device)
        inputs = inputs.to(device)

        # Forward pass to get routing weights
        _, metrics = routed_network(inputs)

        # Routing metrics
        entropy = compute_routing_entropy(metrics.layer_weights[0])
        assert entropy >= 0

        utilization = compute_head_utilization(metrics.layer_weights)
        assert utilization.shape == (routed_network.n_heads,)
        assert torch.allclose(utilization.sum(), torch.tensor(1.0), atol=1e-5)

    def test_compute_metrics_for_transformer(self, grok_transformer, sample_sequence_batch, device):
        """Test computing metrics for transformer."""
        input_ids, _ = sample_sequence_batch
        grok_transformer = grok_transformer.to(device)
        input_ids = input_ids.to(device)

        # Weight norms
        total_norm = compute_total_weight_norm(grok_transformer)
        assert total_norm > 0

        # Representation norm
        repr_norm = compute_representation_norm(grok_transformer, input_ids)
        assert repr_norm > 0


class TestSpectralAnalysisIntegration:
    """Tests for spectral analysis integration."""

    def test_spectral_smoothness_baseline(self, small_p, device):
        """Test spectral smoothness for baseline MLP."""
        dataset = ModularArithmeticDataset(p=small_p, operation="add", train_frac=0.3)

        model = BaselineMLP(
            input_dim=dataset.input_dim,
            hidden_dim=32,
            output_dim=dataset.output_dim,
            n_layers=2,
        ).to(device)

        smoothness = spectral_smoothness(model, small_p, K=2, device=device, is_transformer=False)
        assert 0.0 <= smoothness <= 1.0

    def test_spectral_smoothness_transformer(self, small_p, device):
        """Test spectral smoothness for transformer."""
        dataset = SequenceArithmeticDataset(p=small_p, operation="add", train_frac=0.5)

        model = GrokTransformer(
            vocab_size=dataset.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=2,
            output_dim=dataset.output_dim,
            max_seq_len=5,
        ).to(device)

        smoothness = spectral_smoothness(model, small_p, K=2, device=device, is_transformer=True)
        assert 0.0 <= smoothness <= 1.0


class TestCurvatureMetricsIntegration:
    """Tests for curvature metrics integration."""

    def test_jacobian_norm_baseline(self, baseline_mlp, sample_batch, device):
        """Test Jacobian norm for baseline MLP."""
        inputs, _ = sample_batch
        baseline_mlp = baseline_mlp.to(device)
        inputs = inputs.to(device)

        jac_norm = compute_jacobian_norm(baseline_mlp, inputs, num_samples=5, is_transformer=False)
        assert jac_norm > 0
        assert not torch.isnan(torch.tensor(jac_norm))

    def test_jacobian_norm_transformer(self, grok_transformer, sample_sequence_batch, device):
        """Test Jacobian norm for transformer."""
        input_ids, _ = sample_sequence_batch
        grok_transformer = grok_transformer.to(device)
        input_ids = input_ids.to(device)

        jac_norm = compute_jacobian_norm(grok_transformer, input_ids, num_samples=5, is_transformer=True)
        assert jac_norm > 0
        assert not torch.isnan(torch.tensor(jac_norm))

    def test_hessian_trace_baseline(self, baseline_mlp, sample_batch, device):
        """Test Hessian trace for baseline MLP."""
        inputs, _ = sample_batch
        baseline_mlp = baseline_mlp.to(device)
        inputs = inputs.to(device)

        hess_trace = compute_hessian_trace(baseline_mlp, inputs, num_hutchinson_samples=3, is_transformer=False)
        assert not torch.isnan(torch.tensor(hess_trace))

    def test_hessian_trace_transformer(self, grok_transformer, sample_sequence_batch, device):
        """Test Hessian trace for transformer."""
        input_ids, _ = sample_sequence_batch
        grok_transformer = grok_transformer.to(device)
        input_ids = input_ids.to(device)

        hess_trace = compute_hessian_trace(grok_transformer, input_ids, num_hutchinson_samples=3, is_transformer=True)
        assert not torch.isnan(torch.tensor(hess_trace))


class TestRoutingLayerIntegration:
    """Tests for routing layer integration."""

    def test_routing_gate_with_routed_layer(self, tiny_model_config, device):
        """Test routing gate integrates with routed layer."""
        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]

        layer = RoutedLayer(hidden_dim, n_heads).to(device)

        batch_size = 8
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.zeros(batch_size, hidden_dim, device=device)

        output, new_state, weights = layer(x, routing_state)

        # Verify gate produces valid weights
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5)

        # Verify output has correct shape
        assert output.shape == (batch_size, hidden_dim)

    def test_multi_layer_routing(self, tiny_model_config, device):
        """Test multi-layer routing through network."""
        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]
        n_layers = 3

        layers = nn.ModuleList([
            RoutedLayer(hidden_dim, n_heads) for _ in range(n_layers)
        ]).to(device)

        batch_size = 8
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.zeros(batch_size, hidden_dim, device=device)

        all_weights = []
        for layer in layers:
            x, routing_state, weights = layer(x, routing_state)
            all_weights.append(weights)

        # All routing weights should be valid
        for weights in all_weights:
            assert (weights >= 0).all()
            assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5)


class TestTrainingMetricsIntegration:
    """Tests for training metrics logging integration."""

    def test_log_full_training_cycle(self, small_p, device):
        """Test logging metrics through a training cycle."""
        dataset = ModularArithmeticDataset(p=small_p, operation="add", train_frac=0.3)

        model = BaselineMLP(
            input_dim=dataset.input_dim,
            hidden_dim=32,
            output_dim=dataset.output_dim,
            n_layers=2,
        ).to(device)

        loss_fn = LeastActionLoss(routing_regularizer=None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        history = MetricsHistory()

        train = dataset.get_train()
        test = dataset.get_test()
        train_inputs = train.inputs.to(device)
        train_targets = train.targets.to(device)
        test_inputs = test.inputs.to(device)
        test_targets = test.targets.to(device)

        # Train a few steps
        for step in range(5):
            # Forward
            logits = model(train_inputs)
            loss, loss_dict = loss_fn(logits, train_targets)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate
            with torch.no_grad():
                train_acc = (logits.argmax(dim=-1) == train_targets).float().mean().item()
                test_logits = model(test_inputs)
                test_loss = nn.functional.cross_entropy(test_logits, test_targets).item()
                test_acc = (test_logits.argmax(dim=-1) == test_targets).float().mean().item()

            # Log metrics
            metrics = TrainingMetrics(
                step=step,
                train_loss=loss_dict["task_loss"],
                train_acc=train_acc,
                test_loss=test_loss,
                test_acc=test_acc,
                routing_entropy=0.0,
                head_utilization=[],
            )
            history.log(metrics)

        # Verify history
        assert len(history.history) == 5
        df = history.get_dataframe()
        assert len(df) == 5
        assert "train_loss" in df.columns
        assert "test_acc" in df.columns


class TestCreateModelIntegration:
    """Tests for create_model factory integration."""

    def test_create_model_baseline(self, modular_dataset, tiny_model_config, device):
        """Test create_model creates baseline correctly."""
        model = create_model(
            model_type="baseline",
            input_dim=modular_dataset.input_dim,
            hidden_dim=tiny_model_config["hidden_dim"],
            output_dim=modular_dataset.output_dim,
            n_layers=tiny_model_config["n_layers"],
            n_heads=tiny_model_config["n_heads"],
        ).to(device)

        train = modular_dataset.get_train()
        inputs = train.inputs.to(device)
        logits = model(inputs)

        assert logits.shape == (len(inputs), modular_dataset.output_dim)

    def test_create_model_routed(self, modular_dataset, tiny_model_config, device):
        """Test create_model creates routed network correctly."""
        model = create_model(
            model_type="routed",
            input_dim=modular_dataset.input_dim,
            hidden_dim=tiny_model_config["hidden_dim"],
            output_dim=modular_dataset.output_dim,
            n_layers=tiny_model_config["n_layers"],
            n_heads=tiny_model_config["n_heads"],
        ).to(device)

        train = modular_dataset.get_train()
        inputs = train.inputs.to(device)
        logits, metrics = model(inputs)

        assert logits.shape == (len(inputs), modular_dataset.output_dim)
        assert len(metrics.layer_weights) == tiny_model_config["n_layers"]

    def test_create_model_transformer(self, sequence_dataset, tiny_model_config, device):
        """Test create_model creates transformer correctly."""
        model = create_model(
            model_type="transformer",
            input_dim=sequence_dataset.vocab_size,  # vocab_size for transformer
            hidden_dim=tiny_model_config["hidden_dim"],
            output_dim=sequence_dataset.output_dim,
            n_layers=tiny_model_config["n_layers"],
            n_heads=tiny_model_config["n_heads"],
        ).to(device)

        train = sequence_dataset.get_train()
        input_ids = train.input_ids.to(device)
        logits = model(input_ids)

        assert logits.shape == (len(input_ids), sequence_dataset.output_dim)
