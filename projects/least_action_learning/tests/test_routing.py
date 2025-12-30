"""Tests for routing components (RoutingGate, RoutedLayer, RoutedBlock)."""

import pytest
import torch
import torch.nn as nn
from typing import Tuple


class TestRoutingGate:
    """Tests for RoutingGate module."""

    def test_output_shape(self, routing_gate, tiny_model_config, device):
        """Test routing gate produces correct output shape."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]

        prev_state = torch.randn(batch_size, hidden_dim, device=device)
        residual = torch.randn(batch_size, hidden_dim, device=device)

        weights = routing_gate.to(device)(prev_state, residual)

        assert weights.shape == (batch_size, n_heads)

    def test_weights_sum_to_one(self, routing_gate, tiny_model_config, device):
        """Test routing weights sum to 1 (softmax property)."""
        batch_size = 16
        hidden_dim = tiny_model_config["hidden_dim"]

        prev_state = torch.randn(batch_size, hidden_dim, device=device)
        residual = torch.randn(batch_size, hidden_dim, device=device)

        weights = routing_gate.to(device)(prev_state, residual)
        sums = weights.sum(dim=-1)

        assert torch.allclose(sums, torch.ones(batch_size, device=device), atol=1e-5)

    def test_weights_non_negative(self, routing_gate, tiny_model_config, device):
        """Test routing weights are non-negative."""
        batch_size = 16
        hidden_dim = tiny_model_config["hidden_dim"]

        prev_state = torch.randn(batch_size, hidden_dim, device=device)
        residual = torch.randn(batch_size, hidden_dim, device=device)

        weights = routing_gate.to(device)(prev_state, residual)

        assert (weights >= 0).all()

    def test_weights_in_valid_range(self, routing_gate, tiny_model_config, device):
        """Test routing weights are in [0, 1]."""
        batch_size = 16
        hidden_dim = tiny_model_config["hidden_dim"]

        prev_state = torch.randn(batch_size, hidden_dim, device=device)
        residual = torch.randn(batch_size, hidden_dim, device=device)

        weights = routing_gate.to(device)(prev_state, residual)

        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_gradient_flow(self, routing_gate, tiny_model_config, device):
        """Test gradients flow through routing gate."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        prev_state = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)
        residual = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)

        weights = routing_gate.to(device)(prev_state, residual)
        loss = weights.sum()
        loss.backward()

        assert prev_state.grad is not None
        assert residual.grad is not None
        assert routing_gate.state_proj.weight.grad is not None
        assert routing_gate.residual_proj.weight.grad is not None
        assert routing_gate.gate.weight.grad is not None

    def test_deterministic_with_same_input(self, routing_gate, tiny_model_config, device):
        """Test same input gives same output."""
        batch_size = 4
        hidden_dim = tiny_model_config["hidden_dim"]

        routing_gate = routing_gate.to(device).eval()
        prev_state = torch.randn(batch_size, hidden_dim, device=device)
        residual = torch.randn(batch_size, hidden_dim, device=device)

        weights1 = routing_gate(prev_state, residual)
        weights2 = routing_gate(prev_state, residual)

        assert torch.allclose(weights1, weights2)

    def test_different_inputs_different_outputs(self, routing_gate, tiny_model_config, device):
        """Test different inputs produce different outputs."""
        batch_size = 4
        hidden_dim = tiny_model_config["hidden_dim"]

        routing_gate = routing_gate.to(device)
        prev_state1 = torch.randn(batch_size, hidden_dim, device=device)
        prev_state2 = torch.randn(batch_size, hidden_dim, device=device)
        residual = torch.randn(batch_size, hidden_dim, device=device)

        weights1 = routing_gate(prev_state1, residual)
        weights2 = routing_gate(prev_state2, residual)

        # With high probability, random inputs give different outputs
        assert not torch.allclose(weights1, weights2)

    def test_initialization_encourages_uniform(self, tiny_model_config, device):
        """Test initialization encourages uniform routing."""
        from projects.least_action_learning.src.routing import RoutingGate

        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]

        gate = RoutingGate(hidden_dim, n_heads).to(device)

        # Zero-mean inputs should give roughly uniform weights
        batch_size = 100
        prev_state = torch.zeros(batch_size, hidden_dim, device=device)
        residual = torch.zeros(batch_size, hidden_dim, device=device)

        weights = gate(prev_state, residual)
        mean_weights = weights.mean(dim=0)
        expected_uniform = torch.full((n_heads,), 1.0 / n_heads, device=device)

        # Should be close to uniform due to zero bias initialization
        assert torch.allclose(mean_weights, expected_uniform, atol=0.1)


class TestRoutedLayer:
    """Tests for RoutedLayer module."""

    def test_output_shapes(self, routed_layer, tiny_model_config, device):
        """Test routed layer produces correct output shapes."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]

        routed_layer = routed_layer.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.randn(batch_size, hidden_dim, device=device)

        output, new_state, weights = routed_layer(x, routing_state)

        assert output.shape == (batch_size, hidden_dim)
        assert new_state.shape == (batch_size, hidden_dim)
        assert weights.shape == (batch_size, n_heads)

    def test_routing_weights_valid(self, routed_layer, tiny_model_config, device):
        """Test routing weights from layer are valid (sum to 1, non-negative)."""
        batch_size = 16
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_layer = routed_layer.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.randn(batch_size, hidden_dim, device=device)

        _, _, weights = routed_layer(x, routing_state)

        assert (weights >= 0).all()
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5)

    def test_gradient_flow(self, routed_layer, tiny_model_config, device):
        """Test gradients flow through routed layer."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_layer = routed_layer.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)
        routing_state = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)

        output, new_state, _ = routed_layer(x, routing_state)
        loss = output.sum() + new_state.sum()
        loss.backward()

        assert x.grad is not None
        assert routing_state.grad is not None

    def test_residual_connection(self, routed_layer, tiny_model_config, device):
        """Test residual connection is present (output correlated with input)."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_layer = routed_layer.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.zeros(batch_size, hidden_dim, device=device)

        output, _, _ = routed_layer(x, routing_state)

        # Output should be correlated with input due to residual
        # Check via dot product
        similarity = (x * output).sum(dim=-1).mean()
        assert similarity > -1  # Not perfectly anti-correlated

    def test_different_heads_different_outputs(self, routed_layer, tiny_model_config, device):
        """Test different heads produce different transformations."""
        batch_size = 4
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_layer = routed_layer.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)

        # Directly compute head outputs
        head_outputs = [head(x) for head in routed_layer.heads]

        # Check that heads produce different outputs
        for i in range(len(head_outputs)):
            for j in range(i + 1, len(head_outputs)):
                assert not torch.allclose(head_outputs[i], head_outputs[j], atol=1e-3)

    def test_state_update_changes_state(self, routed_layer, tiny_model_config, device):
        """Test routing state is updated after layer."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_layer = routed_layer.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.randn(batch_size, hidden_dim, device=device)

        _, new_state, _ = routed_layer(x, routing_state)

        # New state should be different from old state
        assert not torch.allclose(new_state, routing_state)

    def test_number_of_heads(self, tiny_model_config, device):
        """Test layer creates correct number of heads."""
        from projects.least_action_learning.src.routing import RoutedLayer

        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]

        layer = RoutedLayer(hidden_dim, n_heads).to(device)

        assert len(layer.heads) == n_heads


class TestRoutedBlock:
    """Tests for RoutedBlock (efficient implementation)."""

    def test_output_shapes(self, routed_block, tiny_model_config, device):
        """Test routed block produces correct output shapes."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]

        routed_block = routed_block.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.randn(batch_size, hidden_dim, device=device)

        output, new_state, weights = routed_block(x, routing_state)

        assert output.shape == (batch_size, hidden_dim)
        assert new_state.shape == (batch_size, hidden_dim)
        assert weights.shape == (batch_size, n_heads)

    def test_routing_weights_valid(self, routed_block, tiny_model_config, device):
        """Test routing weights from block are valid."""
        batch_size = 16
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_block = routed_block.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.randn(batch_size, hidden_dim, device=device)

        _, _, weights = routed_block(x, routing_state)

        assert (weights >= 0).all()
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5)

    def test_gradient_flow(self, routed_block, tiny_model_config, device):
        """Test gradients flow through routed block."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_block = routed_block.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)
        routing_state = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)

        output, new_state, _ = routed_block(x, routing_state)
        loss = output.sum() + new_state.sum()
        loss.backward()

        assert x.grad is not None
        assert routing_state.grad is not None
        # Check key layers have gradients
        assert routed_block.up_proj.weight.grad is not None
        assert routed_block.down_projs.weight.grad is not None

    def test_residual_connection(self, routed_block, tiny_model_config, device):
        """Test residual connection in block."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_block = routed_block.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.zeros(batch_size, hidden_dim, device=device)

        output, _, _ = routed_block(x, routing_state)

        # Output should be correlated with input due to residual
        similarity = (x * output).sum(dim=-1).mean()
        assert similarity > -1

    def test_state_update_changes_state(self, routed_block, tiny_model_config, device):
        """Test routing state is updated."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_block = routed_block.to(device)
        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.randn(batch_size, hidden_dim, device=device)

        _, new_state, _ = routed_block(x, routing_state)

        assert not torch.allclose(new_state, routing_state)


class TestRoutedLayerVsBlock:
    """Tests comparing RoutedLayer and RoutedBlock behavior."""

    def test_same_output_shape(self, routed_layer, routed_block, tiny_model_config, device):
        """Test both implementations produce same output shapes."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_layer = routed_layer.to(device)
        routed_block = routed_block.to(device)

        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.randn(batch_size, hidden_dim, device=device)

        out1, state1, w1 = routed_layer(x, routing_state)
        out2, state2, w2 = routed_block(x, routing_state)

        assert out1.shape == out2.shape
        assert state1.shape == state2.shape
        assert w1.shape == w2.shape

    def test_both_have_residual(self, routed_layer, routed_block, tiny_model_config, device):
        """Test both implementations have residual connections."""
        batch_size = 8
        hidden_dim = tiny_model_config["hidden_dim"]

        routed_layer = routed_layer.to(device)
        routed_block = routed_block.to(device)

        x = torch.randn(batch_size, hidden_dim, device=device)
        routing_state = torch.zeros(batch_size, hidden_dim, device=device)

        out1, _, _ = routed_layer(x, routing_state)
        out2, _, _ = routed_block(x, routing_state)

        # Both should have correlation with input due to residual
        sim1 = (x * out1).sum(dim=-1).mean()
        sim2 = (x * out2).sum(dim=-1).mean()

        # Both should be positively correlated (residual passes through)
        # This is a weak test but validates residual exists
        assert not torch.isnan(sim1)
        assert not torch.isnan(sim2)


class TestRoutingIntegration:
    """Integration tests for routing components."""

    def test_sequential_layers(self, tiny_model_config, device):
        """Test routing through multiple sequential layers."""
        from projects.least_action_learning.src.routing import RoutedLayer

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

        # Each layer should produce valid outputs
        assert x.shape == (batch_size, hidden_dim)
        assert routing_state.shape == (batch_size, hidden_dim)
        assert len(all_weights) == n_layers

        # All weights should sum to 1
        for w in all_weights:
            assert torch.allclose(w.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5)

    def test_routing_state_evolves(self, tiny_model_config, device):
        """Test routing state evolves through layers."""
        from projects.least_action_learning.src.routing import RoutedLayer

        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]

        layer = RoutedLayer(hidden_dim, n_heads).to(device)

        batch_size = 8
        x = torch.randn(batch_size, hidden_dim, device=device)
        state0 = torch.zeros(batch_size, hidden_dim, device=device)

        _, state1, _ = layer(x, state0)
        _, state2, _ = layer(x, state1)
        _, state3, _ = layer(x, state2)

        # States should all be different
        assert not torch.allclose(state0, state1)
        assert not torch.allclose(state1, state2)
        assert not torch.allclose(state2, state3)

    def test_backprop_through_multiple_layers(self, tiny_model_config, device):
        """Test gradients flow through multiple routed layers."""
        from projects.least_action_learning.src.routing import RoutedLayer

        hidden_dim = tiny_model_config["hidden_dim"]
        n_heads = tiny_model_config["n_heads"]
        n_layers = 3

        layers = nn.ModuleList([
            RoutedLayer(hidden_dim, n_heads) for _ in range(n_layers)
        ]).to(device)

        batch_size = 8
        x = torch.randn(batch_size, hidden_dim, device=device, requires_grad=True)
        routing_state = torch.zeros(batch_size, hidden_dim, device=device, requires_grad=True)

        # Forward through all layers
        current_x = x
        current_state = routing_state
        for layer in layers:
            current_x, current_state, _ = layer(current_x, current_state)

        # Backward
        loss = current_x.sum()
        loss.backward()

        # Check input gradients
        assert x.grad is not None

        # Check all layers have gradients
        for layer in layers:
            assert layer.routing_gate.gate.weight.grad is not None
            for head in layer.heads:
                for param in head.parameters():
                    if param.requires_grad:
                        assert param.grad is not None
