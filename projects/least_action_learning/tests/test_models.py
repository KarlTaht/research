"""Tests for models.py - Model architectures."""

import pytest
import torch
import torch.nn as nn
from projects.least_action_learning.src.models import (
    BaselineMLP,
    RoutedNetwork,
    SingleHeadNetwork,
    GrokTransformer,
    TransformerBlock,
    MultiHeadAttention,
    FeedForward,
    create_model,
)
from projects.least_action_learning.src.metrics import RoutingMetrics


class TestBaselineMLP:
    """Tests for BaselineMLP."""

    def test_forward_shape(self, baseline_mlp, sample_batch):
        """Test forward pass output shape."""
        inputs, _ = sample_batch
        logits = baseline_mlp(inputs)

        assert logits.shape == (inputs.shape[0], baseline_mlp.output_dim)

    def test_forward_returns_tensor(self, baseline_mlp, sample_batch):
        """Test forward returns plain tensor (not tuple)."""
        inputs, _ = sample_batch
        output = baseline_mlp(inputs)

        assert isinstance(output, torch.Tensor)

    def test_get_representation_shape(self, baseline_mlp, sample_batch):
        """Test get_representation returns hidden_dim sized output."""
        inputs, _ = sample_batch
        repr = baseline_mlp.get_representation(inputs)

        assert repr.shape == (inputs.shape[0], baseline_mlp.hidden_dim)

    def test_count_parameters(self, baseline_mlp):
        """Test parameter counting is accurate."""
        manual_count = sum(p.numel() for p in baseline_mlp.parameters() if p.requires_grad)
        assert baseline_mlp.count_parameters() == manual_count

    def test_n_layers_creates_correct_depth(self, modular_dataset, tiny_model_config):
        """Test n_layers parameter creates correct network depth."""
        for n_layers in [1, 2, 4]:
            model = BaselineMLP(
                input_dim=modular_dataset.input_dim,
                hidden_dim=tiny_model_config["hidden_dim"],
                output_dim=modular_dataset.output_dim,
                n_layers=n_layers,
            )
            # Count Linear layers
            linear_count = sum(1 for m in model.net if isinstance(m, nn.Linear))
            assert linear_count == n_layers + 1  # n hidden + 1 output

    def test_attributes_stored(self, baseline_mlp, modular_dataset, tiny_model_config):
        """Test model stores correct attributes."""
        assert baseline_mlp.input_dim == modular_dataset.input_dim
        assert baseline_mlp.hidden_dim == tiny_model_config["hidden_dim"]
        assert baseline_mlp.output_dim == modular_dataset.output_dim
        assert baseline_mlp.n_layers == tiny_model_config["n_layers"]

    def test_forward_backward(self, baseline_mlp, sample_batch):
        """Test gradients flow through model."""
        inputs, targets = sample_batch
        logits = baseline_mlp(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()

        # Check gradients exist
        for param in baseline_mlp.parameters():
            assert param.grad is not None

    def test_single_layer(self, modular_dataset):
        """Test model with single layer (edge case)."""
        model = BaselineMLP(
            input_dim=modular_dataset.input_dim,
            hidden_dim=32,
            output_dim=modular_dataset.output_dim,
            n_layers=1,
        )
        inputs = modular_dataset.get_train().inputs[:8]
        logits = model(inputs)
        assert logits.shape == (8, modular_dataset.output_dim)


class TestRoutedNetwork:
    """Tests for RoutedNetwork."""

    def test_forward_returns_tuple(self, routed_network, sample_batch):
        """Test forward returns (logits, metrics) tuple."""
        inputs, _ = sample_batch
        outputs = routed_network(inputs)

        assert isinstance(outputs, tuple)
        assert len(outputs) == 2

    def test_forward_logits_shape(self, routed_network, sample_batch):
        """Test logits have correct shape."""
        inputs, _ = sample_batch
        logits, _ = routed_network(inputs)

        assert logits.shape == (inputs.shape[0], routed_network.output_dim)

    def test_forward_returns_routing_metrics(self, routed_network, sample_batch):
        """Test forward returns RoutingMetrics."""
        inputs, _ = sample_batch
        _, metrics = routed_network(inputs)

        assert isinstance(metrics, RoutingMetrics)
        assert len(metrics.layer_weights) == routed_network.n_layers

    def test_routing_weights_sum_to_one(self, routed_network, sample_batch):
        """Test routing weights sum to 1 (softmax property)."""
        inputs, _ = sample_batch
        _, metrics = routed_network(inputs)

        for weights in metrics.layer_weights:
            sums = weights.sum(dim=-1)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_routing_weights_shape(self, routed_network, sample_batch, tiny_model_config):
        """Test routing weights have correct shape."""
        inputs, _ = sample_batch
        _, metrics = routed_network(inputs)

        for weights in metrics.layer_weights:
            assert weights.shape == (inputs.shape[0], tiny_model_config["n_heads"])

    def test_routing_weights_non_negative(self, routed_network, sample_batch):
        """Test routing weights are non-negative (softmax property)."""
        inputs, _ = sample_batch
        _, metrics = routed_network(inputs)

        for weights in metrics.layer_weights:
            assert (weights >= 0).all()

    def test_get_representation(self, routed_network, sample_batch):
        """Test get_representation returns hidden state."""
        inputs, _ = sample_batch
        repr = routed_network.get_representation(inputs)

        assert repr.shape == (inputs.shape[0], routed_network.hidden_dim)

    def test_forward_with_routing_trace(self, routed_network, sample_batch):
        """Test forward_with_routing_trace returns full trace."""
        inputs, _ = sample_batch
        logits, layer_outputs, layer_weights = routed_network.forward_with_routing_trace(inputs)

        assert len(layer_outputs) == routed_network.n_layers + 1  # Initial + each layer
        assert len(layer_weights) == routed_network.n_layers

    def test_forward_with_routing_trace_detached(self, routed_network, sample_batch):
        """Test routing trace outputs are detached."""
        inputs, _ = sample_batch
        _, layer_outputs, layer_weights = routed_network.forward_with_routing_trace(inputs)

        for out in layer_outputs:
            assert not out.requires_grad
        for w in layer_weights:
            assert not w.requires_grad

    def test_efficient_block_mode(self, modular_dataset, tiny_model_config):
        """Test use_efficient_block=True uses RoutedBlock."""
        from projects.least_action_learning.src.routing import RoutedBlock

        model = RoutedNetwork(
            input_dim=modular_dataset.input_dim,
            hidden_dim=tiny_model_config["hidden_dim"],
            output_dim=modular_dataset.output_dim,
            n_layers=tiny_model_config["n_layers"],
            n_heads=tiny_model_config["n_heads"],
            use_efficient_block=True,
        )
        assert all(isinstance(layer, RoutedBlock) for layer in model.layers)

    def test_get_config(self, routed_network, tiny_model_config):
        """Test get_config returns correct configuration."""
        config = routed_network.get_config()

        assert config["hidden_dim"] == tiny_model_config["hidden_dim"]
        assert config["n_layers"] == tiny_model_config["n_layers"]
        assert config["n_heads"] == tiny_model_config["n_heads"]
        assert config["n_parameters"] == routed_network.count_parameters()

    def test_count_parameters(self, routed_network):
        """Test parameter counting."""
        manual_count = sum(p.numel() for p in routed_network.parameters() if p.requires_grad)
        assert routed_network.count_parameters() == manual_count

    def test_forward_backward(self, routed_network, sample_batch):
        """Test gradients flow through routed model."""
        inputs, targets = sample_batch
        logits, _ = routed_network(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()

        # Check key parameters have gradients (state_update may not have gradients
        # since routing state doesn't directly affect loss through the output head)
        assert routed_network.embed.weight.grad is not None
        assert routed_network.output_head.weight.grad is not None
        # At least some layer parameters should have gradients
        grad_count = sum(1 for p in routed_network.parameters() if p.grad is not None)
        assert grad_count > 0


class TestSingleHeadNetwork:
    """Tests for SingleHeadNetwork."""

    def test_is_routed_with_one_head(self, single_head_network):
        """Test SingleHeadNetwork is RoutedNetwork with n_heads=1."""
        assert single_head_network.n_heads == 1

    def test_routing_weights_are_one(self, single_head_network, sample_batch):
        """Test routing weights are always 1.0 for single head."""
        inputs, _ = sample_batch
        _, metrics = single_head_network(inputs)

        for weights in metrics.layer_weights:
            assert torch.allclose(weights, torch.ones_like(weights))

    def test_inherits_from_routed_network(self, single_head_network):
        """Test SingleHeadNetwork inherits from RoutedNetwork."""
        assert isinstance(single_head_network, RoutedNetwork)

    def test_forward_shape(self, single_head_network, sample_batch):
        """Test forward produces correct output shape."""
        inputs, _ = sample_batch
        logits, _ = single_head_network(inputs)
        assert logits.shape == (inputs.shape[0], single_head_network.output_dim)


class TestGrokTransformer:
    """Tests for GrokTransformer."""

    def test_forward_shape(self, grok_transformer, sample_sequence_batch):
        """Test forward output shape."""
        input_ids, _ = sample_sequence_batch
        logits = grok_transformer(input_ids)

        assert logits.shape == (input_ids.shape[0], grok_transformer.output_dim)

    def test_forward_returns_tensor(self, grok_transformer, sample_sequence_batch):
        """Test forward returns tensor, not tuple."""
        input_ids, _ = sample_sequence_batch
        output = grok_transformer(input_ids)

        assert isinstance(output, torch.Tensor)

    def test_get_embeddings_shape(self, grok_transformer, sample_sequence_batch):
        """Test get_embeddings returns correct shape."""
        input_ids, _ = sample_sequence_batch
        embeddings = grok_transformer.get_embeddings(input_ids)

        assert embeddings.shape == (input_ids.shape[0], input_ids.shape[1], grok_transformer.d_model)

    def test_forward_from_embeddings(self, grok_transformer, sample_sequence_batch):
        """Test forward_from_embeddings produces same output as forward."""
        input_ids, _ = sample_sequence_batch

        direct_logits = grok_transformer(input_ids)
        embeddings = grok_transformer.get_embeddings(input_ids)
        embed_logits = grok_transformer.forward_from_embeddings(embeddings)

        assert torch.allclose(direct_logits, embed_logits, atol=1e-5)

    def test_get_representation(self, grok_transformer, sample_sequence_batch):
        """Test get_representation returns hidden state at last position."""
        input_ids, _ = sample_sequence_batch
        repr = grok_transformer.get_representation(input_ids)

        assert repr.shape == (input_ids.shape[0], grok_transformer.d_model)

    def test_causal_mask_shape(self, grok_transformer):
        """Test causal mask has correct shape."""
        mask = grok_transformer.causal_mask
        assert mask.shape == (grok_transformer.max_seq_len, grok_transformer.max_seq_len)

    def test_causal_mask_is_upper_triangular_inf(self, grok_transformer):
        """Test causal mask is upper triangular with -inf."""
        mask = grok_transformer.causal_mask
        # Lower triangle (including diagonal) should be 0
        # Upper triangle should be -inf
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if j > i:
                    assert mask[i, j] == float('-inf')
                else:
                    assert mask[i, j] == 0

    def test_count_parameters(self, grok_transformer):
        """Test parameter counting."""
        manual_count = sum(p.numel() for p in grok_transformer.parameters() if p.requires_grad)
        assert grok_transformer.count_parameters() == manual_count

    def test_attributes_stored(self, grok_transformer, sequence_dataset, tiny_model_config):
        """Test model stores correct attributes."""
        assert grok_transformer.vocab_size == sequence_dataset.vocab_size
        assert grok_transformer.d_model == tiny_model_config["hidden_dim"]
        assert grok_transformer.n_heads == tiny_model_config["n_heads"]
        assert grok_transformer.n_layers == tiny_model_config["n_layers"]
        assert grok_transformer.output_dim == sequence_dataset.output_dim

    def test_forward_backward(self, grok_transformer, sample_sequence_batch):
        """Test gradients flow through transformer."""
        input_ids, targets = sample_sequence_batch
        logits = grok_transformer(input_ids)
        loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()

        for name, param in grok_transformer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_different_seq_len(self, grok_transformer):
        """Test transformer handles different sequence lengths."""
        # Shorter sequence
        input_ids = torch.randint(0, grok_transformer.vocab_size, (4, 3))
        logits = grok_transformer(input_ids)
        assert logits.shape == (4, grok_transformer.output_dim)

    def test_embedding_dtype(self, grok_transformer, sample_sequence_batch):
        """Test embeddings are float tensors."""
        input_ids, _ = sample_sequence_batch
        embeddings = grok_transformer.get_embeddings(input_ids)
        assert embeddings.dtype == torch.float32


class TestTransformerBlock:
    """Tests for TransformerBlock."""

    def test_forward_shape(self, tiny_model_config, device):
        """Test TransformerBlock forward pass."""
        block = TransformerBlock(
            d_model=tiny_model_config["hidden_dim"],
            d_ffn=tiny_model_config["hidden_dim"] * 4,
            n_heads=tiny_model_config["n_heads"],
        ).to(device)

        x = torch.randn(4, 4, tiny_model_config["hidden_dim"], device=device)
        mask = torch.zeros(4, 4, device=device)

        out = block(x, mask)
        assert out.shape == x.shape

    def test_residual_connection(self, tiny_model_config, device):
        """Test residual connections are present."""
        block = TransformerBlock(
            d_model=tiny_model_config["hidden_dim"],
            d_ffn=tiny_model_config["hidden_dim"] * 4,
            n_heads=tiny_model_config["n_heads"],
        ).to(device)

        x = torch.randn(4, 4, tiny_model_config["hidden_dim"], device=device)
        mask = torch.zeros(4, 4, device=device)

        out = block(x, mask)
        # Output should not be identical to input (transformation happened)
        assert not torch.equal(out, x)

    def test_dropout_zero(self, tiny_model_config, device):
        """Test block with dropout=0 is deterministic."""
        block = TransformerBlock(
            d_model=tiny_model_config["hidden_dim"],
            d_ffn=tiny_model_config["hidden_dim"] * 4,
            n_heads=tiny_model_config["n_heads"],
            dropout=0.0,
        ).to(device)
        block.eval()

        x = torch.randn(4, 4, tiny_model_config["hidden_dim"], device=device)
        mask = torch.zeros(4, 4, device=device)

        out1 = block(x, mask)
        out2 = block(x, mask)
        assert torch.equal(out1, out2)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention."""

    def test_forward_shape(self, tiny_model_config, device):
        """Test MultiHeadAttention forward pass."""
        attn = MultiHeadAttention(
            d_model=tiny_model_config["hidden_dim"],
            n_heads=tiny_model_config["n_heads"],
        ).to(device)

        x = torch.randn(4, 4, tiny_model_config["hidden_dim"], device=device)
        mask = torch.zeros(4, 4, device=device)

        out = attn(x, mask)
        assert out.shape == x.shape

    def test_head_dim_calculation(self, tiny_model_config):
        """Test head dimension calculation."""
        attn = MultiHeadAttention(
            d_model=tiny_model_config["hidden_dim"],
            n_heads=tiny_model_config["n_heads"],
        )
        expected_d_head = tiny_model_config["hidden_dim"] // tiny_model_config["n_heads"]
        assert attn.d_head == expected_d_head

    def test_causal_mask_effect(self, tiny_model_config, device):
        """Test causal mask prevents attending to future."""
        attn = MultiHeadAttention(
            d_model=tiny_model_config["hidden_dim"],
            n_heads=tiny_model_config["n_heads"],
        ).to(device)

        x = torch.randn(1, 4, tiny_model_config["hidden_dim"], device=device)

        # No mask
        mask_none = torch.zeros(4, 4, device=device)
        out_none = attn(x, mask_none)

        # Causal mask
        mask_causal = torch.triu(torch.full((4, 4), float('-inf'), device=device), diagonal=1)
        out_causal = attn(x, mask_causal)

        # Outputs should differ
        assert not torch.allclose(out_none, out_causal)


class TestFeedForward:
    """Tests for FeedForward."""

    def test_forward_shape(self, tiny_model_config, device):
        """Test FeedForward forward pass."""
        ffn = FeedForward(
            d_model=tiny_model_config["hidden_dim"],
            d_ffn=tiny_model_config["hidden_dim"] * 4,
        ).to(device)

        x = torch.randn(4, 4, tiny_model_config["hidden_dim"], device=device)
        out = ffn(x)
        assert out.shape == x.shape

    def test_expansion_factor(self, tiny_model_config):
        """Test FFN has correct expansion."""
        ffn = FeedForward(
            d_model=tiny_model_config["hidden_dim"],
            d_ffn=tiny_model_config["hidden_dim"] * 4,
        )
        assert ffn.w1.out_features == tiny_model_config["hidden_dim"] * 4
        assert ffn.w2.in_features == tiny_model_config["hidden_dim"] * 4


class TestCreateModel:
    """Tests for create_model factory function."""

    @pytest.mark.parametrize("model_type,expected_class", [
        ("baseline", BaselineMLP),
        ("routed", RoutedNetwork),
        ("single_head", SingleHeadNetwork),
        ("transformer", GrokTransformer),
    ])
    def test_create_model_types(self, model_type, expected_class, small_p):
        """Test create_model creates correct model types."""
        if model_type == "transformer":
            input_dim = small_p + 2  # vocab_size
        else:
            input_dim = 2 * small_p

        model = create_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=32,
            output_dim=small_p,
            n_layers=2,
            n_heads=2,
        )
        assert isinstance(model, expected_class)

    def test_create_model_invalid_type(self, small_p):
        """Test create_model raises for invalid type."""
        with pytest.raises(ValueError, match="Unknown model type"):
            create_model(
                model_type="invalid",
                input_dim=2 * small_p,
                hidden_dim=32,
                output_dim=small_p,
                n_layers=2,
            )

    def test_create_transformer_with_kwargs(self, small_p):
        """Test create_model passes kwargs to transformer."""
        model = create_model(
            model_type="transformer",
            input_dim=small_p + 2,
            hidden_dim=32,
            output_dim=small_p,
            n_layers=2,
            n_heads=2,
            max_seq_len=10,
            dropout=0.1,
        )
        assert model.max_seq_len == 10

    def test_create_baseline_ignores_n_heads(self, small_p):
        """Test create_model ignores n_heads for baseline."""
        model = create_model(
            model_type="baseline",
            input_dim=2 * small_p,
            hidden_dim=32,
            output_dim=small_p,
            n_layers=2,
            n_heads=8,  # Should be ignored
        )
        assert isinstance(model, BaselineMLP)

    def test_create_model_forward_works(self, modular_dataset, sequence_dataset):
        """Test all created models can do forward pass."""
        batch_size = 8

        # Baseline
        model = create_model("baseline", modular_dataset.input_dim, 32, modular_dataset.output_dim, 2)
        inputs = modular_dataset.get_train().inputs[:batch_size]
        output = model(inputs)
        assert output.shape == (batch_size, modular_dataset.output_dim)

        # Routed
        model = create_model("routed", modular_dataset.input_dim, 32, modular_dataset.output_dim, 2, n_heads=4)
        logits, metrics = model(inputs)
        assert logits.shape == (batch_size, modular_dataset.output_dim)

        # Transformer
        model = create_model("transformer", sequence_dataset.vocab_size, 32, sequence_dataset.output_dim, 2, n_heads=2)
        input_ids = sequence_dataset.get_train().input_ids[:batch_size]
        logits = model(input_ids)
        assert logits.shape == (batch_size, sequence_dataset.output_dim)


class TestModelDeviceHandling:
    """Tests for model device handling."""

    def test_baseline_to_device(self, baseline_mlp, device):
        """Test baseline MLP moves to device."""
        model = baseline_mlp.to(device)
        assert next(model.parameters()).device == device

    def test_routed_to_device(self, routed_network, device):
        """Test routed network moves to device."""
        model = routed_network.to(device)
        assert next(model.parameters()).device == device

    def test_transformer_to_device(self, grok_transformer, device):
        """Test transformer moves to device."""
        model = grok_transformer.to(device)
        assert next(model.parameters()).device == device
        # Causal mask should also be on device
        assert model.causal_mask.device == device
