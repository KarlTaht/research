#!/usr/bin/env python3
"""Functional tests for CustomTransformer.

Validates that:
1. Model can forward pass
2. Model can backward pass
3. Loss decreases over training steps (overfitting a small dataset)
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from common.models.custom_transfromer.wrapper import CustomTransformerWrapper


class TestCustomTransformerFunctional:
    """Functional tests for CustomTransformer with manual backprop."""

    @pytest.fixture
    def tiny_config(self):
        """Tiny model for fast testing."""
        return {
            'vocab_size': 100,
            'max_seq_len': 32,
            'n_blocks': 2,
            'n_heads': 2,
            'd_model': 32,
            'd_ffn': 64,
            'dtype': torch.float32,  # Use float32 for numerical stability in tests
        }

    @pytest.fixture
    def model(self, tiny_config):
        """Create tiny model."""
        return CustomTransformerWrapper(**tiny_config)

    def test_forward_pass(self, model, tiny_config):
        """Test forward pass returns correct shapes."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))

        outputs = model.forward(input_ids)

        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, tiny_config['vocab_size'])

    def test_forward_with_labels(self, model, tiny_config):
        """Test forward pass computes loss when labels provided."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))
        labels = torch.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))

        outputs = model.forward(input_ids, labels=labels)

        assert 'loss' in outputs
        assert outputs['loss'].item() > 0

    def test_train_step(self, model, tiny_config):
        """Test single training step completes."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))
        labels = torch.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))

        result = model.train_step(input_ids, labels, learning_rate=0.01)

        assert 'loss' in result
        assert result['loss'] > 0

    def test_loss_decreases_on_fixed_batch(self, model, tiny_config):
        """
        CRITICAL TEST: Loss should decrease when overfitting a single batch.

        This validates that the manual backprop is working correctly.
        """
        batch_size, seq_len = 4, 16

        # Fixed batch (same data every step)
        input_ids = torch.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))
        labels = input_ids.clone()  # Next token = same token (easy pattern)

        # Record initial loss
        initial_outputs = model.forward(input_ids, labels=labels)
        initial_loss = initial_outputs['loss'].item()

        # Train for multiple steps
        # Use conservative learning rate for stability
        losses = [initial_loss]
        for _ in range(100):
            result = model.train_step(input_ids, labels, learning_rate=0.01)
            losses.append(result['loss'])

        final_loss = losses[-1]

        # Check loss is not NaN
        assert not torch.isnan(torch.tensor(final_loss)), "Loss became NaN during training"

        # Assert loss decreased (should see noticeable improvement after 100 steps)
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

        print(f"Loss decreased: {initial_loss:.4f} -> {final_loss:.4f}")

    def test_no_nan_gradients(self, model, tiny_config):
        """Test that backward pass produces no NaN gradients."""
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, tiny_config['vocab_size'], (batch_size, seq_len))
        labels = input_ids.clone()

        # Run train step
        model.train_step(input_ids, labels, learning_rate=0.01)

        # Check gradients in cache
        for key, grad in model.model.cache.gradients.items():
            assert not torch.isnan(grad).any(), f"NaN gradient in {key}"
            assert not torch.isinf(grad).any(), f"Inf gradient in {key}"

    def test_generate(self, model, tiny_config):
        """Test text generation works."""
        input_ids = torch.randint(0, tiny_config['vocab_size'], (1, 5))

        generated = model.generate(
            input_ids,
            max_length=20,
            temperature=1.0,
            top_k=10,
        )

        assert generated.shape[0] == 1
        assert generated.shape[1] > input_ids.shape[1]
        assert generated.shape[1] <= 20

    def test_model_info(self, model):
        """Test get_model_info returns expected keys."""
        info = model.get_model_info()

        assert 'model_type' in info
        assert 'vocab_size' in info
        assert 'parameters' in info
        assert info['model_type'] == 'CustomTransformer'
        assert info['parameters'] > 0

    def test_checkpoint_save_load(self, model, tiny_config, tmp_path):
        """Test checkpoint save and load."""
        # Get initial output
        input_ids = torch.randint(0, tiny_config['vocab_size'], (1, 8))
        initial_outputs = model.forward(input_ids)
        initial_logits = initial_outputs['logits'].clone()

        # Save checkpoint
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        model.save_checkpoint(str(checkpoint_path), epoch=1)

        # Create new model and load checkpoint
        model2 = CustomTransformerWrapper(**tiny_config)
        model2.load_checkpoint(str(checkpoint_path))

        # Verify outputs match
        loaded_outputs = model2.forward(input_ids)
        loaded_logits = loaded_outputs['logits']

        assert torch.allclose(initial_logits, loaded_logits, atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
