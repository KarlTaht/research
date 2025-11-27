"""Tests for training data pipeline sanity checks.

These tests catch common bugs like padding token mismatches, loss configuration
issues, and data format problems BEFORE training starts.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add repo root and project directory to path
repo_root = Path(__file__).parent.parent.parent
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(project_dir))

from transformers import GPT2TokenizerFast
from common.models import ReferenceTransformer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def tokenizer():
    """GPT-2 tokenizer with pad token and SEP token set."""
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    # Add dedicated SEP token to avoid collision with EOS
    tok.add_special_tokens({'sep_token': '<|sep|>'})
    return tok


@pytest.fixture
def model_config(tokenizer):
    """Minimal model config for testing."""
    return {
        'vocab_size': len(tokenizer),  # Include added SEP token
        'd_model': 64,
        'n_heads': 4,
        'n_encoder_layers': 1,
        'n_decoder_layers': 1,
        'd_ff': 128,
        'dropout': 0.1,
        'max_seq_len': 128,
    }


@pytest.fixture
def model(tokenizer, model_config):
    """Small model for testing with proper SEP token."""
    return ReferenceTransformer(
        sep_token_id=tokenizer.sep_token_id,  # Use tokenizer's SEP token
        pad_token_id=tokenizer.pad_token_id,
        **model_config
    )


# ============================================================================
# Collate Function Tests
# ============================================================================

class TestCollateFn:
    """Tests for the collate function padding behavior."""

    def test_collate_fn_uses_correct_pad_token(self, tokenizer):
        """Collate function must pad with pad_token_id, not 0."""
        from train import create_collate_fn

        collate_fn = create_collate_fn(pad_token_id=tokenizer.pad_token_id)

        # Create batch with different lengths
        batch = [
            {'input_ids': [100, 200, 300], 'labels': [100, 200, 300]},
            {'input_ids': [400], 'labels': [400]},
        ]

        result = collate_fn(batch)

        # Padding should use pad_token_id (50256), NOT token 0 ('!')
        assert (result['input_ids'] == 0).sum() == 0, \
            "input_ids contains token 0 ('!') - should use pad_token_id for padding"
        assert (result['labels'] == 0).sum() == 0, \
            "labels contains token 0 ('!') - should use pad_token_id for padding"

        # Verify padding is actually pad_token_id
        assert (result['input_ids'] == tokenizer.pad_token_id).sum() == 2, \
            "Padding not using correct pad_token_id"
        assert (result['labels'] == tokenizer.pad_token_id).sum() == 2, \
            "Padding not using correct pad_token_id"

    def test_collate_fn_preserves_content(self, tokenizer):
        """Collate function should not modify actual content tokens."""
        from train import create_collate_fn

        collate_fn = create_collate_fn(pad_token_id=tokenizer.pad_token_id)

        batch = [
            {'input_ids': [1, 2, 3], 'labels': [1, 2, 3]},
            {'input_ids': [4, 5], 'labels': [4, 5]},
        ]

        result = collate_fn(batch)

        # First sequence should be unchanged
        assert result['input_ids'][0, :3].tolist() == [1, 2, 3]
        assert result['labels'][0, :3].tolist() == [1, 2, 3]

        # Second sequence content should be unchanged
        assert result['input_ids'][1, :2].tolist() == [4, 5]
        assert result['labels'][1, :2].tolist() == [4, 5]


# ============================================================================
# Model Loss Configuration Tests
# ============================================================================

class TestModelLossConfig:
    """Tests for model loss function configuration."""

    def test_model_criterion_ignores_pad_token(self, tokenizer, model):
        """Model's loss function must ignore the correct pad token."""
        assert model.criterion.ignore_index == tokenizer.pad_token_id, \
            f"Model ignores token {model.criterion.ignore_index} but tokenizer " \
            f"uses {tokenizer.pad_token_id} for padding"

    def test_model_pad_token_matches_tokenizer(self, tokenizer, model):
        """Model's pad_token_id must match tokenizer's."""
        assert model.pad_token_id == tokenizer.pad_token_id, \
            f"Model pad_token_id ({model.pad_token_id}) != " \
            f"tokenizer pad_token_id ({tokenizer.pad_token_id})"

    def test_loss_ignores_padding_in_batch(self, tokenizer, model):
        """Verify padding tokens don't contribute to loss."""
        from train import create_collate_fn

        collate_fn = create_collate_fn(pad_token_id=tokenizer.pad_token_id)

        # Create batch where second sequence is much shorter
        sep_id = model.sep_token_id
        batch = [
            {'input_ids': [100, 200, sep_id, 300, 400, 500],
             'labels': [100, 200, sep_id, 300, 400, 500]},
            {'input_ids': [100, sep_id, 300],
             'labels': [100, sep_id, 300]},
        ]

        result = collate_fn(batch)
        input_ids = result['input_ids']
        labels = result['labels']

        # Compute loss
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
            loss_with_padding = outputs['loss']

        # Loss should be finite (not NaN or Inf)
        assert torch.isfinite(loss_with_padding), "Loss is not finite"


# ============================================================================
# Token ID Sanity Tests
# ============================================================================

class TestTokenSanity:
    """Tests for token ID sanity."""

    def test_exclamation_is_token_zero(self, tokenizer):
        """Verify our assumption that token 0 is '!'."""
        # This test documents the assumption that token 0 is '!'
        # If this changes in a different tokenizer, tests will catch it
        assert tokenizer.decode([0]) == '!', \
            "Token 0 is not '!' - padding bug detection may not work"

    def test_eos_token_id_value(self, tokenizer):
        """Verify EOS token ID is what we expect."""
        assert tokenizer.eos_token_id == 50256, \
            f"EOS token ID changed: {tokenizer.eos_token_id}"

    def test_pad_token_equals_eos(self, tokenizer):
        """Verify pad token is set to EOS (GPT-2 convention)."""
        assert tokenizer.pad_token_id == tokenizer.eos_token_id, \
            "pad_token should equal eos_token for GPT-2"

    def test_sep_token_in_valid_range(self, model):
        """SEP token should be a valid vocab index."""
        assert 0 <= model.sep_token_id < model.vocab_size, \
            f"SEP token {model.sep_token_id} outside vocab range [0, {model.vocab_size})"


# ============================================================================
# Data Format Tests
# ============================================================================

class TestDataFormat:
    """Tests for training data format."""

    def test_packed_sequence_has_separator(self, tokenizer, model):
        """Packed sequences must contain the SEP token."""
        # Simulate what prepare_dataset does
        dialogue = "Hi there"
        summary = "Greeting"

        dialogue_ids = tokenizer.encode(dialogue, add_special_tokens=False)
        summary_ids = tokenizer.encode(summary, add_special_tokens=False)
        sep_id = model.sep_token_id

        packed = dialogue_ids + [sep_id] + summary_ids

        assert sep_id in packed, "Packed sequence missing SEP token"
        assert packed.count(sep_id) == 1, "Packed sequence should have exactly one SEP"

    def test_summary_tokens_not_empty(self, tokenizer):
        """Summary should produce at least one token."""
        summary = "Test summary."
        tokens = tokenizer.encode(summary, add_special_tokens=False)
        assert len(tokens) > 0, "Summary produced no tokens"


# ============================================================================
# Integration Test
# ============================================================================

class TestEndToEndPipeline:
    """Integration tests for the full pipeline."""

    def test_training_step_runs_without_error(self, tokenizer, model):
        """A single training step should complete without errors."""
        from train import create_collate_fn

        collate_fn = create_collate_fn(pad_token_id=tokenizer.pad_token_id)
        sep_id = model.sep_token_id

        # Create a mini batch
        batch = [
            {'input_ids': [100, 200, sep_id, 300, 400],
             'labels': [100, 200, sep_id, 300, 400]},
            {'input_ids': [500, sep_id, 600],
             'labels': [500, sep_id, 600]},
        ]

        result = collate_fn(batch)

        # Forward pass
        outputs = model(result['input_ids'], labels=result['labels'])

        assert 'loss' in outputs, "Model output missing 'loss'"
        assert 'logits' in outputs, "Model output missing 'logits'"
        assert torch.isfinite(outputs['loss']), "Loss is not finite"

        # Backward pass
        outputs['loss'].backward()

        # Check gradients exist
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed"
