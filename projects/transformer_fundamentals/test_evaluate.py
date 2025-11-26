#!/usr/bin/env python
"""Tests for evaluate.py checkpoint loading and generation."""

import sys
import tempfile
from pathlib import Path
import torch
import pytest
from transformers import GPT2TokenizerFast

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.models import ReferenceTransformer
from projects.transformer_fundamentals.evaluate import (
    load_model_from_checkpoint,
    generate_summary
)


@pytest.fixture
def small_model_config():
    """Small model config for testing."""
    return {
        'vocab_size': 50257,
        'd_model': 64,
        'n_heads': 4,
        'n_encoder_layers': 2,
        'n_decoder_layers': 2,
        'd_ff': 128,
        'dropout': 0.1,
        'max_seq_len': 128,
    }


@pytest.fixture
def tokenizer():
    """GPT2 tokenizer for testing."""
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_checkpoint(model_config, include_model_config=True, training_config=None):
    """Helper to create a test checkpoint.

    Args:
        model_config: Model configuration dict
        include_model_config: If True, save model_config in checkpoint (new format)
                             If False, save empty model_config (old format)
        training_config: Optional training config dict to save

    Returns:
        Path to checkpoint file
    """
    # Create model
    model = ReferenceTransformer(**model_config)

    # Create temporary checkpoint file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
    checkpoint_path = temp_file.name
    temp_file.close()

    # Prepare checkpoint dict
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'vocab_size': model_config['vocab_size'],
        'epoch': 5,
        'loss': 1.23,
    }

    if include_model_config:
        # New format: include full model_config
        checkpoint['model_config'] = {
            k: v for k, v in model_config.items() if k != 'vocab_size'
        }
    else:
        # Old format: empty model_config
        checkpoint['model_config'] = {}

    if training_config is not None:
        checkpoint['training_config'] = training_config

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def test_load_checkpoint_new_format(small_model_config):
    """Test loading checkpoint with full model_config (new format)."""
    # Create checkpoint with model_config
    checkpoint_path = create_checkpoint(small_model_config, include_model_config=True)

    try:
        # Load checkpoint
        device = torch.device('cpu')
        model, loaded_config = load_model_from_checkpoint(checkpoint_path, device)

        # Verify model loaded correctly
        assert model is not None
        assert isinstance(model, ReferenceTransformer)
        assert model.d_model == small_model_config['d_model']
        assert model.n_heads == small_model_config['n_heads']
        assert model.n_encoder_layers == small_model_config['n_encoder_layers']
        assert model.n_decoder_layers == small_model_config['n_decoder_layers']
        assert model.d_ff == small_model_config['d_ff']

        # Verify config was loaded
        assert loaded_config['d_model'] == small_model_config['d_model']
        assert loaded_config['n_heads'] == small_model_config['n_heads']

    finally:
        # Cleanup
        Path(checkpoint_path).unlink()


def test_load_checkpoint_old_format(small_model_config):
    """Test loading checkpoint with empty model_config (old format).

    This tests the config inference logic that extracts hyperparameters
    from the model state_dict.
    """
    # Create checkpoint without model_config
    checkpoint_path = create_checkpoint(small_model_config, include_model_config=False)

    try:
        # Load checkpoint
        device = torch.device('cpu')
        model, loaded_config = load_model_from_checkpoint(checkpoint_path, device)

        # Verify model loaded correctly
        assert model is not None
        assert isinstance(model, ReferenceTransformer)

        # Verify config was inferred correctly from state_dict
        assert loaded_config['d_model'] == small_model_config['d_model']
        assert loaded_config['n_heads'] == small_model_config['n_heads']
        assert loaded_config['d_ff'] == small_model_config['d_ff']
        assert loaded_config['n_encoder_layers'] == small_model_config['n_encoder_layers']
        assert loaded_config['n_decoder_layers'] == small_model_config['n_decoder_layers']

        # Verify model has correct architecture
        assert model.d_model == small_model_config['d_model']
        assert model.n_heads == small_model_config['n_heads']

    finally:
        # Cleanup
        Path(checkpoint_path).unlink()


def test_model_forward_pass(small_model_config):
    """Test that loaded model can perform forward pass."""
    checkpoint_path = create_checkpoint(small_model_config, include_model_config=False)

    try:
        # Load checkpoint
        device = torch.device('cpu')
        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        model.eval()

        # Create test input: [source tokens] [SEP] [target tokens]
        # Source: [100, 200, 300], SEP: 50256, Target: [400, 500]
        sep_token_id = model.sep_token_id
        input_ids = torch.tensor([[100, 200, 300, sep_token_id, 400, 500]], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)

        # Verify output structure
        assert 'logits' in outputs
        assert outputs['logits'].shape[0] == 1  # batch size
        assert outputs['logits'].shape[1] == 2  # target sequence length
        assert outputs['logits'].shape[2] == small_model_config['vocab_size']

    finally:
        # Cleanup
        Path(checkpoint_path).unlink()


def test_generate_summary(small_model_config, tokenizer):
    """Test that generation works without errors."""
    checkpoint_path = create_checkpoint(small_model_config, include_model_config=False)

    try:
        # Load checkpoint
        device = torch.device('cpu')
        model, _ = load_model_from_checkpoint(checkpoint_path, device)

        # Generate summary for test dialogue
        dialogue = "Alice: Hi! Bob: Hello! How are you?"

        # This should not raise an error
        summary = generate_summary(
            model=model,
            tokenizer=tokenizer,
            dialogue=dialogue,
            max_length=10,  # Short to speed up test
            temperature=1.0,
            device=device
        )

        # Verify we got a string back
        assert isinstance(summary, str)
        # Summary might be empty or gibberish since model is untrained,
        # but it should not crash

    finally:
        # Cleanup
        Path(checkpoint_path).unlink()


def test_config_inference_multiple_layers():
    """Test config inference works for different layer counts."""
    configs = [
        {'vocab_size': 50257, 'd_model': 128, 'n_heads': 8,
         'n_encoder_layers': 3, 'n_decoder_layers': 3, 'd_ff': 512},
        {'vocab_size': 50257, 'd_model': 256, 'n_heads': 4,
         'n_encoder_layers': 6, 'n_decoder_layers': 6, 'd_ff': 1024},
    ]

    for config in configs:
        checkpoint_path = create_checkpoint(config, include_model_config=False)

        try:
            device = torch.device('cpu')
            model, loaded_config = load_model_from_checkpoint(checkpoint_path, device)

            # Verify layer counts were inferred correctly
            assert loaded_config['n_encoder_layers'] == config['n_encoder_layers']
            assert loaded_config['n_decoder_layers'] == config['n_decoder_layers']
            assert loaded_config['d_model'] == config['d_model']
            assert loaded_config['d_ff'] == config['d_ff']

        finally:
            Path(checkpoint_path).unlink()


def test_model_state_dict_compatibility(small_model_config):
    """Test that state_dict loads correctly and weights match."""
    checkpoint_path = create_checkpoint(small_model_config, include_model_config=False)

    try:
        # Load original checkpoint to get original weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        original_embedding = checkpoint['model_state_dict']['embedding'].clone()

        # Load through our function
        device = torch.device('cpu')
        model, _ = load_model_from_checkpoint(checkpoint_path, device)

        # Verify weights match
        assert torch.allclose(model.embedding, original_embedding)

    finally:
        Path(checkpoint_path).unlink()


def test_checkpoint_with_training_config(small_model_config):
    """Test that training_config is saved and can be loaded from checkpoint."""
    # Create a mock training config
    training_config = {
        'training': {
            'learning_rate': 0.0001,
            'batch_size': 32,
            'warmup_steps': 1000,
            'max_steps': 10000,
        },
        'data': {
            'dataset_name': 'samsum',
            'max_dialogue_length': 256,
            'max_summary_length': 64,
        },
        'model': small_model_config,
    }

    # Create checkpoint with training config
    checkpoint_path = create_checkpoint(
        small_model_config,
        include_model_config=True,
        training_config=training_config
    )

    try:
        # Load checkpoint and verify training_config is present
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        assert 'training_config' in checkpoint
        assert checkpoint['training_config']['training']['learning_rate'] == 0.0001
        assert checkpoint['training_config']['training']['batch_size'] == 32
        assert checkpoint['training_config']['data']['dataset_name'] == 'samsum'

        # Verify model still loads correctly
        device = torch.device('cpu')
        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        assert model is not None

    finally:
        # Cleanup
        Path(checkpoint_path).unlink()


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
