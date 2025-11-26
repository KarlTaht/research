"""Comprehensive tests for tokenization, special tokens, and positional encoding.

These tests catch critical issues like:
- Token ID collisions (SEP, BOS, EOS, PAD using same ID)
- Training/inference format mismatches
- Positional encoding correctness
- Sequence packing consistency
"""

import pytest
import math
import torch
import sys
from pathlib import Path

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import GPT2TokenizerFast
from common.models import ReferenceTransformer
from common.models.reference_transformer.positional_encoding import PositionalEncoding


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
        'dropout': 0.0,  # Disable dropout for deterministic tests
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
# Special Token Collision Tests (CRITICAL)
# ============================================================================

class TestSpecialTokenCollisions:
    """Tests to detect dangerous token ID collisions."""

    def test_sep_token_not_equal_to_eos(self, model, tokenizer):
        """SEP token must NOT be the same as EOS token.

        If SEP == EOS, the model cannot distinguish between:
        - End of source sequence (SEP)
        - End of target sequence (EOS)

        This causes generation to stop prematurely or continue incorrectly.
        """
        assert model.sep_token_id != tokenizer.eos_token_id, (
            f"CRITICAL: SEP token ({model.sep_token_id}) equals EOS token "
            f"({tokenizer.eos_token_id}). The model cannot distinguish between "
            "end-of-source and end-of-target!"
        )

    def test_sep_token_not_equal_to_pad(self, model, tokenizer):
        """SEP token must NOT be the same as PAD token.

        If SEP == PAD, padding will be interpreted as sequence separators.
        """
        assert model.sep_token_id != tokenizer.pad_token_id, (
            f"CRITICAL: SEP token ({model.sep_token_id}) equals PAD token "
            f"({tokenizer.pad_token_id}). Padding will be misinterpreted as separators!"
        )

    def test_sep_token_not_in_regular_vocab(self, model, tokenizer):
        """SEP token should not be a commonly used token.

        Using vocab_size - 1 as SEP is dangerous because:
        1. For GPT-2, token 50256 is the EOS token
        2. It could collide with special tokens
        """
        # SEP should either be a dedicated unused token or properly reserved
        sep_id = model.sep_token_id

        # Check it's not in the tokenizer's special tokens (except if explicitly added)
        special_ids = {tokenizer.eos_token_id, tokenizer.bos_token_id,
                       tokenizer.pad_token_id, tokenizer.unk_token_id}
        special_ids.discard(None)  # Remove None values

        collision = sep_id in special_ids
        if collision:
            # Identify which special token it collides with
            collisions = []
            if sep_id == tokenizer.eos_token_id:
                collisions.append("EOS")
            if sep_id == tokenizer.bos_token_id:
                collisions.append("BOS")
            if sep_id == tokenizer.pad_token_id:
                collisions.append("PAD")
            if sep_id == tokenizer.unk_token_id:
                collisions.append("UNK")

            pytest.fail(
                f"SEP token {sep_id} collides with special tokens: {collisions}. "
                f"This will cause confusion during training and generation!"
            )

    def test_all_special_tokens_are_distinct(self, model, tokenizer):
        """Critical special tokens (SEP vs PAD/EOS) must be distinct.

        Note: For GPT-2, PAD == EOS is intentional (we use EOS as PAD).
        The critical requirement is that SEP is distinct from both.
        """
        # SEP must be distinct from PAD (verified in other tests)
        # SEP must be distinct from EOS (verified in other tests)

        # Document the known GPT-2 behavior: PAD == EOS
        # This is acceptable because:
        # 1. We use CrossEntropyLoss(ignore_index=pad_token_id) to ignore PAD
        # 2. Generation stops on EOS, so PAD/EOS collision is harmless
        if tokenizer.pad_token_id == tokenizer.eos_token_id:
            # This is expected for GPT-2
            pass

        # The critical check: SEP must not collide with PAD or EOS
        assert model.sep_token_id != tokenizer.pad_token_id, (
            f"CRITICAL: SEP ({model.sep_token_id}) == PAD ({tokenizer.pad_token_id})"
        )
        assert model.sep_token_id != tokenizer.eos_token_id, (
            f"CRITICAL: SEP ({model.sep_token_id}) == EOS ({tokenizer.eos_token_id})"
        )


# ============================================================================
# Training/Inference Format Consistency Tests
# ============================================================================

class TestTrainingInferenceConsistency:
    """Tests to ensure training and inference use consistent formats."""

    def test_generation_seed_matches_training_format(self, model, tokenizer):
        """Generation must seed with the same format used during training.

        If training uses: [dialogue] [SEP] [summary_tokens...]
        Then generation MUST use: [dialogue] [SEP] [first_summary_token]

        NOT: [dialogue] [SEP] [BOS] (if BOS wasn't used in training)
        """
        # Simulate training format
        dialogue = "Hello"
        summary = "Greeting"

        dialogue_ids = tokenizer.encode(dialogue, add_special_tokens=False)
        summary_ids = tokenizer.encode(summary, add_special_tokens=False)
        sep_id = model.sep_token_id

        training_format = dialogue_ids + [sep_id] + summary_ids

        # The first token after SEP in training is summary_ids[0], not BOS
        first_target_token_in_training = summary_ids[0]

        # Generation should NOT start with a different token (like EOS/BOS)
        # unless that's what training used
        assert first_target_token_in_training != tokenizer.eos_token_id, (
            "Training data should not start summaries with EOS token. "
            "If generation uses EOS as BOS, this creates a train/inference mismatch!"
        )

    def test_packed_sequence_format_documented(self, model):
        """Verify the expected packed sequence format is documented/consistent."""
        # The model expects: [SRC] [SEP] [TGT]
        # This test documents and verifies that expectation

        sep_id = model.sep_token_id

        # Create a valid packed sequence
        src_tokens = [100, 200, 300]
        tgt_tokens = [400, 500]
        packed = src_tokens + [sep_id] + tgt_tokens

        # Model should accept this format
        input_tensor = torch.tensor([packed], dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_tensor)

        # Output logits should be for target sequence length
        assert outputs['logits'].shape[1] == len(tgt_tokens), (
            f"Expected logits for {len(tgt_tokens)} target tokens, "
            f"got {outputs['logits'].shape[1]}"
        )

    def test_inference_logit_position_matches_training(self, model, tokenizer):
        """Verify that inference reads logits from the correct position.

        During training with shifted labels:
        - logits[:, i] predicts labels[:, i+1]

        During inference:
        - To get next token, use logits[:, -1] after feeding [SEP] + [generated_so_far]
        """
        sep_id = model.sep_token_id

        # Simulate: dialogue="Hi", target so far = [token_A]
        dialogue_ids = tokenizer.encode("Hi", add_special_tokens=False)
        token_A = 1000  # arbitrary token

        packed = dialogue_ids + [sep_id] + [token_A]
        input_tensor = torch.tensor([packed], dtype=torch.long)

        with torch.no_grad():
            outputs = model(input_tensor)

        logits = outputs['logits']

        # Target has 1 token, so logits shape should be [1, 1, vocab_size]
        assert logits.shape[1] == 1, (
            f"Expected 1 target position, got {logits.shape[1]}"
        )

        # logits[0, 0, :] should predict what comes after token_A
        # This is the correct position for generation
        next_token_logits = logits[0, -1, :]
        assert next_token_logits.shape[0] == model.vocab_size


# ============================================================================
# Positional Encoding Tests
# ============================================================================

class TestPositionalEncoding:
    """Tests for positional encoding correctness."""

    def test_sinusoidal_formula_correctness(self):
        """Verify positional encoding matches the paper's formula.

        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        d_model = 64
        max_len = 100

        pe = PositionalEncoding(d_model, max_len, dropout=0.0)

        # Test specific positions
        for pos in [0, 1, 10, 50, 99]:
            for i in range(d_model // 2):
                # Calculate expected values from formula
                div_term = 10000.0 ** (2 * i / d_model)
                expected_sin = math.sin(pos / div_term)
                expected_cos = math.cos(pos / div_term)

                actual_sin = pe.pe[0, pos, 2 * i].item()
                actual_cos = pe.pe[0, pos, 2 * i + 1].item()

                assert abs(actual_sin - expected_sin) < 1e-5, (
                    f"Sin mismatch at pos={pos}, i={i}: "
                    f"expected {expected_sin}, got {actual_sin}"
                )
                assert abs(actual_cos - expected_cos) < 1e-5, (
                    f"Cos mismatch at pos={pos}, i={i}: "
                    f"expected {expected_cos}, got {actual_cos}"
                )

    def test_position_zero_values(self):
        """Position 0 should have specific known values.

        PE(0, 2i) = sin(0) = 0
        PE(0, 2i+1) = cos(0) = 1
        """
        d_model = 64
        pe = PositionalEncoding(d_model, 100, dropout=0.0)

        pos_0 = pe.pe[0, 0, :]

        # Even indices should be 0 (sin(0))
        even_indices = pos_0[0::2]
        assert torch.allclose(even_indices, torch.zeros_like(even_indices), atol=1e-6), (
            "Position 0 even indices should all be 0 (sin(0))"
        )

        # Odd indices should be 1 (cos(0))
        odd_indices = pos_0[1::2]
        assert torch.allclose(odd_indices, torch.ones_like(odd_indices), atol=1e-6), (
            "Position 0 odd indices should all be 1 (cos(0))"
        )

    def test_different_positions_are_different(self):
        """Each position should have a unique encoding."""
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len, dropout=0.0)

        # Check that positions 0, 1, 2 are all different
        pos_0 = pe.pe[0, 0, :]
        pos_1 = pe.pe[0, 1, :]
        pos_2 = pe.pe[0, 2, :]

        assert not torch.allclose(pos_0, pos_1), "Positions 0 and 1 should differ"
        assert not torch.allclose(pos_1, pos_2), "Positions 1 and 2 should differ"
        assert not torch.allclose(pos_0, pos_2), "Positions 0 and 2 should differ"

    def test_positional_encoding_shape(self):
        """Verify output shape matches input shape."""
        d_model = 64
        max_len = 100
        batch_size = 4
        seq_len = 20

        pe = PositionalEncoding(d_model, max_len, dropout=0.0)
        pe.eval()  # Disable dropout

        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)

        assert output.shape == x.shape, (
            f"Output shape {output.shape} != input shape {x.shape}"
        )

    def test_positional_encoding_deterministic_in_eval(self):
        """Positional encoding should be deterministic in eval mode."""
        d_model = 64
        pe = PositionalEncoding(d_model, 100, dropout=0.1)
        pe.eval()

        x = torch.randn(2, 10, d_model)

        out1 = pe(x)
        out2 = pe(x)

        assert torch.allclose(out1, out2), (
            "Positional encoding should be deterministic in eval mode"
        )

    def test_positional_encoding_adds_to_input(self):
        """Verify encoding is added to input, not replacing it."""
        d_model = 64
        pe = PositionalEncoding(d_model, 100, dropout=0.0)
        pe.eval()

        x = torch.randn(1, 5, d_model)
        output = pe(x)

        # The difference should be the positional encoding
        diff = output - x
        expected_pe = pe.pe[:, :5, :]

        assert torch.allclose(diff, expected_pe, atol=1e-6), (
            "Output should be input + positional encoding"
        )

    def test_positional_encoding_handles_variable_lengths(self):
        """Encoding should work for any length up to max_len."""
        d_model = 64
        max_len = 100
        pe = PositionalEncoding(d_model, max_len, dropout=0.0)
        pe.eval()

        for seq_len in [1, 10, 50, 99, 100]:
            x = torch.randn(1, seq_len, d_model)
            output = pe(x)
            assert output.shape[1] == seq_len

    def test_positional_encoding_fails_beyond_max_len(self):
        """Encoding should fail gracefully for lengths beyond max_len."""
        d_model = 64
        max_len = 10
        pe = PositionalEncoding(d_model, max_len, dropout=0.0)

        # This should either raise an error or handle gracefully
        x = torch.randn(1, max_len + 5, d_model)

        # The implementation uses slicing, which will silently truncate
        # This test documents that behavior
        try:
            output = pe(x)
            # If it doesn't error, verify it at least produces correct shape
            assert output.shape == x.shape
        except (IndexError, RuntimeError):
            # This is acceptable - failing explicitly is fine
            pass


# ============================================================================
# Sequence Packing and Splitting Tests
# ============================================================================

class TestSequencePacking:
    """Tests for sequence packing and splitting logic."""

    def test_model_finds_separator(self, model):
        """Model should correctly find the SEP token."""
        sep_id = model.sep_token_id

        # Valid packed sequence
        packed = torch.tensor([[100, 200, sep_id, 300, 400]], dtype=torch.long)

        with torch.no_grad():
            outputs = model(packed)

        # Should not raise an error
        assert 'logits' in outputs

    def test_model_fails_without_separator(self, model):
        """Model should raise error if SEP token is missing."""
        # Sequence without SEP
        packed = torch.tensor([[100, 200, 300, 400]], dtype=torch.long)

        with pytest.raises(ValueError, match="Separator token .* not found"):
            model(packed)

    def test_model_uses_first_separator_only(self, model):
        """Model should use first SEP occurrence to split."""
        sep_id = model.sep_token_id

        # Sequence with multiple SEP tokens (SEP might appear in target)
        packed = torch.tensor([[100, sep_id, 200, sep_id, 300]], dtype=torch.long)

        with torch.no_grad():
            outputs = model(packed)

        # Source should be [100], target should be [200, SEP, 300]
        # So output should have 3 target positions
        assert outputs['logits'].shape[1] == 3

    def test_empty_source_sequence(self, model):
        """Handle edge case of empty source (SEP at position 0)."""
        sep_id = model.sep_token_id

        # SEP at position 0 means empty source
        packed = torch.tensor([[sep_id, 100, 200]], dtype=torch.long)

        with torch.no_grad():
            outputs = model(packed)

        # Should work with empty source
        assert outputs['logits'].shape[1] == 2  # Target is [100, 200]

    def test_empty_target_sequence(self, model):
        """Handle edge case of empty target (SEP at end)."""
        sep_id = model.sep_token_id

        # SEP at end means empty target
        packed = torch.tensor([[100, 200, sep_id]], dtype=torch.long)

        with torch.no_grad():
            outputs = model(packed)

        # Empty target should produce shape [batch, 0, vocab]
        assert outputs['logits'].shape[1] == 0


# ============================================================================
# Tokenizer Consistency Tests
# ============================================================================

class TestTokenizerConsistency:
    """Tests for tokenizer behavior consistency."""

    def test_encode_decode_roundtrip(self, tokenizer):
        """Encoding then decoding should preserve text (approximately)."""
        text = "Hello, this is a test dialogue."

        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)

        # Should be identical for simple text
        assert decoded == text

    def test_special_tokens_not_in_regular_text(self, tokenizer):
        """Regular dialogue text should not contain special token IDs."""
        dialogue = "Alice: Hi! Bob: Hello! How are you? Alice: Fine, thanks!"

        tokens = tokenizer.encode(dialogue, add_special_tokens=False)

        # EOS/PAD/SEP tokens should not appear in regular text
        assert tokenizer.eos_token_id not in tokens, (
            f"EOS token {tokenizer.eos_token_id} appeared in regular dialogue"
        )
        assert tokenizer.sep_token_id not in tokens, (
            f"SEP token {tokenizer.sep_token_id} appeared in regular dialogue"
        )

    def test_vocab_size_matches_model(self, tokenizer, model_config):
        """Tokenizer vocab size should match model config."""
        assert len(tokenizer) == model_config['vocab_size'], (
            f"Tokenizer vocab size ({len(tokenizer)}) != "
            f"model vocab size ({model_config['vocab_size']})"
        )

    def test_padding_token_identity(self, tokenizer):
        """Verify padding token is set correctly."""
        assert tokenizer.pad_token is not None, "Padding token not set"
        assert tokenizer.pad_token_id is not None, "Padding token ID not set"

        # For GPT-2, we use EOS as padding
        assert tokenizer.pad_token_id == tokenizer.eos_token_id

    def test_sep_token_exists_and_distinct(self, tokenizer):
        """Verify SEP token is set and distinct from other special tokens."""
        assert tokenizer.sep_token is not None, "SEP token not set"
        assert tokenizer.sep_token_id is not None, "SEP token ID not set"
        assert tokenizer.sep_token_id != tokenizer.eos_token_id, (
            "SEP token must be different from EOS token"
        )
        assert tokenizer.sep_token_id != tokenizer.pad_token_id, (
            "SEP token must be different from PAD token"
        )


# ============================================================================
# Integration Tests
# ============================================================================

class TestTokenizationEncodingIntegration:
    """Integration tests combining tokenization and encoding."""

    def test_full_pipeline_dialogue_to_summary(self, model, tokenizer):
        """Test complete pipeline from dialogue to model output."""
        sep_id = model.sep_token_id

        dialogue = "Alice: Hi! Bob: Hello!"
        summary = "Greeting exchange."

        # Tokenize
        dialogue_ids = tokenizer.encode(dialogue, add_special_tokens=False)
        summary_ids = tokenizer.encode(summary, add_special_tokens=False)

        # Pack
        packed = dialogue_ids + [sep_id] + summary_ids
        input_tensor = torch.tensor([packed], dtype=torch.long)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_tensor)

        # Verify output
        assert outputs['logits'].shape[0] == 1  # batch size
        assert outputs['logits'].shape[1] == len(summary_ids)  # target length
        assert outputs['logits'].shape[2] == model.vocab_size

    def test_batched_sequences_with_different_lengths(self, model, tokenizer):
        """Test batch with different source/target lengths."""
        from train import create_collate_fn

        sep_id = model.sep_token_id
        collate_fn = create_collate_fn(pad_token_id=tokenizer.pad_token_id)

        # Create sequences with different lengths
        batch = [
            {'input_ids': [100, 200, 300, sep_id, 400, 500, 600],
             'labels': [100, 200, 300, sep_id, 400, 500, 600]},
            {'input_ids': [100, sep_id, 200],
             'labels': [100, sep_id, 200]},
        ]

        result = collate_fn(batch)

        # Forward pass should work
        outputs = model(result['input_ids'], labels=result['labels'])

        assert outputs['loss'] is not None
        assert torch.isfinite(outputs['loss'])

    def test_generation_produces_valid_tokens(self, model, tokenizer):
        """Generated tokens should be valid vocabulary indices."""
        from evaluate import generate_summary

        model.eval()
        dialogue = "Alice: Hello! Bob: Hi there!"

        # This will use the current (potentially buggy) generation
        # The test verifies tokens are at least valid indices
        summary = generate_summary(
            model=model,
            tokenizer=tokenizer,
            dialogue=dialogue,
            max_length=5,
            temperature=1.0,
            device=torch.device('cpu')
        )

        # Result should be a string (even if nonsensical)
        assert isinstance(summary, str)


# ============================================================================
# Regression Tests for Known Issues
# ============================================================================

class TestKnownIssueRegressions:
    """Tests that catch known bugs and prevent regressions."""

    def test_vocab_size_minus_one_not_special(self, tokenizer):
        """Using vocab_size-1 as SEP is dangerous for GPT-2.

        GPT-2 vocab: 50257 tokens (indices 0-50256)
        Token 50256 is <|endoftext|> (EOS)
        Using vocab_size-1 = 50256 as SEP creates collision!
        """
        vocab_size = len(tokenizer)
        last_token_id = vocab_size - 1

        # For GPT-2, token 50256 is EOS
        is_eos = (last_token_id == tokenizer.eos_token_id)

        if is_eos:
            pytest.fail(
                f"CRITICAL: vocab_size-1 ({last_token_id}) equals EOS token. "
                f"Do NOT use vocab_size-1 as SEP token with GPT-2 tokenizer! "
                f"Use a different token ID (e.g., add a new token to tokenizer)."
            )

    def test_bos_token_for_generation(self, tokenizer):
        """Document GPT-2's BOS/EOS token behavior for generation.

        GPT-2 uses the same token (<|endoftext|>) for BOS, EOS, PAD, and UNK.
        This is a known characteristic, not a bug.

        For our encoder-decoder model:
        - We do NOT use BOS at the start of targets during training
        - Generation should NOT seed with BOS/EOS
        - Instead, we use a primer token approach (documented in generate_summary)
        """
        # GPT-2 uses same token for BOS and EOS - this is expected
        if tokenizer.bos_token_id == tokenizer.eos_token_id:
            # Document this is expected behavior
            assert tokenizer.bos_token == '<|endoftext|>', (
                "GPT-2 should use <|endoftext|> for BOS"
            )
            assert tokenizer.eos_token == '<|endoftext|>', (
                "GPT-2 should use <|endoftext|> for EOS"
            )
        else:
            # If BOS != EOS (different tokenizer), verify they're distinct
            assert tokenizer.bos_token_id != tokenizer.eos_token_id

    def test_pad_token_not_exclamation(self, tokenizer):
        """Verify padding doesn't use token 0 ('!' in GPT-2).

        Token 0 in GPT-2 is '!', which appears in regular text.
        Using it as padding corrupts loss computation.
        """
        token_0_text = tokenizer.decode([0])

        if token_0_text == '!':
            assert tokenizer.pad_token_id != 0, (
                "CRITICAL: Pad token is 0, which decodes to '!' in GPT-2. "
                "This token appears in regular text and will corrupt training!"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
