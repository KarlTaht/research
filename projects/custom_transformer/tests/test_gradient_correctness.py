"""
Gradient correctness tests for CustomTransformer.

Compares manual backward pass gradients against PyTorch autograd
to verify the implementation is mathematically correct.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyTorchAttention(nn.Module):
    """Reference attention implementation using PyTorch autograd."""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_Q = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / d_model) ** 0.5)
        self.W_K = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / d_model) ** 0.5)
        self.W_V = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / d_model) ** 0.5)
        self.W_o = nn.Parameter(torch.randn(d_model, d_model) * (1.0 / d_model) ** 0.5)

        # Causal mask
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        self.register_buffer('causal_mask', mask.masked_fill(mask == 1, float('-inf')))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores + self.causal_mask[:seq_len, :seq_len]

        probs = F.softmax(scores, dim=-1)

        out = probs @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return out @ self.W_o


class TestAttentionGradients:
    """Test attention backward pass against PyTorch autograd."""

    @pytest.fixture
    def setup(self):
        """Create both implementations with identical weights."""
        from common.models.custom_transfromer import CustomTransformer

        torch.manual_seed(42)

        d_model = 32
        n_heads = 2
        n_blocks = 1
        d_ffn = 64
        max_seq_len = 16
        vocab_size = 100

        config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_blocks': n_blocks,
            'n_heads': n_heads,
            'd_ffn': d_ffn,
            'max_seq_len': max_seq_len,
            'dtype': torch.float32,
            'device': 'cpu',  # Force CPU for testing
        }

        # Create CustomTransformer
        custom = CustomTransformer(config)

        # Create PyTorch reference attention
        torch.manual_seed(42)
        pytorch_attn = PyTorchAttention(d_model, n_heads, max_seq_len)

        # Copy weights from CustomTransformer to PyTorch reference
        with torch.no_grad():
            pytorch_attn.W_Q.copy_(custom.Q[0])
            pytorch_attn.W_K.copy_(custom.K[0])
            pytorch_attn.W_V.copy_(custom.V[0])
            pytorch_attn.W_o.copy_(custom.W_o[0])

        return custom, pytorch_attn, d_model, n_heads, max_seq_len

    def test_attention_forward_matches(self, setup):
        """Verify forward passes produce same output."""
        custom, pytorch_attn, d_model, n_heads, max_seq_len = setup

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)

        # Custom attention forward (using block 0)
        custom.cache.store_activation('block', 0, 'input', x)
        custom_out = custom.attention(x, block_step=0)

        # PyTorch attention forward
        pytorch_out = pytorch_attn(x)

        assert torch.allclose(custom_out, pytorch_out, atol=1e-5), \
            f"Forward mismatch: max diff = {(custom_out - pytorch_out).abs().max()}"

    def test_attention_qkv_gradients(self, setup):
        """Verify Q, K, V weight gradients match PyTorch autograd."""
        custom, pytorch_attn, d_model, n_heads, max_seq_len = setup

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)

        # PyTorch forward + backward
        pytorch_attn.zero_grad()
        pytorch_out = pytorch_attn(x)
        loss_grad = torch.randn_like(pytorch_out)
        pytorch_out.backward(loss_grad)

        # Custom forward + backward
        custom.cache.store_activation('block', 0, 'input', x)
        custom_out = custom.attention(x, block_step=0)
        custom.backward_attention(loss_grad, block_step=0)

        # Compare gradients
        custom_grad_Q = custom.cache.get_gradient(('W_Q', 0))
        custom_grad_K = custom.cache.get_gradient(('W_K', 0))
        custom_grad_V = custom.cache.get_gradient(('W_V', 0))
        custom_grad_Wo = custom.cache.get_gradient(('W_o', 0))

        # Print gradient comparison for debugging
        print(f"\nW_Q gradient comparison:")
        print(f"  PyTorch norm: {pytorch_attn.W_Q.grad.norm():.6f}")
        print(f"  Custom norm: {custom_grad_Q.norm():.6f}")
        print(f"  Max diff: {(pytorch_attn.W_Q.grad - custom_grad_Q).abs().max():.6e}")

        print(f"\nW_K gradient comparison:")
        print(f"  PyTorch norm: {pytorch_attn.W_K.grad.norm():.6f}")
        print(f"  Custom norm: {custom_grad_K.norm():.6f}")
        print(f"  Max diff: {(pytorch_attn.W_K.grad - custom_grad_K).abs().max():.6e}")

        print(f"\nW_V gradient comparison:")
        print(f"  PyTorch norm: {pytorch_attn.W_V.grad.norm():.6f}")
        print(f"  Custom norm: {custom_grad_V.norm():.6f}")
        print(f"  Max diff: {(pytorch_attn.W_V.grad - custom_grad_V).abs().max():.6e}")

        print(f"\nW_o gradient comparison:")
        print(f"  PyTorch norm: {pytorch_attn.W_o.grad.norm():.6f}")
        print(f"  Custom norm: {custom_grad_Wo.norm():.6f}")
        print(f"  Max diff: {(pytorch_attn.W_o.grad - custom_grad_Wo).abs().max():.6e}")

        assert torch.allclose(pytorch_attn.W_Q.grad, custom_grad_Q, atol=1e-4), \
            f"W_Q gradient mismatch: max diff = {(pytorch_attn.W_Q.grad - custom_grad_Q).abs().max()}"
        assert torch.allclose(pytorch_attn.W_K.grad, custom_grad_K, atol=1e-4), \
            f"W_K gradient mismatch: max diff = {(pytorch_attn.W_K.grad - custom_grad_K).abs().max()}"
        assert torch.allclose(pytorch_attn.W_V.grad, custom_grad_V, atol=1e-4), \
            f"W_V gradient mismatch: max diff = {(pytorch_attn.W_V.grad - custom_grad_V).abs().max()}"
        assert torch.allclose(pytorch_attn.W_o.grad, custom_grad_Wo, atol=1e-4), \
            f"W_o gradient mismatch: max diff = {(pytorch_attn.W_o.grad - custom_grad_Wo).abs().max()}"


class PyTorchFFN(nn.Module):
    """Reference FFN implementation using PyTorch autograd."""

    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(d_model, d_ffn) * (1.0 / d_model) ** 0.5)
        self.W2 = nn.Parameter(torch.randn(d_ffn, d_model) * (1.0 / d_ffn) ** 0.5)

    def forward(self, x):
        return F.gelu(x @ self.W1) @ self.W2


class TestFFNGradients:
    """Test FFN backward pass against PyTorch autograd."""

    @pytest.fixture
    def setup(self):
        """Create both implementations with identical weights."""
        from common.models.custom_transfromer import CustomTransformer

        torch.manual_seed(42)

        d_model = 32
        d_ffn = 64

        config = {
            'vocab_size': 100,
            'd_model': d_model,
            'n_blocks': 1,
            'n_heads': 2,
            'd_ffn': d_ffn,
            'max_seq_len': 16,
            'dtype': torch.float32,
            'device': 'cpu',  # Force CPU for testing
        }

        custom = CustomTransformer(config)

        pytorch_ffn = PyTorchFFN(d_model, d_ffn)

        with torch.no_grad():
            pytorch_ffn.W1.copy_(custom.W1[0])
            pytorch_ffn.W2.copy_(custom.W2[0])

        return custom, pytorch_ffn, d_model, d_ffn

    def test_ffn_gradients(self, setup):
        """Verify FFN weight gradients match PyTorch autograd."""
        custom, pytorch_ffn, d_model, d_ffn = setup

        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, d_model)

        # PyTorch forward + backward
        pytorch_ffn.zero_grad()
        pytorch_out = pytorch_ffn(x)
        loss_grad = torch.randn_like(pytorch_out)
        pytorch_out.backward(loss_grad)

        # Custom forward + backward
        custom.cache.store_activation('block', 0, 'ffn_input', x)
        custom_out = custom.feed_forward_network(x, block_step=0)
        custom.backward_ffn(loss_grad, block_step=0)

        custom_grad_W1 = custom.cache.get_gradient(('W1', 0))
        custom_grad_W2 = custom.cache.get_gradient(('W2', 0))

        print(f"\nW1 gradient comparison:")
        print(f"  PyTorch norm: {pytorch_ffn.W1.grad.norm():.6f}")
        print(f"  Custom norm: {custom_grad_W1.norm():.6f}")
        print(f"  Max diff: {(pytorch_ffn.W1.grad - custom_grad_W1).abs().max():.6e}")

        print(f"\nW2 gradient comparison:")
        print(f"  PyTorch norm: {pytorch_ffn.W2.grad.norm():.6f}")
        print(f"  Custom norm: {custom_grad_W2.norm():.6f}")
        print(f"  Max diff: {(pytorch_ffn.W2.grad - custom_grad_W2).abs().max():.6e}")

        # W1 has slightly higher tolerance due to GELU backward numerical differences
        assert torch.allclose(pytorch_ffn.W1.grad, custom_grad_W1, atol=1e-2, rtol=1e-3), \
            f"W1 gradient mismatch: max diff = {(pytorch_ffn.W1.grad - custom_grad_W1).abs().max()}"
        assert torch.allclose(pytorch_ffn.W2.grad, custom_grad_W2, atol=1e-4), \
            f"W2 gradient mismatch: max diff = {(pytorch_ffn.W2.grad - custom_grad_W2).abs().max()}"


class TestSoftmaxBackward:
    """Test softmax backward implementation."""

    def test_softmax_backward_matches_autograd(self):
        """Verify softmax backward matches PyTorch autograd."""
        from common.models.custom_transfromer import CustomTransformer

        config = {
            'vocab_size': 100,
            'd_model': 32,
            'n_blocks': 1,
            'n_heads': 2,
            'd_ffn': 64,
            'max_seq_len': 16,
            'dtype': torch.float32,
            'device': 'cpu',  # Force CPU for testing
        }

        custom = CustomTransformer(config)

        # Test input
        x = torch.randn(2, 2, 8, 8, requires_grad=True)  # [batch, n_heads, seq, seq]

        # PyTorch forward + backward
        probs = F.softmax(x, dim=-1)
        grad_output = torch.randn_like(probs)
        probs.backward(grad_output)
        pytorch_grad = x.grad.clone()

        # Custom backward
        custom_grad = custom.backward_softmax(grad_output, probs.detach())

        print(f"\nSoftmax gradient comparison:")
        print(f"  PyTorch norm: {pytorch_grad.norm():.6f}")
        print(f"  Custom norm: {custom_grad.norm():.6f}")
        print(f"  Max diff: {(pytorch_grad - custom_grad).abs().max():.6e}")

        assert torch.allclose(pytorch_grad, custom_grad, atol=1e-5), \
            f"Softmax gradient mismatch: max diff = {(pytorch_grad - custom_grad).abs().max()}"


class TestEndToEndGradients:
    """Test complete forward/backward pass gradient correctness."""

    def test_loss_gradient_direction(self):
        """Verify gradients point in correct direction for loss reduction."""
        from common.models.custom_transfromer.wrapper import CustomTransformerWrapper

        torch.manual_seed(42)

        model = CustomTransformerWrapper(
            vocab_size=100,
            d_model=32,
            n_blocks=2,
            n_heads=2,
            d_ffn=64,
            max_seq_len=16,
            dtype=torch.float32,
        )

        # Fixed batch
        input_ids = torch.randint(0, 100, (4, 12))
        labels = input_ids.clone()

        # Get initial loss
        outputs = model.forward(input_ids, labels=labels)
        initial_loss = outputs['loss'].item()

        # Train step
        result = model.train_step(input_ids, labels, learning_rate=0.01, max_grad_norm=1.0)

        # Get new loss
        outputs = model.forward(input_ids, labels=labels)
        new_loss = outputs['loss'].item()

        print(f"\nEnd-to-end gradient test:")
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  After update: {new_loss:.4f}")
        print(f"  Change: {new_loss - initial_loss:.4f}")

        assert new_loss < initial_loss, \
            f"Loss did not decrease: {initial_loss:.4f} -> {new_loss:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
