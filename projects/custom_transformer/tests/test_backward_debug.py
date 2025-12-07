#!/usr/bin/env python3
"""Debug test for backward pass gradient computation.

This script traces through the backward pass to identify where gradients
diverge from expected values.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn.functional as F
from common.models.custom_transfromer.wrapper import CustomTransformerWrapper


def test_backward_output_projection_gradient():
    """Test that backward computes correct output projection gradient."""

    config = {
        'vocab_size': 100,
        'max_seq_len': 32,
        'n_blocks': 2,
        'n_heads': 2,
        'd_model': 32,
        'd_ffn': 64,
        'dtype': torch.float32,
    }

    model = CustomTransformerWrapper(**config)

    torch.manual_seed(123)
    input_ids = torch.randint(0, 100, (1, 4))
    labels = input_ids.clone()

    # Manually step through train_step
    input_ids_dev = input_ids.to(model.device)
    labels_dev = labels.to(model.device)

    # Forward pass
    logits = model.model.forward(input_ids_dev)
    print(f"Logits shape: {logits.shape}")

    # Loss computation (from train_step)
    batch_size, seq_len = input_ids.shape
    logits_fp32 = logits.to(torch.float32)

    shift_logits = logits_fp32[..., :-1, :].contiguous()
    shift_labels = labels_dev[..., 1:].contiguous()

    loss = F.cross_entropy(shift_logits.view(-1, model.vocab_size), shift_labels.view(-1))
    print(f"Loss: {loss.item():.6f}")

    # Compute gradient (from train_step)
    shift_probs = F.softmax(shift_logits, dim=-1)
    shift_one_hot = F.one_hot(shift_labels, num_classes=model.vocab_size).to(torch.float32)
    num_loss_tokens = batch_size * (seq_len - 1)

    shift_gradient = (shift_probs - shift_one_hot) / num_loss_tokens
    loss_gradient = torch.zeros(batch_size, seq_len, model.vocab_size,
                               dtype=torch.float32, device=model.device)
    loss_gradient[:, :-1, :] = shift_gradient

    print(f"\nloss_gradient shape: {loss_gradient.shape}")
    print(f"loss_gradient norm: {loss_gradient.norm().item():.8f}")
    print(f"loss_gradient[0,0,0]: {loss_gradient[0, 0, 0].item():.8f}")

    # Get latent_result before backward
    latent_result = model.model.cache.get_activation('latent_result').clone()
    print(f"\nlatent_result shape: {latent_result.shape}")
    print(f"latent_result norm: {latent_result.norm().item():.8f}")

    # What the output projection gradient SHOULD be
    expected_grad = (latent_result.transpose(-2, -1) @ loss_gradient).sum(dim=0)
    print(f"\nExpected output_projection gradient:")
    print(f"  [0,0]: {expected_grad[0, 0].item():.8f}")
    print(f"  norm: {expected_grad.norm().item():.8f}")

    # Now call backward and see what it stores
    print("\n=== Calling backward ===")
    model.model.backward(loss_gradient)

    actual_grad = model.model.cache.gradients['output_projection']
    print(f"Actual output_projection gradient:")
    print(f"  [0,0]: {actual_grad[0, 0].item():.8f}")
    print(f"  norm: {actual_grad.norm().item():.8f}")

    ratio = expected_grad.norm().item() / actual_grad.norm().item()
    print(f"\nRatio (expected/actual): {ratio:.2f}")

    # Check latent_result after backward
    latent_after = model.model.cache.get_activation('latent_result')
    print(f"\nlatent_result after backward: {latent_after[0, 0, 0].item():.8f}")
    print(f"Same as before? {torch.allclose(latent_result, latent_after)}")

    # Verify with numerical gradient
    print("\n=== Numerical gradient check ===")
    param = model.model.output_projection
    eps = 1e-4

    # First get original loss
    logits_orig = model.model.forward(input_ids_dev)
    shift_logits_orig = logits_orig[..., :-1, :].contiguous().float()
    shift_labels_for_num = labels_dev[..., 1:].contiguous()
    loss_orig = F.cross_entropy(shift_logits_orig.view(-1, 100), shift_labels_for_num.view(-1))
    print(f"Original loss: {loss_orig.item():.8f}")

    # Perturb +eps
    original_val = param[0, 0].item()
    print(f"Original param[0,0]: {original_val:.8f}")
    param[0, 0] = original_val + eps
    print(f"After +eps param[0,0]: {param[0, 0].item():.8f}")
    print(f"Modification worked? {abs(param[0, 0].item() - (original_val + eps)) < 1e-10}")

    # Also check model.model.output_projection directly
    print(f"Direct check model.model.output_projection[0,0]: {model.model.output_projection[0, 0].item():.8f}")

    logits_plus = model.model.forward(input_ids_dev)
    print(f"Logits[0,0,0] after +eps: {logits_plus[0, 0, 0].item():.8f}")
    print(f"Logits_orig[0,0,0]: {logits_orig[0, 0, 0].item():.8f}")
    print(f"Logits changed? {not torch.allclose(logits_plus, logits_orig)}")

    shift_logits_plus = logits_plus[..., :-1, :].contiguous().float()
    loss_plus = F.cross_entropy(shift_logits_plus.view(-1, 100), shift_labels_for_num.view(-1))
    print(f"Loss after +eps: {loss_plus.item():.8f}")

    # Perturb -eps
    param[0, 0] = original_val - eps
    logits_minus = model.model.forward(input_ids_dev)
    shift_logits_minus = logits_minus[..., :-1, :].contiguous().float()
    loss_minus = F.cross_entropy(shift_logits_minus.view(-1, 100), shift_labels_for_num.view(-1))
    print(f"Loss after -eps: {loss_minus.item():.8f}")

    # Restore
    param[0, 0] = original_val

    loss_diff = loss_plus.item() - loss_minus.item()
    print(f"Loss difference (plus - minus): {loss_diff:.10f}")

    num_grad = loss_diff / (2 * eps)
    print(f"\nNumerical gradient[0,0]: {num_grad:.8f}")
    print(f"Expected gradient[0,0]: {expected_grad[0, 0].item():.8f}")
    print(f"Actual gradient[0,0]: {actual_grad[0, 0].item():.8f}")

    if abs(num_grad) > 1e-8 and abs(actual_grad[0, 0].item()) > 1e-8:
        grad_ratio = num_grad / actual_grad[0, 0].item()
        print(f"Ratio (numerical/analytical): {grad_ratio:.4f}")
        rel_error = abs(num_grad - actual_grad[0, 0].item()) / (abs(num_grad) + abs(actual_grad[0, 0].item()))
        print(f"Relative error: {rel_error:.4f}")

    # Check block gradients
    print("\n=== Block gradients ===")
    for block in range(2):
        w1_grad = model.model.cache.gradients.get(('W1', block))
        w2_grad = model.model.cache.gradients.get(('W2', block))
        ffn_gamma = model.model.cache.gradients.get(('ffn_gamma', block))
        attn_gamma = model.model.cache.gradients.get(('attention_gamma', block))
        if w1_grad is not None:
            print(f"Block {block}:")
            print(f"  W1 norm: {w1_grad.norm().item():.6f}")
            print(f"  W2 norm: {w2_grad.norm().item():.6f}")
            if ffn_gamma is not None:
                print(f"  ffn_gamma norm: {ffn_gamma.norm().item():.6f}")
            if attn_gamma is not None:
                print(f"  attn_gamma norm: {attn_gamma.norm().item():.6f}")

    # The test assertion
    assert ratio < 1.5, f"Expected gradient should match actual gradient (ratio={ratio:.2f})"


def test_loss_gradient_computation():
    """
    Debug the loss_gradient computation to understand the numerical mismatch.

    The backward pass computes correctly (latent.T @ loss_grad), but the
    loss_grad we're passing might be wrong.
    """
    config = {
        'vocab_size': 10,  # Smaller for easier debugging
        'max_seq_len': 8,
        'n_blocks': 1,
        'n_heads': 1,
        'd_model': 4,
        'd_ffn': 8,
        'dtype': torch.float32,
    }

    model = CustomTransformerWrapper(**config)

    torch.manual_seed(42)
    input_ids = torch.tensor([[1, 2, 3]])  # 3 positions
    labels = input_ids.clone()

    input_ids_dev = input_ids.to(model.device)
    labels_dev = labels.to(model.device)

    # Forward pass
    logits = model.model.forward(input_ids_dev)
    print(f"Input: {input_ids}")
    print(f"Labels: {labels}")
    print(f"Logits shape: {logits.shape}")

    # After shifting:
    # shift_logits = logits[:, 0:2, :]  (positions 0, 1)
    # shift_labels = labels[:, 1:3]     (values at positions 1, 2)

    batch_size, seq_len = input_ids.shape
    print(f"\nbatch_size={batch_size}, seq_len={seq_len}")
    print(f"After shift: predict {seq_len-1} tokens")

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels_dev[..., 1:].contiguous()
    print(f"shift_logits shape: {shift_logits.shape}")  # [1, 2, 10]
    print(f"shift_labels: {shift_labels}")  # [1, 2] = [[2, 3]]

    # Loss computation
    loss = F.cross_entropy(shift_logits.view(-1, 10), shift_labels.view(-1))
    print(f"\nLoss: {loss.item():.6f}")

    # Our manual gradient computation
    shift_probs = F.softmax(shift_logits, dim=-1)
    shift_one_hot = F.one_hot(shift_labels, num_classes=10).float()
    num_loss_tokens = batch_size * (seq_len - 1)  # 1 * 2 = 2
    print(f"num_loss_tokens: {num_loss_tokens}")

    # Our gradient
    our_shift_gradient = (shift_probs - shift_one_hot) / num_loss_tokens
    print(f"\nOur shift_gradient norm: {our_shift_gradient.norm().item():.6f}")

    # What PyTorch's autograd would give
    shift_logits_with_grad = shift_logits.clone().requires_grad_(True)
    loss_for_autograd = F.cross_entropy(
        shift_logits_with_grad.view(-1, 10),
        shift_labels.view(-1)
    )
    loss_for_autograd.backward()
    pytorch_grad = shift_logits_with_grad.grad
    print(f"PyTorch autograd gradient norm: {pytorch_grad.norm().item():.6f}")

    # Compare
    print(f"\nPer-element comparison (first token, first 5 vocab):")
    print(f"  Our:     {our_shift_gradient[0, 0, :5]}")
    print(f"  PyTorch: {pytorch_grad[0, 0, :5]}")

    # Ratio
    ratio = pytorch_grad.norm() / our_shift_gradient.norm()
    print(f"\nRatio (PyTorch/Ours): {ratio.item():.4f}")

    # If they don't match, the issue is in our manual gradient computation!
    if not torch.allclose(our_shift_gradient, pytorch_grad, rtol=0.01):
        print("\n*** MISMATCH: Our loss gradient doesn't match PyTorch autograd! ***")
        print("This explains the numerical gradient discrepancy.")
    else:
        print("\n*** MATCH: Our loss gradient matches PyTorch autograd ***")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: backward output projection gradient")
    print("=" * 60)
    test_backward_output_projection_gradient()

    print("\n" + "=" * 60)
    print("Test 2: loss gradient computation")
    print("=" * 60)
    test_loss_gradient_computation()
