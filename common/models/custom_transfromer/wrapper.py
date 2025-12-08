"""Wrapper class that adapts CustomTransformer to BaseLanguageModel-like interface."""

from typing import Dict, Optional, Any
import torch
import torch.nn.functional as F

from .CustomTransformer import CustomTransformer
from .config import CustomTransformerConfig


class CustomTransformerWrapper:
    """
    Wrapper that adapts CustomTransformer to BaseLanguageModel-like interface.

    Note: This is NOT an nn.Module because CustomTransformer uses manual backprop
    with raw tensors instead of nn.Parameter. Use train_step() for training
    instead of the typical loss.backward() + optimizer.step() pattern.

    Example:
        model = CustomTransformerWrapper(vocab_size=50257, d_model=256)

        # For inference:
        outputs = model.forward(input_ids, labels=labels)
        loss = outputs['loss']

        # For training (manual backprop):
        result = model.train_step(input_ids, labels, learning_rate=0.001)
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 128,
        n_blocks: int = 8,
        n_heads: int = 4,
        d_model: int = 128,
        d_ffn: int = 128,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        pad_token_id: Optional[int] = None,
    ):
        """
        Initialize CustomTransformerWrapper.

        Args:
            vocab_size: Size of the vocabulary
            max_seq_len: Maximum sequence length
            n_blocks: Number of transformer blocks
            n_heads: Number of attention heads
            d_model: Model dimension
            d_ffn: Feed-forward network dimension
            device: Device to use (auto-detected if None)
            dtype: Data type (default: bfloat16)
            pad_token_id: Padding token ID (if provided, padding tokens are masked in loss)
        """
        self.vocab_size = vocab_size
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        config = CustomTransformerConfig(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            n_blocks=n_blocks,
            n_heads=n_heads,
            d_model=d_model,
            d_ffn=d_ffn,
            device=device,
            dtype=dtype,
        )
        self.model = CustomTransformer(config)
        self.device = self.model.device
        self.dtype = self.model.dtype

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass compatible with BaseLanguageModel interface.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target token IDs for computing loss [batch_size, seq_len]

        Returns:
            Dict containing:
                - 'logits': Model output logits [batch_size, seq_len, vocab_size]
                - 'loss': Computed loss (if labels provided)
        """
        # Move to device if needed
        input_ids = input_ids.to(self.device)

        # Get logits from CustomTransformer
        logits = self.model.forward(input_ids)  # [batch, seq, vocab]

        output = {'logits': logits}

        if labels is not None:
            labels = labels.to(self.device)

            # Dynamic padding mask - ignore padding tokens in loss
            if self.pad_token_id is not None:
                labels = labels.clone()
                labels[labels == self.pad_token_id] = -100

            # PRECISION MIXING: Compute loss in float32 for numerical stability
            logits_for_loss = logits
            if self.dtype == torch.bfloat16:
                logits_for_loss = logits.to(torch.float32)

            # CAUSAL LM: Shift so logits[i] predicts labels[i+1]
            shift_logits = logits_for_loss[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # Ignore masked padding tokens
            )
            output['loss'] = loss

        return output

    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Complete training step with manual backprop and numerical stability.

        This replaces the typical PyTorch pattern:
            outputs = model(input_ids, labels=labels)
            loss.backward()
            optimizer.step()

        Includes:
        - Float32 loss computation for numerical stability
        - NaN/Inf detection with skip-update recovery
        - Gradient clipping before parameter update

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            labels: Target token IDs [batch_size, seq_len]
            learning_rate: Learning rate for SGD update
            max_grad_norm: Maximum gradient norm for clipping

        Returns:
            Dict with:
                - 'loss': Loss value (float)
                - 'status': 'ok', 'nan_detected', or 'nan_gradient'
                - 'grad_norm': Gradient norm before clipping (if status='ok')
        """
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        # Dynamic padding mask - ignore padding tokens in loss
        if self.pad_token_id is not None:
            labels = labels.clone()
            labels[labels == self.pad_token_id] = -100

        # Forward pass
        logits = self.model.forward(input_ids)

        # PRECISION MIXING: Compute loss in float32 for numerical stability
        batch_size, seq_len = input_ids.shape
        logits_fp32 = logits.to(torch.float32) if self.dtype == torch.bfloat16 else logits

        # CAUSAL LM: Shift so logits[i] predicts labels[i+1]
        # logits: [batch, seq, vocab] -> shift to [batch, seq-1, vocab]
        # labels: [batch, seq] -> shift to [batch, seq-1]
        shift_logits = logits_fp32[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Create mask for valid (non-padding) positions
        valid_mask = (shift_labels != -100)  # [batch, seq-1]
        num_valid_tokens = valid_mask.sum().item()

        # For loss computation, temporarily replace -100 with 0 (valid index)
        shift_labels_for_loss = shift_labels.clone()
        shift_labels_for_loss[~valid_mask] = 0

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # NaN/Inf detection in loss
        if torch.isnan(loss) or torch.isinf(loss):
            return {
                'loss': float('nan'),
                'status': 'nan_detected',
                'grad_norm': None,
            }

        # Compute loss gradient for backprop in float32, then cast back
        # Cross-entropy gradient: softmax(logits) - one_hot(targets)
        # Must use SHIFTED logits/labels to match the loss computation
        shift_probs = F.softmax(shift_logits, dim=-1)

        # Use valid labels for one_hot (replace -100 with 0 temporarily)
        shift_one_hot = F.one_hot(shift_labels_for_loss, num_classes=self.vocab_size).to(torch.float32)

        # Gradient for shifted positions, normalized by number of VALID tokens
        # Zero out gradient for padding positions
        shift_gradient = (shift_probs - shift_one_hot)
        shift_gradient = shift_gradient * valid_mask.unsqueeze(-1).float()  # Zero out padding
        if num_valid_tokens > 0:
            shift_gradient = shift_gradient / num_valid_tokens

        # Pad gradient back to full sequence length (last position has no loss)
        loss_gradient = torch.zeros(batch_size, seq_len, self.vocab_size,
                                    dtype=torch.float32, device=self.device)
        loss_gradient[:, :-1, :] = shift_gradient

        # Cast gradient back to model dtype for backward pass
        if self.dtype == torch.bfloat16:
            loss_gradient = loss_gradient.to(torch.bfloat16)

        # Manual backward pass
        self.model.backward(loss_gradient)

        # Check for NaN/Inf in gradients before update
        if self._has_nan_gradients():
            return {
                'loss': loss.item(),
                'status': 'nan_gradient',
                'grad_norm': None,
            }

        # Gradient clipping
        grad_norm_before, grad_norm_after = self._clip_gradients(max_grad_norm)

        # Manual parameter update
        self.model.update_parameters(learning_rate)

        return {
            'loss': loss.item(),
            'status': 'ok',
            'grad_norm': grad_norm_after,
            'grad_norm_before_clip': grad_norm_before,
        }

    def _has_nan_gradients(self) -> bool:
        """Check all cached gradients for NaN or Inf values."""
        for key, grad in self.model.cache.gradients.items():
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                return True
        return False

    def _clip_gradients(self, max_norm: float) -> tuple:
        """
        Clip gradients by global norm (manual implementation for raw tensors).

        Args:
            max_norm: Maximum allowed gradient norm

        Returns:
            Tuple of (grad_norm_before_clip, grad_norm_after_clip)
        """
        # Collect all gradients
        all_grads = list(self.model.cache.gradients.values())

        if not all_grads:
            return 0.0, 0.0

        # Compute total norm (L2 norm across all gradients)
        total_norm_sq = sum(g.pow(2).sum() for g in all_grads)
        total_norm = torch.sqrt(total_norm_sq).item()
        grad_norm_before = total_norm

        # Clip if norm exceeds max_norm
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for key in self.model.cache.gradients:
                self.model.cache.gradients[key] = self.model.cache.gradients[key] * clip_coef
            grad_norm_after = max_norm
        else:
            grad_norm_after = total_norm

        return grad_norm_before, grad_norm_after

    def eval(self):
        """Set to evaluation mode (no-op for manual backprop model)."""
        pass

    def train(self):
        """Set to training mode (no-op for manual backprop model)."""
        pass

    def to(self, device):
        """Move model to device (handled in __init__)."""
        return self

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            eos_token_id: If set, stop generation when this token is produced

        Returns:
            Generated token IDs [batch_size, generated_length]
        """
        input_ids = input_ids.to(self.device)
        generated = input_ids

        for _ in range(max_length - input_ids.size(1)):
            # Check max_seq_len constraint
            if generated.size(1) >= self.model.max_seq_len:
                break

            outputs = self.forward(generated)
            logits = outputs['logits']

            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k)
                next_token_logits[next_token_logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs.float(), num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return generated

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for logging."""
        return {
            'model_type': 'CustomTransformer',
            'vocab_size': self.vocab_size,
            'd_model': self.model.d_model,
            'n_blocks': self.model.n_blocks,
            'n_heads': self.model.n_heads,
            'd_ffn': self.model.d_ffn,
            'max_seq_len': self.model.max_seq_len,
            'parameters': self.count_parameters(),
        }

    def count_parameters(self) -> int:
        """Count total parameters."""
        total = 0
        # Embeddings
        total += self.model.vocab_embedding.numel()
        total += self.model.pos_embedding.numel()
        # Attention per block
        total += self.model.Q.numel()
        total += self.model.K.numel()
        total += self.model.V.numel()
        total += self.model.W_o.numel()
        # FFN per block
        total += self.model.W1.numel()
        total += self.model.W2.numel()
        # Layer norms
        total += self.model.attention_gamma.numel()
        total += self.model.attention_beta.numel()
        total += self.model.ffn_gamma.numel()
        total += self.model.ffn_beta.numel()
        # Output projection
        total += self.model.output_projection.numel()
        return total

    def save_checkpoint(self, path: str, **kwargs):
        """
        Save model state.

        Args:
            path: Path to save checkpoint
            **kwargs: Additional metadata to save (epoch, loss, etc.)
        """
        self.model.save_checkpoint(path, **kwargs)

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load model state from checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Dict with checkpoint metadata
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Handle multiple checkpoint formats:
        # 1. CheckpointManager format: 'model_state_dict'
        # 2. CustomTransformer.save_checkpoint format: 'state_dict'
        # 3. Legacy format: direct tensor storage
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Checkpoint loaded: {path}")
            return checkpoint.get('model_config', {})
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"Checkpoint loaded: {path}")
            return checkpoint.get('metadata', {})
        else:
            # Legacy format - direct tensor storage
            legacy_state = {k: v for k, v in checkpoint.items()
                          if k not in ('config', 'metadata')}
            self.model.load_state_dict(legacy_state, strict=False)
            print(f"Checkpoint loaded: {path}")
            return checkpoint.get('config', {})

    @classmethod
    def from_checkpoint(cls, path: str, device: str = None) -> 'CustomTransformerWrapper':
        """
        Create wrapper from checkpoint file.

        Args:
            path: Path to checkpoint file
            device: Device to load model onto

        Returns:
            CustomTransformerWrapper instance

        Example:
            model = CustomTransformerWrapper.from_checkpoint('checkpoint.pt')
        """
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)

        # Get config from checkpoint (handle multiple formats)
        config = checkpoint.get('model_config') or checkpoint.get('config', {})

        # Parse dtype from string if present
        dtype = None
        dtype_str = config.get('dtype')
        if dtype_str:
            dtype_map = {
                'torch.float32': torch.float32,
                'torch.float16': torch.float16,
                'torch.bfloat16': torch.bfloat16,
                'torch.float64': torch.float64,
            }
            dtype = dtype_map.get(dtype_str)

        # Create wrapper with saved config
        wrapper = cls(
            vocab_size=config.get('vocab_size', 128),
            max_seq_len=config.get('max_seq_len', 128),
            n_blocks=config.get('n_blocks', 8),
            n_heads=config.get('n_heads', 4),
            d_model=config.get('d_model', 128),
            d_ffn=config.get('d_ffn', 128),
            device=device or config.get('device'),
            dtype=dtype,
        )

        # Load state (handle multiple formats)
        if 'model_state_dict' in checkpoint:
            wrapper.model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            wrapper.model.load_state_dict(checkpoint['state_dict'])
        else:
            # Legacy format
            legacy_state = {k: v for k, v in checkpoint.items()
                          if k not in ('config', 'metadata', 'model_config')}
            wrapper.model.load_state_dict(legacy_state, strict=False)

        print(f"Model loaded from checkpoint: {path}")
        return wrapper
