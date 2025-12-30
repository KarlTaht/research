"""Neural network models for least action learning experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .routing import RoutedLayer, RoutedBlock
from .metrics import RoutingMetrics, compute_routing_entropy, compute_head_utilization


class BaselineMLP(nn.Module):
    """
    Standard MLP without routing for baseline comparison.

    This replicates the standard grokking experiment setup.

    Args:
        input_dim: Input dimension (2*p for one-hot encoded pairs)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (p classes)
        n_layers: Number of hidden layers
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        # Build network
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning logits."""
        return self.net(x)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class RoutedNetwork(nn.Module):
    """
    Network with learned routing through parallel MLP heads.

    At each layer, a routing gate determines how to blend multiple
    parallel MLP heads based on the routing state (path history)
    and the current residual (new information).

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        n_layers: Number of routed layers
        n_heads: Number of parallel heads per layer
        use_efficient_block: Use memory-efficient RoutedBlock instead of RoutedLayer
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        n_heads: int = 4,
        use_efficient_block: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Input embedding
        self.embed = nn.Linear(input_dim, hidden_dim)

        # Initialize routing state from input
        self.state_init = nn.Linear(hidden_dim, hidden_dim)

        # Routed layers
        LayerClass = RoutedBlock if use_efficient_block else RoutedLayer
        self.layers = nn.ModuleList([
            LayerClass(hidden_dim, n_heads) for _ in range(n_layers)
        ])

        # Output head
        self.output_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, RoutingMetrics]:
        """
        Forward pass through routed network.

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            logits: Output logits [batch, output_dim]
            metrics: Routing metrics for this forward pass
        """
        # Embed input
        h = self.embed(x)

        # Initialize routing state from embedded input
        routing_state = self.state_init(h)

        # Pass through routed layers
        all_weights = []
        for layer in self.layers:
            h, routing_state, weights = layer(h, routing_state)
            all_weights.append(weights)

        # Output projection
        logits = self.output_head(h)

        # Compute routing metrics
        metrics = RoutingMetrics(
            layer_weights=all_weights,
            routing_entropy=compute_routing_entropy(torch.stack(all_weights)),
            head_utilization=compute_head_utilization(all_weights),
        )

        return logits, metrics

    def forward_with_routing_trace(
        self,
        x: Tensor,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """
        Forward pass that returns full routing trace for analysis.

        Returns:
            logits: Output logits
            layer_outputs: List of hidden states after each layer
            layer_weights: List of routing weights at each layer
        """
        h = self.embed(x)
        routing_state = self.state_init(h)

        layer_outputs = [h.detach().clone()]
        layer_weights = []

        for layer in self.layers:
            h, routing_state, weights = layer(h, routing_state)
            layer_outputs.append(h.detach().clone())
            layer_weights.append(weights.detach().clone())

        logits = self.output_head(h)
        return logits, layer_outputs, layer_weights

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        """Get model configuration."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_parameters": self.count_parameters(),
        }


class SingleHeadNetwork(RoutedNetwork):
    """
    RoutedNetwork with n_heads=1 for ablation.

    This is equivalent to a standard MLP but uses the same
    architecture as RoutedNetwork for fair comparison.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_heads=1,  # Single head = no routing choice
        )


class GrokTransformer(nn.Module):
    """
    Decoder-only transformer for grokking experiments.

    Matches the original grokking paper setup:
    - Input: [a, op, b, =] token sequence
    - Output: logits for predicting result at the last position
    - Causal attention masking

    Args:
        vocab_size: Vocabulary size (p + 2 for residues + op + equals)
        d_model: Model/embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        output_dim: Output dimension (p classes for result)
        max_seq_len: Maximum sequence length (default 5 for [a,op,b,=,result])
        dropout: Dropout probability (default 0.0 for grokking)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        output_dim: int,
        max_seq_len: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, d_model * 4, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output head (predicts result class, not next token)
        self.output_head = nn.Linear(d_model, output_dim)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        init_std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            logits: Prediction logits at last position [batch, output_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Causal mask for this sequence length
        mask = self.causal_mask[:seq_len, :seq_len]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_f(x)

        # Get logits from last position only
        last_hidden = x[:, -1, :]  # [batch, d_model]
        logits = self.output_head(last_hidden)  # [batch, output_dim]

        return logits

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    """Single transformer block with attention and feedforward."""

    def __init__(self, d_model: int, d_ffn: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ffn, dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.ln1(x), mask)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V and reshape for multi-head
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores + mask  # Apply causal mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ V

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class FeedForward(nn.Module):
    """Feedforward network with GELU activation."""

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn)
        self.w2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.dropout(F.gelu(self.w1(x))))


def create_model(
    model_type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    n_layers: int,
    n_heads: int = 4,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create models by name.

    Args:
        model_type: One of "baseline", "routed", "single_head", "transformer"
        input_dim: Input dimension (vocab_size for transformer)
        hidden_dim: Hidden dimension (d_model for transformer)
        output_dim: Output dimension
        n_layers: Number of layers
        n_heads: Number of heads (for routed/transformer models)
        **kwargs: Additional arguments (e.g., max_seq_len for transformer)

    Returns:
        Instantiated model
    """
    if model_type == "baseline":
        return BaselineMLP(input_dim, hidden_dim, output_dim, n_layers)
    elif model_type == "routed":
        return RoutedNetwork(input_dim, hidden_dim, output_dim, n_layers, n_heads)
    elif model_type == "single_head":
        return SingleHeadNetwork(input_dim, hidden_dim, output_dim, n_layers)
    elif model_type == "transformer":
        return GrokTransformer(
            vocab_size=input_dim,  # For transformer, input_dim is vocab_size
            d_model=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            output_dim=output_dim,
            max_seq_len=kwargs.get("max_seq_len", 5),
            dropout=kwargs.get("dropout", 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
