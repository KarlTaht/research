"""Routing components for learned DAG traversal."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RoutingGate(nn.Module):
    """
    Decides routing weights based on previous routing state and new residual.

    Implements the "path so far + new information" routing decision:
        routing_weights = softmax(gate(gelu(state_proj(prev) + residual_proj(new))))

    Args:
        hidden_dim: Dimension of hidden representations
        n_heads: Number of parallel heads to route between
    """

    def __init__(self, hidden_dim: int, n_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.residual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, n_heads)

        # Initialize gate bias to encourage uniform routing initially
        nn.init.zeros_(self.gate.bias)
        nn.init.xavier_uniform_(self.gate.weight, gain=0.1)

    def forward(self, prev_state: Tensor, residual: Tensor) -> Tensor:
        """
        Compute routing weights.

        Args:
            prev_state: Previous routing state [batch, hidden_dim]
            residual: New residual information [batch, hidden_dim]

        Returns:
            Soft routing weights [batch, n_heads], sums to 1 per sample
        """
        combined = self.state_proj(prev_state) + self.residual_proj(residual)
        routing_logits = self.gate(F.gelu(combined))
        return F.softmax(routing_logits, dim=-1)


class RoutedLayer(nn.Module):
    """
    A layer with multiple parallel MLP heads and soft routing.

    Each head is an independent MLP that transforms the input.
    The routing gate determines how to blend the head outputs.
    The routing state is updated based on the layer output.

    Args:
        hidden_dim: Dimension of hidden representations
        n_heads: Number of parallel MLP heads
        expansion_factor: FFN expansion factor (default 4x like transformers)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Parallel MLP heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * expansion_factor),
                nn.GELU(),
                nn.Linear(hidden_dim * expansion_factor, hidden_dim),
            )
            for _ in range(n_heads)
        ])

        # Routing gate
        self.routing_gate = RoutingGate(hidden_dim, n_heads)

        # State update: combines previous state with new output
        self.state_update = nn.Linear(hidden_dim * 2, hidden_dim)

        # Layer norm for stability
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        routing_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through routed layer.

        Args:
            x: Input tensor [batch, hidden_dim]
            routing_state: Current routing state [batch, hidden_dim]

        Returns:
            output: Transformed output [batch, hidden_dim]
            new_state: Updated routing state [batch, hidden_dim]
            weights: Routing weights used [batch, n_heads]
        """
        # Get routing weights
        weights = self.routing_gate(routing_state, x)  # [batch, n_heads]

        # Compute all head outputs
        head_outputs = torch.stack(
            [head(x) for head in self.heads],
            dim=1,
        )  # [batch, n_heads, hidden_dim]

        # Weighted sum of head outputs
        output = (weights.unsqueeze(-1) * head_outputs).sum(dim=1)  # [batch, hidden_dim]

        # Residual connection and normalization
        output = self.norm(x + output)

        # Update routing state (replace strategy)
        new_state = self.state_update(torch.cat([routing_state, output], dim=-1))

        return output, new_state, weights


class RoutedBlock(nn.Module):
    """
    Alternative implementation that's more memory-efficient for many heads.

    Uses a single larger weight matrix instead of separate head modules.
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        expansion_factor: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.expansion_factor = expansion_factor
        self.intermediate_dim = hidden_dim * expansion_factor

        # Shared first projection (all heads use same up-projection)
        # Then route between different down-projections
        self.up_proj = nn.Linear(hidden_dim, self.intermediate_dim)
        self.down_projs = nn.Linear(self.intermediate_dim, hidden_dim * n_heads)

        self.routing_gate = RoutingGate(hidden_dim, n_heads)
        self.state_update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        routing_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through efficient routed block."""
        batch_size = x.shape[0]

        # Get routing weights
        weights = self.routing_gate(routing_state, x)  # [batch, n_heads]

        # Shared up-projection
        intermediate = F.gelu(self.up_proj(x))  # [batch, intermediate_dim]

        # Multiple down-projections
        all_outputs = self.down_projs(intermediate)  # [batch, hidden_dim * n_heads]
        all_outputs = all_outputs.view(batch_size, self.n_heads, self.hidden_dim)

        # Weighted sum
        output = (weights.unsqueeze(-1) * all_outputs).sum(dim=1)

        # Residual and norm
        output = self.norm(x + output)

        # Update state
        new_state = self.state_update(torch.cat([routing_state, output], dim=-1))

        return output, new_state, weights
