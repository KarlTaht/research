"""Shared pytest fixtures for least_action_learning tests."""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from typing import Generator

import sys
# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from projects.least_action_learning.src.data import (
    ModularArithmeticDataset,
    SequenceArithmeticDataset,
    ModularArithmeticLoader,
    DatasetSplit,
    SequenceDatasetSplit,
)
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
from projects.least_action_learning.src.routing import RoutingGate, RoutedLayer, RoutedBlock
from projects.least_action_learning.src.losses import LeastActionLoss
from projects.least_action_learning.src.trainer import TrainerConfig, Trainer
from projects.least_action_learning.src.metrics import (
    MetricsHistory,
    TrainingMetrics,
    RoutingMetrics,
)


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture
def device() -> torch.device:
    """Get CPU device for testing (consistent across platforms)."""
    return torch.device("cpu")


@pytest.fixture
def cuda_device() -> torch.device:
    """Get CUDA device if available, skip test otherwise."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda")


# ============================================================================
# Seed Fixture
# ============================================================================

@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility in all tests."""
    torch.manual_seed(42)
    yield


# ============================================================================
# Data Fixtures - Small Primes for Fast Testing
# ============================================================================

@pytest.fixture
def small_p() -> int:
    """Small prime for fast testing (p=7)."""
    return 7


@pytest.fixture
def medium_p() -> int:
    """Medium prime for integration tests (p=17)."""
    return 17


@pytest.fixture
def modular_dataset(small_p: int) -> ModularArithmeticDataset:
    """Create small modular arithmetic dataset (add operation)."""
    return ModularArithmeticDataset(
        p=small_p,
        operation="add",
        train_frac=0.3,
        seed=42,
    )


@pytest.fixture
def multiply_dataset(small_p: int) -> ModularArithmeticDataset:
    """Create modular multiplication dataset."""
    return ModularArithmeticDataset(
        p=small_p,
        operation="multiply",
        train_frac=0.3,
        seed=42,
    )


@pytest.fixture
def sequence_dataset(small_p: int) -> SequenceArithmeticDataset:
    """Create sequence-based dataset for transformer."""
    return SequenceArithmeticDataset(
        p=small_p,
        operation="add",
        train_frac=0.3,
        seed=42,
    )


@pytest.fixture
def modular_loader(small_p: int) -> ModularArithmeticLoader:
    """Create modular arithmetic loader with default batch size."""
    return ModularArithmeticLoader(
        p=small_p,
        operation="add",
        train_frac=0.3,
        batch_size=-1,  # Full batch
        seed=42,
    )


# ============================================================================
# Model Configuration Fixtures
# ============================================================================

@pytest.fixture
def tiny_model_config() -> dict:
    """Configuration for tiny test models (fast testing)."""
    return {
        "hidden_dim": 32,
        "n_layers": 2,
        "n_heads": 2,
    }


# ============================================================================
# Model Fixtures - Tiny Models for Fast Testing
# ============================================================================

@pytest.fixture
def baseline_mlp(modular_dataset: ModularArithmeticDataset, tiny_model_config: dict) -> BaselineMLP:
    """Create tiny baseline MLP."""
    return BaselineMLP(
        input_dim=modular_dataset.input_dim,
        hidden_dim=tiny_model_config["hidden_dim"],
        output_dim=modular_dataset.output_dim,
        n_layers=tiny_model_config["n_layers"],
    )


@pytest.fixture
def routed_network(modular_dataset: ModularArithmeticDataset, tiny_model_config: dict) -> RoutedNetwork:
    """Create tiny routed network."""
    return RoutedNetwork(
        input_dim=modular_dataset.input_dim,
        hidden_dim=tiny_model_config["hidden_dim"],
        output_dim=modular_dataset.output_dim,
        n_layers=tiny_model_config["n_layers"],
        n_heads=tiny_model_config["n_heads"],
    )


@pytest.fixture
def single_head_network(modular_dataset: ModularArithmeticDataset, tiny_model_config: dict) -> SingleHeadNetwork:
    """Create single head network (routed with n_heads=1)."""
    return SingleHeadNetwork(
        input_dim=modular_dataset.input_dim,
        hidden_dim=tiny_model_config["hidden_dim"],
        output_dim=modular_dataset.output_dim,
        n_layers=tiny_model_config["n_layers"],
    )


@pytest.fixture
def grok_transformer(sequence_dataset: SequenceArithmeticDataset, tiny_model_config: dict) -> GrokTransformer:
    """Create tiny transformer for grokking."""
    return GrokTransformer(
        vocab_size=sequence_dataset.vocab_size,
        d_model=tiny_model_config["hidden_dim"],
        n_heads=tiny_model_config["n_heads"],
        n_layers=tiny_model_config["n_layers"],
        output_dim=sequence_dataset.output_dim,
        max_seq_len=5,
        dropout=0.0,
    )


# ============================================================================
# Routing Component Fixtures
# ============================================================================

@pytest.fixture
def routing_gate(tiny_model_config: dict) -> RoutingGate:
    """Create routing gate."""
    return RoutingGate(
        hidden_dim=tiny_model_config["hidden_dim"],
        n_heads=tiny_model_config["n_heads"],
    )


@pytest.fixture
def routed_layer(tiny_model_config: dict) -> RoutedLayer:
    """Create routed layer."""
    return RoutedLayer(
        hidden_dim=tiny_model_config["hidden_dim"],
        n_heads=tiny_model_config["n_heads"],
    )


@pytest.fixture
def routed_block(tiny_model_config: dict) -> RoutedBlock:
    """Create efficient routed block."""
    return RoutedBlock(
        hidden_dim=tiny_model_config["hidden_dim"],
        n_heads=tiny_model_config["n_heads"],
    )


# ============================================================================
# Loss Fixtures
# ============================================================================

@pytest.fixture
def least_action_loss() -> LeastActionLoss:
    """Create standard least action loss with entropy regularization."""
    return LeastActionLoss(
        routing_regularizer="entropy",
        lambda_routing=0.01,
        lambda_spectral=0.0,
    )


@pytest.fixture
def least_action_loss_with_spectral(small_p: int) -> LeastActionLoss:
    """Create least action loss with spectral regularization."""
    return LeastActionLoss(
        routing_regularizer="entropy",
        lambda_routing=0.01,
        lambda_spectral=0.1,
        spectral_k=small_p // 4,
        spectral_interval=10,
    )


@pytest.fixture
def task_only_loss() -> LeastActionLoss:
    """Create loss with only task loss (no regularization)."""
    return LeastActionLoss(
        routing_regularizer=None,
        lambda_routing=0.0,
        lambda_spectral=0.0,
    )


# ============================================================================
# Trainer Config Fixtures
# ============================================================================

@pytest.fixture
def trainer_config(small_p: int) -> TrainerConfig:
    """Create minimal trainer config for fast testing (baseline MLP)."""
    return TrainerConfig(
        p=small_p,
        operation="add",
        train_frac=0.3,
        data_seed=42,
        model_type="baseline",
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        epochs=100,  # Very few epochs for testing
        lr=1e-3,
        weight_decay=0.1,
        optimizer="adamw",
        routing_regularizer=None,
        lambda_routing=0.0,
        lambda_spectral=0.0,
        log_every=10,
        save_routing_every=50,
        checkpoint_every=50,
        name="test_run",
        seed=42,
    )


@pytest.fixture
def routed_trainer_config(small_p: int) -> TrainerConfig:
    """Create config for routed network training."""
    return TrainerConfig(
        p=small_p,
        operation="add",
        train_frac=0.3,
        data_seed=42,
        model_type="routed",
        hidden_dim=32,
        n_layers=2,
        n_heads=4,
        epochs=100,
        lr=1e-3,
        weight_decay=0.1,
        optimizer="adamw",
        routing_regularizer="entropy",
        lambda_routing=0.01,
        log_every=10,
        save_routing_every=50,
        checkpoint_every=50,
        name="routed_test",
        seed=42,
    )


@pytest.fixture
def transformer_trainer_config(small_p: int) -> TrainerConfig:
    """Create config for transformer training."""
    return TrainerConfig(
        p=small_p,
        operation="add",
        train_frac=0.5,
        data_seed=42,
        model_type="transformer",
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        epochs=100,
        lr=1e-3,
        weight_decay=1.0,
        optimizer="adamw",
        warmup_epochs=10,
        routing_regularizer=None,
        log_every=10,
        checkpoint_every=50,
        name="transformer_test",
        seed=42,
    )


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def tmp_output_dir() -> Generator[Path, None, None]:
    """Create temporary output directory for tests, cleaned up after."""
    tmp_dir = Path(tempfile.mkdtemp())
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================================
# Sample Tensor Fixtures
# ============================================================================

@pytest.fixture
def sample_batch(modular_dataset: ModularArithmeticDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get sample batch of inputs and targets (16 samples)."""
    train_split = modular_dataset.get_train()
    batch_size = min(16, len(train_split.inputs))
    inputs = train_split.inputs[:batch_size].to(device)
    targets = train_split.targets[:batch_size].to(device)
    return inputs, targets


@pytest.fixture
def sample_sequence_batch(sequence_dataset: SequenceArithmeticDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Get sample sequence batch for transformer (16 samples)."""
    train_split = sequence_dataset.get_train()
    batch_size = min(16, len(train_split.input_ids))
    input_ids = train_split.input_ids[:batch_size].to(device)
    targets = train_split.targets[:batch_size].to(device)
    return input_ids, targets


@pytest.fixture
def sample_routing_weights(tiny_model_config: dict) -> list[torch.Tensor]:
    """Generate sample routing weights for testing (softmax normalized)."""
    batch_size = 16
    n_layers = tiny_model_config["n_layers"]
    n_heads = tiny_model_config["n_heads"]

    weights = []
    for _ in range(n_layers):
        w = torch.softmax(torch.randn(batch_size, n_heads), dim=-1)
        weights.append(w)
    return weights


@pytest.fixture
def uniform_routing_weights(tiny_model_config: dict) -> list[torch.Tensor]:
    """Generate uniform routing weights for testing."""
    batch_size = 16
    n_layers = tiny_model_config["n_layers"]
    n_heads = tiny_model_config["n_heads"]

    weights = []
    for _ in range(n_layers):
        w = torch.full((batch_size, n_heads), 1.0 / n_heads)
        weights.append(w)
    return weights


@pytest.fixture
def one_hot_routing_weights(tiny_model_config: dict) -> list[torch.Tensor]:
    """Generate one-hot routing weights for testing (all weight on first head)."""
    batch_size = 16
    n_layers = tiny_model_config["n_layers"]
    n_heads = tiny_model_config["n_heads"]

    weights = []
    for _ in range(n_layers):
        w = torch.zeros(batch_size, n_heads)
        w[:, 0] = 1.0
        weights.append(w)
    return weights


# ============================================================================
# Metrics Fixtures
# ============================================================================

@pytest.fixture
def sample_training_metrics() -> TrainingMetrics:
    """Create sample training metrics for testing."""
    return TrainingMetrics(
        step=100,
        train_loss=0.5,
        train_acc=0.8,
        test_loss=0.6,
        test_acc=0.7,
        routing_entropy=0.5,
        head_utilization=[0.25, 0.25, 0.25, 0.25],
        spectral_smoothness=0.8,
        total_weight_norm=10.0,
        representation_norm=5.0,
        jacobian_norm=100.0,
        hessian_trace=50.0,
    )


@pytest.fixture
def metrics_history() -> MetricsHistory:
    """Create empty metrics history for testing."""
    return MetricsHistory()
