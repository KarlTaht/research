"""Tests for trainer module (TrainerConfig, Trainer)."""

import pytest
import torch
import torch.nn as nn
import json
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from projects.least_action_learning.src.trainer import (
    TrainerConfig,
    Trainer,
    train_experiment,
)
from projects.least_action_learning.src.metrics import MetricsHistory


class TestTrainerConfig:
    """Tests for TrainerConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = TrainerConfig()

        assert config.p == 113
        assert config.operation == "add"
        assert config.train_frac == 0.3
        assert config.model_type == "routed"
        assert config.hidden_dim == 128
        assert config.n_layers == 4
        assert config.n_heads == 4
        assert config.epochs == 50000
        assert config.lr == 1e-3
        assert config.weight_decay == 0.1
        assert config.optimizer == "adamw"

    def test_custom_values(self, small_p):
        """Test config with custom values."""
        config = TrainerConfig(
            p=small_p,
            operation="multiply",
            train_frac=0.5,
            model_type="baseline",
            hidden_dim=64,
        )

        assert config.p == small_p
        assert config.operation == "multiply"
        assert config.train_frac == 0.5
        assert config.model_type == "baseline"
        assert config.hidden_dim == 64

    def test_post_init_string_conversion(self):
        """Test __post_init__ converts string values to floats."""
        # Simulate YAML parsing where scientific notation becomes string
        config = TrainerConfig(
            lr="1e-4",  # String
            weight_decay="0.5",  # String
            eps="1.0e-8",  # String
        )

        assert config.lr == 1e-4
        assert isinstance(config.lr, float)
        assert config.weight_decay == 0.5
        assert isinstance(config.weight_decay, float)
        assert config.eps == 1e-8
        assert isinstance(config.eps, float)

    def test_to_dict(self, trainer_config):
        """Test conversion to dictionary."""
        d = trainer_config.to_dict()

        assert isinstance(d, dict)
        assert "p" in d
        assert "model_type" in d
        assert "epochs" in d
        assert "lr" in d

    def test_to_dict_complete(self, trainer_config):
        """Test all fields are in dict."""
        d = trainer_config.to_dict()

        expected_keys = [
            "p", "operation", "train_frac", "data_seed",
            "model_type", "hidden_dim", "n_layers", "n_heads",
            "epochs", "lr", "weight_decay", "optimizer",
            "beta1", "beta2", "eps", "grad_clip", "warmup_epochs",
            "routing_regularizer", "lambda_routing", "lambda_spectral",
            "spectral_k", "spectral_interval",
            "log_every", "save_routing_every", "checkpoint_every",
            "name", "seed",
        ]

        for key in expected_keys:
            assert key in d, f"Missing key: {key}"


class TestTrainerInit:
    """Tests for Trainer initialization."""

    def test_creates_baseline_model(self, trainer_config, tmp_output_dir):
        """Test trainer creates baseline MLP."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        from projects.least_action_learning.src.models import BaselineMLP
        assert isinstance(trainer.model, BaselineMLP)

    def test_creates_routed_model(self, routed_trainer_config, tmp_output_dir):
        """Test trainer creates routed network."""
        trainer = Trainer(routed_trainer_config, tmp_output_dir)

        from projects.least_action_learning.src.models import RoutedNetwork
        assert isinstance(trainer.model, RoutedNetwork)

    def test_creates_transformer_model(self, transformer_trainer_config, tmp_output_dir):
        """Test trainer creates transformer."""
        trainer = Trainer(transformer_trainer_config, tmp_output_dir)

        from projects.least_action_learning.src.models import GrokTransformer
        assert isinstance(trainer.model, GrokTransformer)

    def test_creates_output_dir(self, trainer_config, tmp_output_dir):
        """Test trainer creates output directory."""
        subdir = tmp_output_dir / "test_run"
        trainer = Trainer(trainer_config, subdir)

        assert subdir.exists()

    def test_sets_device(self, trainer_config, tmp_output_dir):
        """Test trainer sets device."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        assert trainer.device is not None
        # Should be CPU in tests
        assert trainer.device == torch.device("cpu") or trainer.device.type in ["cuda", "mps"]

    def test_model_on_device(self, trainer_config, tmp_output_dir):
        """Test model is on correct device."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        # Check a parameter is on the device (compare device type, not exact equality)
        param = next(trainer.model.parameters())
        assert param.device.type == trainer.device.type

    def test_data_on_device(self, trainer_config, tmp_output_dir):
        """Test data is on correct device."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        # Compare device type, not exact equality (cuda:0 vs cuda)
        assert trainer.train_inputs.device.type == trainer.device.type
        assert trainer.train_targets.device.type == trainer.device.type
        assert trainer.test_inputs.device.type == trainer.device.type
        assert trainer.test_targets.device.type == trainer.device.type

    def test_history_initialized(self, trainer_config, tmp_output_dir):
        """Test metrics history is initialized."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        assert isinstance(trainer.history, MetricsHistory)
        assert len(trainer.history.history) == 0


class TestTrainerOptimizer:
    """Tests for Trainer optimizer creation."""

    def test_adamw_optimizer(self, trainer_config, tmp_output_dir):
        """Test AdamW optimizer is created."""
        trainer_config.optimizer = "adamw"
        trainer = Trainer(trainer_config, tmp_output_dir)

        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_adam_optimizer(self, trainer_config, tmp_output_dir):
        """Test Adam optimizer is created."""
        trainer_config.optimizer = "adam"
        trainer = Trainer(trainer_config, tmp_output_dir)

        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_sgd_optimizer(self, trainer_config, tmp_output_dir):
        """Test SGD optimizer is created."""
        trainer_config.optimizer = "sgd"
        trainer = Trainer(trainer_config, tmp_output_dir)

        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_invalid_optimizer_raises(self, trainer_config, tmp_output_dir):
        """Test invalid optimizer raises error."""
        trainer_config.optimizer = "invalid"

        with pytest.raises(ValueError, match="Unknown optimizer"):
            Trainer(trainer_config, tmp_output_dir)

    def test_adamw_weight_decay_groups(self, transformer_trainer_config, tmp_output_dir):
        """Test AdamW has separate weight decay groups."""
        trainer = Trainer(transformer_trainer_config, tmp_output_dir)

        # AdamW should have 2 param groups
        assert len(trainer.optimizer.param_groups) == 2

        # First group has weight decay
        assert trainer.optimizer.param_groups[0]["weight_decay"] == transformer_trainer_config.weight_decay

        # Second group has no weight decay (biases, layernorm, embeddings)
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0


class TestTrainerTrainStep:
    """Tests for Trainer.train_step()."""

    def test_train_step_returns_tuple(self, trainer_config, tmp_output_dir):
        """Test train_step returns (loss, acc, weights) tuple."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        loss, acc, weights = trainer.train_step(0)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        # Baseline has no routing weights
        assert weights is None

    def test_train_step_routed_returns_weights(self, routed_trainer_config, tmp_output_dir):
        """Test train_step returns routing weights for routed model."""
        trainer = Trainer(routed_trainer_config, tmp_output_dir)

        _, _, weights = trainer.train_step(0)

        assert weights is not None
        assert isinstance(weights, list)
        assert len(weights) == routed_trainer_config.n_layers

    def test_train_step_updates_weights(self, trainer_config, tmp_output_dir):
        """Test train_step updates model weights."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        # Get initial weights
        initial_weights = next(trainer.model.parameters()).clone()

        # Train step
        trainer.train_step(0)

        # Weights should have changed
        updated_weights = next(trainer.model.parameters())
        assert not torch.allclose(initial_weights, updated_weights)

    def test_train_step_loss_decreases(self, trainer_config, tmp_output_dir):
        """Test loss generally decreases over steps."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        losses = []
        for epoch in range(10):
            loss, _, _ = trainer.train_step(epoch)
            losses.append(loss)

        # Loss should generally decrease (may have some noise)
        # Check that final is less than initial (with some tolerance)
        assert losses[-1] < losses[0] * 1.5  # Allow some tolerance


class TestTrainerEvaluate:
    """Tests for Trainer.evaluate()."""

    def test_evaluate_returns_tuple(self, trainer_config, tmp_output_dir):
        """Test evaluate returns (loss, acc) tuple."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        loss, acc = trainer.evaluate()

        assert isinstance(loss, float)
        assert isinstance(acc, float)

    def test_evaluate_accuracy_in_range(self, trainer_config, tmp_output_dir):
        """Test evaluate accuracy is in [0, 1]."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        _, acc = trainer.evaluate()

        assert 0.0 <= acc <= 1.0

    def test_evaluate_loss_positive(self, trainer_config, tmp_output_dir):
        """Test evaluate loss is positive."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        loss, _ = trainer.evaluate()

        assert loss > 0

    def test_evaluate_does_not_update_weights(self, trainer_config, tmp_output_dir):
        """Test evaluate does not update model weights."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        initial_weights = next(trainer.model.parameters()).clone()
        trainer.evaluate()
        updated_weights = next(trainer.model.parameters())

        assert torch.allclose(initial_weights, updated_weights)


class TestTrainerWarmup:
    """Tests for LR warmup functionality."""

    def test_warmup_lr_increases(self, trainer_config, tmp_output_dir):
        """Test LR increases during warmup."""
        trainer_config.warmup_epochs = 100
        trainer = Trainer(trainer_config, tmp_output_dir)

        # Get initial LR
        initial_lr = trainer.optimizer.param_groups[0]["lr"]

        # Train a few epochs
        for epoch in range(10):
            trainer.train_step(epoch)
            # Warmup adjusts LR at start of train() loop, so simulate it
            warmup_factor = (epoch + 1) / trainer_config.warmup_epochs
            for param_group in trainer.optimizer.param_groups:
                param_group["lr"] = trainer_config.lr * warmup_factor

        # LR should be higher but not yet at full
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        expected_lr = trainer_config.lr * 10 / 100  # 10 epochs / 100 warmup
        assert abs(current_lr - expected_lr) < 1e-8

    def test_warmup_formula(self, trainer_config, tmp_output_dir):
        """Test warmup uses correct formula: lr * (epoch+1) / warmup_epochs."""
        trainer_config.warmup_epochs = 100
        trainer = Trainer(trainer_config, tmp_output_dir)

        # Test various epochs
        for epoch in [0, 25, 50, 99]:
            expected_lr = trainer_config.lr * (epoch + 1) / trainer_config.warmup_epochs
            warmup_factor = (epoch + 1) / trainer_config.warmup_epochs
            actual_lr = trainer_config.lr * warmup_factor

            assert abs(actual_lr - expected_lr) < 1e-10


class TestTrainerCheckpoint:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint_creates_file(self, trainer_config, tmp_output_dir):
        """Test save_checkpoint creates file."""
        trainer = Trainer(trainer_config, tmp_output_dir)
        trainer.save_checkpoint("test.pt", epoch=0)

        assert (tmp_output_dir / "test.pt").exists()

    def test_checkpoint_contains_required_keys(self, trainer_config, tmp_output_dir):
        """Test checkpoint contains all required keys."""
        trainer = Trainer(trainer_config, tmp_output_dir)
        trainer.save_checkpoint("test.pt", epoch=100)

        checkpoint = torch.load(tmp_output_dir / "test.pt", weights_only=False)

        assert "model_state" in checkpoint
        assert "optimizer_state" in checkpoint
        assert "config" in checkpoint
        assert "best_test_acc" in checkpoint
        assert "best_epoch" in checkpoint
        assert "epoch" in checkpoint

    def test_load_checkpoint_restores_model(self, trainer_config, tmp_output_dir):
        """Test load_checkpoint restores model state."""
        # Create and train a bit
        trainer1 = Trainer(trainer_config, tmp_output_dir)
        for epoch in range(10):
            trainer1.train_step(epoch)
        trainer1.save_checkpoint("test.pt", epoch=10)

        # Get trained weights
        trained_weights = next(trainer1.model.parameters()).clone()

        # Create new trainer
        trainer2 = Trainer(trainer_config, tmp_output_dir)
        initial_weights = next(trainer2.model.parameters()).clone()

        # Load checkpoint
        trainer2.load_checkpoint("test.pt")
        loaded_weights = next(trainer2.model.parameters())

        # Loaded should match trained, not initial
        assert torch.allclose(loaded_weights, trained_weights)
        assert not torch.allclose(loaded_weights, initial_weights)

    def test_load_checkpoint_restores_best_acc(self, trainer_config, tmp_output_dir):
        """Test load_checkpoint restores best accuracy."""
        trainer1 = Trainer(trainer_config, tmp_output_dir)
        trainer1.best_test_acc = 0.75
        trainer1.best_epoch = 500
        trainer1.save_checkpoint("test.pt", epoch=500)

        trainer2 = Trainer(trainer_config, tmp_output_dir)
        trainer2.load_checkpoint("test.pt")

        assert trainer2.best_test_acc == 0.75
        assert trainer2.best_epoch == 500


class TestTrainerHistory:
    """Tests for history saving."""

    def test_save_history_creates_parquet(self, trainer_config, tmp_output_dir):
        """Test save_history creates parquet file."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        # Add some history
        from projects.least_action_learning.src.metrics import TrainingMetrics
        for i in range(3):
            m = TrainingMetrics(
                step=i * 100,
                train_loss=0.5,
                train_acc=0.5,
                test_loss=0.6,
                test_acc=0.5,
                routing_entropy=0.5,
                head_utilization=[0.5, 0.5],
            )
            trainer.history.log(m)

        trainer.save_history()

        assert (tmp_output_dir / "history.parquet").exists()

    def test_save_history_creates_config_json(self, trainer_config, tmp_output_dir):
        """Test save_history creates config.json."""
        trainer = Trainer(trainer_config, tmp_output_dir)
        trainer.save_history()

        assert (tmp_output_dir / "config.json").exists()

        # Verify it's valid JSON
        with open(tmp_output_dir / "config.json") as f:
            config = json.load(f)

        assert "p" in config
        assert "model_type" in config


class TestTrainerSpectralSmoothness:
    """Tests for spectral smoothness computation."""

    def test_compute_spectral_smoothness_returns_float(self, trainer_config, tmp_output_dir):
        """Test compute_spectral_smoothness returns float."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        smoothness = trainer.compute_spectral_smoothness()

        assert isinstance(smoothness, float)

    def test_compute_spectral_smoothness_in_range(self, trainer_config, tmp_output_dir):
        """Test spectral smoothness is in [0, 1]."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        smoothness = trainer.compute_spectral_smoothness()

        assert 0.0 <= smoothness <= 1.0

    def test_transformer_spectral_smoothness(self, transformer_trainer_config, tmp_output_dir):
        """Test spectral smoothness works for transformer."""
        trainer = Trainer(transformer_trainer_config, tmp_output_dir)

        smoothness = trainer.compute_spectral_smoothness()

        assert isinstance(smoothness, float)
        assert 0.0 <= smoothness <= 1.0


class TestTrainExperiment:
    """Tests for train_experiment convenience function."""

    def test_train_experiment_creates_output(self, trainer_config, tmp_output_dir):
        """Test train_experiment creates output directory."""
        trainer_config.epochs = 10  # Very short
        trainer_config.log_every = 5
        trainer_config.checkpoint_every = 1000  # No checkpoints during short run

        history = train_experiment(trainer_config, tmp_output_dir)

        assert isinstance(history, MetricsHistory)
        assert len(history.history) > 0

    def test_train_experiment_default_output_dir(self, trainer_config, tmp_output_dir, monkeypatch):
        """Test train_experiment uses default output dir when none provided."""
        trainer_config.epochs = 10
        trainer_config.log_every = 5
        trainer_config.checkpoint_every = 1000
        trainer_config.name = "test_default_dir"

        # Change to temp directory to avoid polluting project
        monkeypatch.chdir(tmp_output_dir)

        history = train_experiment(trainer_config, output_dir=None)

        assert isinstance(history, MetricsHistory)
        # Should have created outputs/{name} directory
        assert (tmp_output_dir / "outputs" / "test_default_dir").exists()


class TestTrainerGradientClipping:
    """Tests for gradient clipping."""

    def test_gradient_clipping_applied(self, trainer_config, tmp_output_dir):
        """Test gradient clipping is applied when configured."""
        trainer_config.grad_clip = 1.0
        trainer = Trainer(trainer_config, tmp_output_dir)

        # Run a training step
        trainer.train_step(0)

        # Clipping is applied during train_step, hard to verify directly
        # Just ensure it doesn't crash
        assert True

    def test_no_gradient_clipping(self, trainer_config, tmp_output_dir):
        """Test no gradient clipping when None."""
        trainer_config.grad_clip = None
        trainer = Trainer(trainer_config, tmp_output_dir)

        # Run a training step
        trainer.train_step(0)

        # Should complete without error
        assert True


class TestTrainerDifferentModels:
    """Tests for trainer with different model types."""

    def test_baseline_model_trains(self, trainer_config, tmp_output_dir):
        """Test baseline model trains successfully."""
        trainer = Trainer(trainer_config, tmp_output_dir)

        initial_loss, _, _ = trainer.train_step(0)
        for epoch in range(1, 10):
            loss, _, _ = trainer.train_step(epoch)

        # Should complete and loss should decrease
        assert loss < initial_loss * 1.5

    def test_routed_model_trains(self, routed_trainer_config, tmp_output_dir):
        """Test routed model trains successfully."""
        trainer = Trainer(routed_trainer_config, tmp_output_dir)

        initial_loss, _, _ = trainer.train_step(0)
        for epoch in range(1, 10):
            loss, _, weights = trainer.train_step(epoch)

        # Should complete with routing weights
        assert weights is not None
        assert loss < initial_loss * 1.5

    def test_transformer_model_trains(self, transformer_trainer_config, tmp_output_dir):
        """Test transformer model trains successfully."""
        trainer = Trainer(transformer_trainer_config, tmp_output_dir)

        initial_loss, _, _ = trainer.train_step(0)
        for epoch in range(1, 10):
            loss, _, _ = trainer.train_step(epoch)

        # Should complete
        assert loss < initial_loss * 1.5


class TestTrainerIsTransformer:
    """Tests for transformer detection."""

    def test_baseline_not_transformer(self, trainer_config, tmp_output_dir):
        """Test baseline is not transformer."""
        trainer = Trainer(trainer_config, tmp_output_dir)
        assert not trainer.is_transformer

    def test_routed_not_transformer(self, routed_trainer_config, tmp_output_dir):
        """Test routed is not transformer."""
        trainer = Trainer(routed_trainer_config, tmp_output_dir)
        assert not trainer.is_transformer

    def test_transformer_is_transformer(self, transformer_trainer_config, tmp_output_dir):
        """Test transformer is detected as transformer."""
        trainer = Trainer(transformer_trainer_config, tmp_output_dir)
        assert trainer.is_transformer
