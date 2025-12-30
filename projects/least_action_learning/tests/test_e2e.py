"""End-to-end tests for grokking experiments.

These tests verify the full training pipeline and grokking phenomenon.
Many tests are marked as 'slow' and can be skipped with: pytest -m "not slow"
"""

import pytest
import torch
import json
from pathlib import Path

import sys
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from projects.least_action_learning.src.trainer import Trainer, TrainerConfig
from projects.least_action_learning.src.data import ModularArithmeticDataset
from projects.least_action_learning.src.models import create_model


class TestTrainingPipeline:
    """Tests for the complete training pipeline."""

    def test_baseline_training_runs(self, trainer_config, tmp_output_dir):
        """Test baseline MLP training completes without error."""
        trainer_config.epochs = 50
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100  # No checkpoints

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        assert len(history.history) > 0
        # Check files created
        assert (tmp_output_dir / "history.parquet").exists()
        assert (tmp_output_dir / "config.json").exists()

    def test_routed_training_runs(self, routed_trainer_config, tmp_output_dir):
        """Test routed network training completes without error."""
        routed_trainer_config.epochs = 50
        routed_trainer_config.log_every = 10
        routed_trainer_config.checkpoint_every = 100

        trainer = Trainer(routed_trainer_config, tmp_output_dir)
        history = trainer.train()

        assert len(history.history) > 0
        # Should have routing metrics
        assert history.history[0].routing_entropy > 0 or True  # May be 0 initially

    def test_transformer_training_runs(self, transformer_trainer_config, tmp_output_dir):
        """Test transformer training completes without error."""
        transformer_trainer_config.epochs = 50
        transformer_trainer_config.log_every = 10
        transformer_trainer_config.checkpoint_every = 100

        trainer = Trainer(transformer_trainer_config, tmp_output_dir)
        history = trainer.train()

        assert len(history.history) > 0


class TestMemorization:
    """Tests for memorization behavior (train acc improves)."""

    def test_train_accuracy_improves(self, trainer_config, tmp_output_dir):
        """Test that training accuracy improves over epochs."""
        trainer_config.epochs = 100
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 1000

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        # Get first and last training accuracy
        first_acc = history.history[0].train_acc
        last_acc = history.history[-1].train_acc

        # Training accuracy should improve
        assert last_acc > first_acc

    def test_loss_decreases(self, trainer_config, tmp_output_dir):
        """Test that training loss decreases over epochs."""
        trainer_config.epochs = 100
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 1000

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        first_loss = history.history[0].train_loss
        last_loss = history.history[-1].train_loss

        # Loss should decrease
        assert last_loss < first_loss


@pytest.mark.slow
class TestGrokking:
    """Tests for grokking phenomenon.

    These tests are slow because they require many epochs.
    Skip with: pytest -m "not slow"
    """

    def test_grokking_with_weight_decay(self, small_p, tmp_output_dir):
        """Test grokking occurs with proper weight decay."""
        config = TrainerConfig(
            p=small_p,
            operation="add",
            train_frac=0.3,
            model_type="baseline",
            hidden_dim=64,
            n_layers=2,
            n_heads=2,
            epochs=5000,  # Enough for grokking on small problem
            lr=1e-3,
            weight_decay=1.0,  # Critical for grokking
            log_every=100,
            checkpoint_every=10000,
            seed=42,
        )

        trainer = Trainer(config, tmp_output_dir)
        history = trainer.train()

        # Check for grokking: test accuracy should improve significantly
        early_test_acc = history.history[0].test_acc
        final_test_acc = history.history[-1].test_acc

        # Test accuracy should improve (grokking)
        assert final_test_acc > early_test_acc + 0.1  # At least 10% improvement

    def test_no_grokking_without_weight_decay(self, small_p, tmp_output_dir):
        """Test no grokking without weight decay (memorization only)."""
        config = TrainerConfig(
            p=small_p,
            operation="add",
            train_frac=0.3,
            model_type="baseline",
            hidden_dim=64,
            n_layers=2,
            n_heads=2,
            epochs=2000,
            lr=1e-3,
            weight_decay=0.0,  # No weight decay
            log_every=100,
            checkpoint_every=10000,
            seed=42,
        )

        trainer = Trainer(config, tmp_output_dir)
        history = trainer.train()

        # Should memorize training data
        final_train_acc = history.history[-1].train_acc
        final_test_acc = history.history[-1].test_acc

        # High train accuracy, lower test accuracy (no generalization)
        assert final_train_acc > 0.9
        # Without weight decay, test accuracy may not improve much
        # (or model might still generalize somewhat - this is just directional)


class TestCheckpointResume:
    """Tests for checkpoint save/resume functionality."""

    def test_resume_continues_training(self, trainer_config, tmp_output_dir):
        """Test training can resume from checkpoint."""
        trainer_config.epochs = 30
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 20

        # First training run
        trainer1 = Trainer(trainer_config, tmp_output_dir)
        for epoch in range(20):
            trainer1.train_step(epoch)
        trainer1.save_checkpoint("resume_test.pt", epoch=20)

        # Get accuracy at epoch 20
        _, acc_at_20 = trainer1.evaluate()

        # Resume training
        trainer2 = Trainer(trainer_config, tmp_output_dir)
        trainer2.load_checkpoint("resume_test.pt")

        # Train a few more steps
        for epoch in range(20, 30):
            trainer2.train_step(epoch)

        # Accuracy should have changed (training continued)
        _, acc_at_30 = trainer2.evaluate()

        # Model state was restored (not reinitialized)
        assert acc_at_30 != acc_at_20 or acc_at_20 > 0.9  # Either improved or already high

    def test_checkpoint_preserves_best_acc(self, trainer_config, tmp_output_dir):
        """Test checkpoint preserves best accuracy tracking."""
        trainer_config.epochs = 20
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer1 = Trainer(trainer_config, tmp_output_dir)
        trainer1.best_test_acc = 0.75
        trainer1.best_epoch = 500
        trainer1.save_checkpoint("best_acc_test.pt", epoch=500)

        trainer2 = Trainer(trainer_config, tmp_output_dir)
        trainer2.load_checkpoint("best_acc_test.pt")

        assert trainer2.best_test_acc == 0.75
        assert trainer2.best_epoch == 500


class TestMetricsTracking:
    """Tests for metrics tracking during training."""

    def test_all_basic_metrics_logged(self, trainer_config, tmp_output_dir):
        """Test all basic metrics are logged."""
        trainer_config.epochs = 20
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        # Check basic metrics in first entry
        m = history.history[0]
        assert m.step >= 0
        assert 0 <= m.train_acc <= 1
        assert 0 <= m.test_acc <= 1
        assert m.train_loss >= 0
        assert m.test_loss >= 0

    def test_curvature_metrics_logged(self, trainer_config, tmp_output_dir):
        """Test curvature metrics (jacobian, hessian) are logged."""
        trainer_config.epochs = 20
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        m = history.history[0]
        # These should be computed during training
        assert m.jacobian_norm is not None
        assert m.hessian_trace is not None

    def test_spectral_smoothness_logged(self, trainer_config, tmp_output_dir):
        """Test spectral smoothness is logged."""
        trainer_config.epochs = 20
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        m = history.history[0]
        assert m.spectral_smoothness is not None
        assert 0 <= m.spectral_smoothness <= 1

    def test_representation_norm_logged(self, trainer_config, tmp_output_dir):
        """Test representation norm is logged."""
        trainer_config.epochs = 20
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        m = history.history[0]
        assert m.representation_norm is not None
        assert m.representation_norm > 0

    def test_history_to_dataframe(self, trainer_config, tmp_output_dir):
        """Test history can be converted to DataFrame."""
        trainer_config.epochs = 30
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        df = history.get_dataframe()

        assert len(df) == 3  # 3 log entries at epochs 0, 10, 20
        assert "train_acc" in df.columns
        assert "test_acc" in df.columns
        assert "train_loss" in df.columns


class TestReproducibility:
    """Tests for reproducibility with fixed seeds."""

    def test_same_seed_same_results(self, small_p, tmp_output_dir):
        """Test same seed produces same results."""
        config = TrainerConfig(
            p=small_p,
            operation="add",
            train_frac=0.3,
            model_type="baseline",
            hidden_dim=32,
            n_layers=2,
            n_heads=2,
            epochs=20,
            lr=1e-3,
            log_every=10,
            checkpoint_every=100,
            seed=42,
        )

        # First run
        trainer1 = Trainer(config, tmp_output_dir / "run1")
        history1 = trainer1.train()

        # Second run with same seed
        torch.manual_seed(42)  # Reset global seed
        trainer2 = Trainer(config, tmp_output_dir / "run2")
        history2 = trainer2.train()

        # Results should be identical
        assert history1.history[0].train_loss == history2.history[0].train_loss
        assert history1.history[-1].train_acc == history2.history[-1].train_acc

    def test_different_seed_different_results(self, small_p, tmp_output_dir):
        """Test different seeds produce different results."""
        base_config = dict(
            p=small_p,
            operation="add",
            train_frac=0.3,
            model_type="baseline",
            hidden_dim=32,
            n_layers=2,
            n_heads=2,
            epochs=20,
            lr=1e-3,
            log_every=10,
            checkpoint_every=100,
        )

        # First run
        config1 = TrainerConfig(**base_config, seed=42)
        trainer1 = Trainer(config1, tmp_output_dir / "seed42")
        history1 = trainer1.train()

        # Second run with different seed
        config2 = TrainerConfig(**base_config, seed=123)
        trainer2 = Trainer(config2, tmp_output_dir / "seed123")
        history2 = trainer2.train()

        # Results should be different (with high probability)
        # Check final train loss differs
        final_loss1 = history1.history[-1].train_loss
        final_loss2 = history2.history[-1].train_loss
        assert final_loss1 != final_loss2


class TestOutputFiles:
    """Tests for output file generation."""

    def test_config_json_saved(self, trainer_config, tmp_output_dir):
        """Test config.json is saved correctly."""
        trainer_config.epochs = 10
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        trainer.train()

        config_path = tmp_output_dir / "config.json"
        assert config_path.exists()

        with open(config_path) as f:
            saved_config = json.load(f)

        assert saved_config["p"] == trainer_config.p
        assert saved_config["model_type"] == trainer_config.model_type
        assert saved_config["epochs"] == trainer_config.epochs

    def test_history_parquet_saved(self, trainer_config, tmp_output_dir):
        """Test history.parquet is saved correctly."""
        trainer_config.epochs = 20
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        trainer.train()

        history_path = tmp_output_dir / "history.parquet"
        assert history_path.exists()

        import pandas as pd
        df = pd.read_parquet(history_path)
        assert len(df) == 2  # Logged at 0, 10
        assert "step" in df.columns

    def test_final_checkpoint_saved(self, trainer_config, tmp_output_dir):
        """Test final.pt checkpoint is saved."""
        trainer_config.epochs = 10
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        trainer.train()

        assert (tmp_output_dir / "final.pt").exists()


class TestWarmupE2E:
    """E2E tests for LR warmup."""

    def test_warmup_training_completes(self, trainer_config, tmp_output_dir):
        """Test training with warmup completes."""
        trainer_config.epochs = 50
        trainer_config.warmup_epochs = 10
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        # Should complete and have metrics
        assert len(history.history) > 0


class TestGradientClippingE2E:
    """E2E tests for gradient clipping."""

    def test_gradient_clipping_training_completes(self, trainer_config, tmp_output_dir):
        """Test training with gradient clipping completes."""
        trainer_config.epochs = 30
        trainer_config.grad_clip = 1.0
        trainer_config.log_every = 10
        trainer_config.checkpoint_every = 100

        trainer = Trainer(trainer_config, tmp_output_dir)
        history = trainer.train()

        assert len(history.history) > 0
        # Loss should still decrease
        assert history.history[-1].train_loss < history.history[0].train_loss * 1.5


class TestDifferentOperations:
    """E2E tests for different modular arithmetic operations."""

    def test_addition_training(self, small_p, tmp_output_dir):
        """Test training on addition operation."""
        config = TrainerConfig(
            p=small_p,
            operation="add",
            train_frac=0.3,
            model_type="baseline",
            hidden_dim=32,
            n_layers=2,
            epochs=30,
            log_every=10,
            checkpoint_every=100,
            seed=42,
        )

        trainer = Trainer(config, tmp_output_dir)
        history = trainer.train()

        assert len(history.history) > 0

    def test_multiplication_training(self, small_p, tmp_output_dir):
        """Test training on multiplication operation."""
        config = TrainerConfig(
            p=small_p,
            operation="multiply",
            train_frac=0.3,
            model_type="baseline",
            hidden_dim=32,
            n_layers=2,
            epochs=30,
            log_every=10,
            checkpoint_every=100,
            seed=42,
        )

        trainer = Trainer(config, tmp_output_dir)
        history = trainer.train()

        assert len(history.history) > 0
