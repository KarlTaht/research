"""Experiment loading utilities for analysis.

Provides clean API for loading experiment artifacts (parquet, config, checkpoints).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torch.nn as nn


def get_default_output_dir() -> Path:
    """Get the default output directory for experiments."""
    project_root = Path(__file__).parent.parent.parent
    return project_root / "outputs"


@dataclass
class ExperimentData:
    """Container for loaded experiment data.

    Attributes:
        name: Experiment directory name (identifier)
        config: Training configuration dict from config.json
        history_df: Metrics history as pandas DataFrame
        checkpoint_paths: List of available checkpoint files
        routing_snapshots: Optional routing weights at key steps
        output_dir: Path to experiment output directory
    """

    name: str
    config: dict
    history_df: pd.DataFrame
    checkpoint_paths: list[Path] = field(default_factory=list)
    routing_snapshots: Optional[list[dict]] = None
    output_dir: Optional[Path] = None

    @property
    def p(self) -> int:
        """Prime modulus from config."""
        return self.config.get("p", 113)

    @property
    def n_layers(self) -> int:
        """Number of layers from config."""
        return self.config.get("n_layers", 2)

    @property
    def n_heads(self) -> int:
        """Number of attention/routing heads from config."""
        return self.config.get("n_heads", 4)

    @property
    def model_type(self) -> str:
        """Model type from config."""
        return self.config.get("model_type", "transformer")

    @property
    def lr(self) -> float:
        """Learning rate from config."""
        return self.config.get("lr", 1e-3)

    @property
    def weight_decay(self) -> float:
        """Weight decay from config."""
        return self.config.get("weight_decay", 1.0)

    @property
    def train_frac(self) -> float:
        """Train fraction from config."""
        return self.config.get("train_frac", 0.5)

    @property
    def max_epoch(self) -> int:
        """Maximum epoch in history."""
        if self.history_df.empty:
            return 0
        return int(self.history_df["step"].max())

    @property
    def has_grokked(self) -> bool:
        """Check if experiment achieved >95% test accuracy."""
        if self.history_df.empty or "test_acc" not in self.history_df.columns:
            return False
        return self.history_df["test_acc"].max() >= 0.95

    @property
    def has_routing(self) -> bool:
        """Check if this experiment has routing data."""
        return self.routing_snapshots is not None and len(self.routing_snapshots) > 0

    @property
    def best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint (by test accuracy)."""
        for path in self.checkpoint_paths:
            if path.name == "best.pt":
                return path
        # Fall back to final.pt
        for path in self.checkpoint_paths:
            if path.name == "final.pt":
                return path
        # Fall back to any checkpoint
        return self.checkpoint_paths[0] if self.checkpoint_paths else None


class ExperimentLoader:
    """Load experiment data from saved artifacts.

    Example:
        loader = ExperimentLoader()
        exp = loader.load('p17_lr3e-4_wd1.0')
        print(exp.history_df.columns)

        # Load checkpoint
        checkpoint = loader.load_checkpoint(exp)
        print(checkpoint.keys())

        # Reconstruct model
        model = loader.reconstruct_model(exp)
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """Initialize loader with output directory.

        Args:
            output_dir: Directory containing experiment subdirectories.
                       Defaults to project outputs/ directory.
        """
        if output_dir is None:
            self.output_dir = get_default_output_dir()
        else:
            self.output_dir = Path(output_dir)

    def list_experiments(self) -> list[str]:
        """List all available experiment names.

        Returns:
            List of experiment directory names that have history.parquet
        """
        if not self.output_dir.exists():
            return []

        experiments = []
        for path in self.output_dir.iterdir():
            if path.is_dir() and (path / "history.parquet").exists():
                experiments.append(path.name)

        return sorted(experiments)

    def load(self, experiment_name: str) -> ExperimentData:
        """Load experiment by name.

        Args:
            experiment_name: Name of experiment directory

        Returns:
            ExperimentData with all loaded artifacts

        Raises:
            FileNotFoundError: If experiment directory or required files not found
        """
        exp_dir = self.output_dir / experiment_name
        return self.load_from_path(exp_dir)

    def load_from_path(self, exp_dir: Path) -> ExperimentData:
        """Load experiment from directory path.

        Args:
            exp_dir: Path to experiment directory

        Returns:
            ExperimentData with all loaded artifacts

        Raises:
            FileNotFoundError: If required files not found
        """
        exp_dir = Path(exp_dir)

        # Load history (required)
        history_path = exp_dir / "history.parquet"
        if not history_path.exists():
            raise FileNotFoundError(f"No history.parquet found in {exp_dir}")
        history_df = pd.read_parquet(history_path)

        # Load config (optional but expected)
        config_path = exp_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}

        # Find checkpoint files
        checkpoint_paths = []
        for name in ["best.pt", "final.pt"]:
            path = exp_dir / name
            if path.exists():
                checkpoint_paths.append(path)
        # Add numbered checkpoints
        for path in sorted(exp_dir.glob("checkpoint_*.pt")):
            checkpoint_paths.append(path)

        # Load routing snapshots if available
        routing_snapshots = None
        routing_path = exp_dir / "routing_snapshots.pt"
        if routing_path.exists():
            routing_data = torch.load(routing_path, map_location="cpu", weights_only=False)
            routing_snapshots = routing_data.get("snapshots", [])

        return ExperimentData(
            name=exp_dir.name,
            config=config,
            history_df=history_df,
            checkpoint_paths=checkpoint_paths,
            routing_snapshots=routing_snapshots,
            output_dir=exp_dir,
        )

    def load_checkpoint(
        self,
        experiment: ExperimentData,
        epoch: Optional[int] = None,
    ) -> dict:
        """Load checkpoint data from experiment.

        Args:
            experiment: ExperimentData instance
            epoch: Specific epoch to load, or None for best checkpoint

        Returns:
            Checkpoint dict with model_state, optimizer_state, config, etc.

        Raises:
            FileNotFoundError: If no matching checkpoint found
        """
        if epoch is None:
            # Load best checkpoint
            checkpoint_path = experiment.best_checkpoint_path
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoint found for {experiment.name}")
        else:
            # Find checkpoint closest to requested epoch
            checkpoint_path = self._find_checkpoint_for_epoch(experiment, epoch)
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"No checkpoint found for epoch {epoch} in {experiment.name}"
                )

        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def _find_checkpoint_for_epoch(
        self,
        experiment: ExperimentData,
        target_epoch: int,
    ) -> Optional[Path]:
        """Find checkpoint closest to target epoch.

        Args:
            experiment: ExperimentData instance
            target_epoch: Target epoch to find

        Returns:
            Path to closest checkpoint, or None if not found
        """
        if not experiment.checkpoint_paths:
            return None

        # Build list of (path, epoch) pairs
        candidates = []
        for path in experiment.checkpoint_paths:
            if path.name == "best.pt":
                # Load to get epoch
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                epoch = ckpt.get("best_epoch", ckpt.get("epoch", 0))
                candidates.append((path, epoch))
            elif path.name == "final.pt":
                # Final is at config epochs
                ckpt = torch.load(path, map_location="cpu", weights_only=False)
                epoch = ckpt.get("epoch", experiment.config.get("epochs", 100000))
                candidates.append((path, epoch))
            elif path.name.startswith("checkpoint_"):
                # Parse epoch from filename
                try:
                    epoch = int(path.stem.split("_")[1])
                    candidates.append((path, epoch))
                except (ValueError, IndexError):
                    continue

        if not candidates:
            return None

        # Find closest to target
        closest_path, closest_dist = None, float("inf")
        for path, epoch in candidates:
            dist = abs(epoch - target_epoch)
            if dist < closest_dist:
                closest_dist = dist
                closest_path = path

        return closest_path

    def list_checkpoints(self, experiment: ExperimentData) -> list[int]:
        """List available checkpoint epochs.

        Args:
            experiment: ExperimentData instance

        Returns:
            List of epoch numbers with available checkpoints
        """
        epochs = []
        for path in experiment.checkpoint_paths:
            if path.name.startswith("checkpoint_"):
                try:
                    epoch = int(path.stem.split("_")[1])
                    epochs.append(epoch)
                except (ValueError, IndexError):
                    continue
        return sorted(epochs)

    def reconstruct_model(
        self,
        experiment: ExperimentData,
        epoch: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> nn.Module:
        """Reconstruct model from checkpoint with weights loaded.

        Args:
            experiment: ExperimentData instance
            epoch: Specific epoch checkpoint to load, or None for best
            device: Device to load model to (default: CPU)

        Returns:
            Model with loaded weights

        Raises:
            FileNotFoundError: If no matching checkpoint found
            ValueError: If model type is unknown
        """
        # Import here to avoid circular dependency
        from ..models import create_model

        checkpoint = self.load_checkpoint(experiment, epoch)
        config = experiment.config

        # Determine model parameters from config
        model_type = config.get("model_type", "transformer")
        p = config.get("p", 113)
        hidden_dim = config.get("hidden_dim", 128)
        n_layers = config.get("n_layers", 2)
        n_heads = config.get("n_heads", 4)

        # Input/output dims depend on model type
        if model_type == "transformer":
            input_dim = p + 2  # vocab_size = p + op + equals
            output_dim = p
            kwargs = {
                "max_seq_len": config.get("max_seq_len", 5),
                "tie_embeddings": config.get("tie_embeddings", False),
            }
        else:
            input_dim = 2 * p  # one-hot encoded pair
            output_dim = p
            kwargs = {}

        # Create model
        model = create_model(
            model_type=model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            **kwargs,
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state"])

        if device is not None:
            model = model.to(device)

        model.eval()
        return model


def get_routing_at_step(
    experiment: ExperimentData,
    step_percent: float,
) -> Optional[dict]:
    """Get routing snapshot closest to given training percentage.

    Args:
        experiment: ExperimentData with routing snapshots
        step_percent: Training progress as percentage (0-100)

    Returns:
        Routing snapshot dict or None if no snapshots available
    """
    if not experiment.has_routing:
        return None

    snapshots = experiment.routing_snapshots
    if not snapshots:
        return None

    # Find snapshot closest to target step
    target_step = int(experiment.max_epoch * step_percent / 100)

    closest = None
    closest_dist = float("inf")

    for snapshot in snapshots:
        step = snapshot.get("step", 0)
        dist = abs(step - target_step)
        if dist < closest_dist:
            closest_dist = dist
            closest = snapshot

    return closest
