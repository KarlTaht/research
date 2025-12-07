"""Checkpoint management for training resumption."""

import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union
import json


class CheckpointManager:
    """
    Manages checkpoint saving, loading, and training resumption.

    Designed for CustomTransformer (manual backprop) but works with any model
    that implements save_checkpoint/load_state_dict.

    Features:
    - Automatic checkpoint rotation (keeps N most recent)
    - latest.pt pointer for easy resumption
    - best.pt tracking for best model by metric
    - Full training state for resumption (epoch, step, lr, config)

    Example:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir='checkpoints',
            model=model,
            experiment_name='transformer_tinystories',
        )

        # Training loop
        for epoch in range(start_epoch, num_epochs):
            for step, batch in enumerate(train_loader):
                # ... training step ...
                global_step += 1

            # Save at end of epoch
            checkpoint_manager.save_checkpoint(
                epoch=epoch + 1,
                global_step=global_step,
                train_config=config,
                learning_rate=lr,
                metrics={'val_loss': val_loss, 'val_perplexity': val_ppl},
                is_best=(val_loss < best_val_loss),
            )

        # Resume from checkpoint
        resume_state = checkpoint_manager.load_checkpoint()
        if resume_state:
            start_epoch = resume_state['epoch']
            global_step = resume_state['global_step']
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        model: Any,
        experiment_name: str,
        max_checkpoints: int = 5,
    ):
        """
        Initialize CheckpointManager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            model: Model instance (CustomTransformerWrapper or BaseLanguageModel)
            experiment_name: Name for this experiment (used in filenames)
            max_checkpoints: Maximum number of epoch checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.experiment_name = experiment_name
        self.max_checkpoints = max_checkpoints

    def save_checkpoint(
        self,
        epoch: int,
        global_step: int,
        train_config: dict,
        learning_rate: float,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
    ) -> Path:
        """
        Save a training checkpoint.

        Note: For CustomTransformer with manual backprop, there's no optimizer
        state to save. The learning rate is saved for resumption.

        Args:
            epoch: Current epoch number (1-indexed, i.e., after completing epoch)
            global_step: Total training steps completed
            train_config: Training configuration dict
            learning_rate: Current learning rate
            metrics: Optional dict of metrics (loss, perplexity, etc.)
            is_best: If True, also save as best.pt

        Returns:
            Path to saved checkpoint file
        """
        # Get model state
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'state_dict'):
            # CustomTransformerWrapper
            model_state = self.model.model.state_dict()
            model_config = self.model.model.get_config()
        elif hasattr(self.model, 'state_dict'):
            # BaseLanguageModel
            model_state = self.model.state_dict()
            model_config = getattr(self.model, 'model_config', {})
        else:
            raise ValueError("Model must have state_dict() method")

        checkpoint = {
            # Model state
            'model_state_dict': model_state,
            'model_config': model_config,

            # Training state
            'epoch': epoch,
            'global_step': global_step,
            'learning_rate': learning_rate,
            'train_config': train_config,

            # Metrics
            'metrics': metrics or {},

            # Metadata
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name,
        }

        # Save epoch checkpoint
        filename = f"checkpoint_epoch{epoch}_step{global_step}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

        # Always update latest pointer
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best if applicable
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Best checkpoint updated: {best_path}")

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        return filepath

    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint and restore model state.

        Args:
            checkpoint_path: Specific checkpoint to load.
                           If None, loads 'latest.pt'.
                           Can also be 'best' to load best.pt.

        Returns:
            Dict with training state for resumption, or None if no checkpoint found.
            Contains: epoch, global_step, learning_rate, train_config, metrics
        """
        # Determine checkpoint path
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest.pt"
        elif checkpoint_path == 'best':
            checkpoint_path = self.checkpoint_dir / "best.pt"
        else:
            checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            print(f"No checkpoint found at {checkpoint_path}")
            return None

        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Restore model state
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'load_state_dict'):
            # CustomTransformerWrapper
            self.model.model.load_state_dict(checkpoint['model_state_dict'])
        elif hasattr(self.model, 'load_state_dict'):
            # BaseLanguageModel
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise ValueError("Model must have load_state_dict() method")

        print(f"Restored model from epoch {checkpoint['epoch']}, step {checkpoint['global_step']}")

        return {
            'epoch': checkpoint['epoch'],
            'global_step': checkpoint['global_step'],
            'learning_rate': checkpoint['learning_rate'],
            'train_config': checkpoint.get('train_config', {}),
            'metrics': checkpoint.get('metrics', {}),
        }

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_checkpoints most recent."""
        # Find all epoch checkpoints (exclude latest.pt and best.pt)
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Remove old ones
        for old_ckpt in checkpoints[self.max_checkpoints:]:
            old_ckpt.unlink()
            print(f"Removed old checkpoint: {old_ckpt.name}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to latest checkpoint if it exists."""
        latest_path = self.checkpoint_dir / "latest.pt"
        return latest_path if latest_path.exists() else None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to best checkpoint if it exists."""
        best_path = self.checkpoint_dir / "best.pt"
        return best_path if best_path.exists() else None

    def list_checkpoints(self) -> Dict[str, Any]:
        """
        List all available checkpoints with metadata.

        Returns:
            Dict mapping checkpoint name to metadata
        """
        checkpoints = {}

        for ckpt_path in self.checkpoint_dir.glob("*.pt"):
            try:
                # Load just metadata (don't load full state dict)
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                checkpoints[ckpt_path.name] = {
                    'epoch': checkpoint.get('epoch'),
                    'global_step': checkpoint.get('global_step'),
                    'timestamp': checkpoint.get('timestamp'),
                    'metrics': checkpoint.get('metrics', {}),
                }
            except Exception as e:
                checkpoints[ckpt_path.name] = {'error': str(e)}

        return checkpoints
