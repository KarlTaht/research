"""Tests for ReferenceTransformer training pipeline.

Fast integration test with tiny model and synthetic data.
Should complete in < 30 seconds.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2TokenizerFast
import pandas as pd

from common.models import ReferenceTransformer
from common.training import Evaluator
from common.utils import save_experiment


class SyntheticSummarizationDataset(Dataset):
    """Synthetic dialogue-summary pairs for testing."""

    def __init__(self, num_examples=20, sep_token_id=50256, max_len=64):
        """Create synthetic dataset.

        Args:
            num_examples: Number of synthetic examples
            sep_token_id: Separator token ID
            max_len: Maximum sequence length
        """
        self.examples = []
        self.sep_token_id = sep_token_id

        for i in range(num_examples):
            # Create simple synthetic sequences
            # Dialogue (source): random tokens
            dialogue_len = torch.randint(10, 20, (1,)).item()
            dialogue_ids = torch.randint(100, 1000, (dialogue_len,)).tolist()

            # Summary (target): random tokens
            summary_len = torch.randint(5, 10, (1,)).item()
            summary_ids = torch.randint(100, 1000, (summary_len,)).tolist()

            # Pack: [dialogue] [SEP] [summary]
            input_ids = dialogue_ids + [sep_token_id] + summary_ids

            # Truncate if too long
            input_ids = input_ids[:max_len]

            self.examples.append({
                'input_ids': input_ids,
                'labels': input_ids.copy()
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Pad sequences in batch."""
    max_len = max(len(item['input_ids']) for item in batch)

    input_ids_list = []
    labels_list = []

    for item in batch:
        input_ids = item['input_ids']
        labels = item['labels']

        # Pad with 0
        padding_len = max_len - len(input_ids)
        input_ids = input_ids + [0] * padding_len
        labels = labels + [0] * padding_len

        input_ids_list.append(input_ids)
        labels_list.append(labels)

    return {
        'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
        'labels': torch.tensor(labels_list, dtype=torch.long)
    }


class TestTransformerTraining:
    """Test suite for ReferenceTransformer training pipeline."""

    @pytest.fixture
    def tiny_config(self):
        """Tiny model configuration for fast testing."""
        return {
            'vocab_size': 1000,      # Small vocab
            'd_model': 32,            # Tiny embedding
            'n_heads': 2,             # Minimal heads
            'n_encoder_layers': 1,    # Single encoder layer
            'n_decoder_layers': 1,    # Single decoder layer
            'd_ff': 64,               # Small feedforward (2 * d_model)
            'dropout': 0.0,           # No dropout for determinism
            'max_seq_len': 64,        # Short sequences
        }

    @pytest.fixture
    def device(self):
        """Get available device (CPU for CI compatibility)."""
        return torch.device('cpu')  # Use CPU for deterministic testing

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test artifacts."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    def test_model_initialization(self, tiny_config):
        """Test model initializes correctly."""
        model = ReferenceTransformer(**tiny_config)

        # Check model was created
        assert model is not None
        assert model.vocab_size == tiny_config['vocab_size']
        assert model.d_model == tiny_config['d_model']

        # Check parameters exist
        param_count = model.count_parameters()
        assert param_count > 0
        print(f"✓ Model initialized with {param_count:,} parameters")

    def test_forward_pass(self, tiny_config, device):
        """Test forward pass with synthetic data."""
        model = ReferenceTransformer(**tiny_config)
        model = model.to(device)
        model.eval()

        # Create synthetic batch: [dialogue] [SEP] [summary]
        sep_id = tiny_config['vocab_size'] - 1
        dialogue = [10, 20, 30, 40]
        summary = [50, 60]
        input_ids = torch.tensor([dialogue + [sep_id] + summary], dtype=torch.long).to(device)
        labels = input_ids.clone()

        # Forward pass
        with torch.no_grad():
            output = model(input_ids, labels=labels)

        # Check outputs
        assert 'logits' in output
        assert 'loss' in output
        assert output['logits'].shape[0] == 1  # batch size
        assert output['logits'].shape[2] == tiny_config['vocab_size']
        assert output['loss'].item() > 0

        print(f"✓ Forward pass successful, loss: {output['loss'].item():.4f}")

    def test_training_epoch(self, tiny_config, device):
        """Test single training epoch completes successfully."""
        # Initialize model
        model = ReferenceTransformer(**tiny_config, pad_token_id=0)
        model = model.to(device)

        # Create synthetic dataset
        train_dataset = SyntheticSummarizationDataset(
            num_examples=20,
            sep_token_id=tiny_config['vocab_size'] - 1,
            max_len=tiny_config['max_seq_len']
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            shuffle=True,
            collate_fn=collate_fn
        )

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train for one epoch
        model.train()
        losses = []

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, labels=labels)
            loss = outputs['loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        assert avg_loss > 0
        assert avg_loss < 100  # Reasonable range

        print(f"✓ Training epoch completed, avg loss: {avg_loss:.4f}")

    def test_loss_decreases(self, tiny_config, device):
        """Test that loss decreases over multiple epochs."""
        # Initialize model
        model = ReferenceTransformer(**tiny_config, pad_token_id=0)
        model = model.to(device)

        # Create synthetic dataset
        train_dataset = SyntheticSummarizationDataset(
            num_examples=30,
            sep_token_id=tiny_config['vocab_size'] - 1
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=6,
            shuffle=False,  # Deterministic
            collate_fn=collate_fn
        )

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train for 3 epochs
        epoch_losses = []

        for epoch in range(3):
            model.train()
            losses = []

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            avg_loss = sum(losses) / len(losses)
            epoch_losses.append(avg_loss)

        # Check loss decreased
        assert epoch_losses[-1] < epoch_losses[0], \
            f"Loss did not decrease: {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}"

        print(f"✓ Loss decreased over epochs: {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}")

    def test_checkpoint_save_load(self, tiny_config, device, temp_dir):
        """Test checkpoint saving and loading."""
        # Initialize model
        model = ReferenceTransformer(**tiny_config)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        model.save_checkpoint(
            checkpoint_path,
            optimizer=optimizer,
            epoch=5,
            loss=1.234
        )

        # Check file was created
        assert checkpoint_path.exists()

        # Load checkpoint into new model
        new_model = ReferenceTransformer(**tiny_config)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        metadata = new_model.load_checkpoint(checkpoint_path, optimizer=new_optimizer)

        # Check metadata
        assert metadata['epoch'] == 5
        assert 'model_config' in metadata

        # Check state matches
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

        print(f"✓ Checkpoint save/load successful")

    def test_evaluator_integration(self, tiny_config, device):
        """Test Evaluator works with ReferenceTransformer."""
        model = ReferenceTransformer(**tiny_config, pad_token_id=0)
        model = model.to(device)

        # Create validation dataset
        val_dataset = SyntheticSummarizationDataset(
            num_examples=10,
            sep_token_id=tiny_config['vocab_size'] - 1
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            collate_fn=collate_fn
        )

        # Initialize evaluator
        evaluator = Evaluator(model, device=device)

        # Evaluate
        metrics = evaluator.evaluate(val_loader, max_batches=None)

        # Check metrics
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'num_tokens' in metrics
        assert metrics['loss'] > 0
        assert metrics['perplexity'] > 1.0

        print(f"✓ Evaluator integration successful")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Perplexity: {metrics['perplexity']:.2f}")

    def test_experiment_tracking(self, temp_dir):
        """Test experiment tracking with save_experiment."""
        # Create sample results
        results_df = pd.DataFrame({
            'epoch': [1, 2, 3],
            'train_loss': [2.5, 1.8, 1.2],
            'val_loss': [2.3, 1.7, 1.3],
            'val_perplexity': [10.0, 5.5, 3.7]
        })

        metadata = {
            'model': 'ReferenceTransformer',
            'test': True,
            'd_model': 32,
            'n_layers': 1
        }

        # Save experiment
        output_path = save_experiment(
            'test_transformer_experiment',
            results_df,
            metadata=metadata,
            output_dir=temp_dir
        )

        # Check file was created
        assert output_path.exists()
        assert output_path.suffix == '.parquet'

        # Load and verify
        loaded_df = pd.read_parquet(output_path)
        assert len(loaded_df) == 3
        assert 'epoch' in loaded_df.columns
        assert 'experiment_name' in loaded_df.columns
        assert all(loaded_df['experiment_name'] == 'test_transformer_experiment')

        print(f"✓ Experiment tracking successful")

    def test_full_training_pipeline(self, tiny_config, device, temp_dir):
        """Full integration test: train -> save -> load -> evaluate."""
        # 1. Initialize model
        model = ReferenceTransformer(**tiny_config, pad_token_id=0)
        model = model.to(device)

        # 2. Create datasets with correct separator token
        sep_token_id = tiny_config['vocab_size'] - 1
        train_dataset = SyntheticSummarizationDataset(
            num_examples=20, sep_token_id=sep_token_id
        )
        val_dataset = SyntheticSummarizationDataset(
            num_examples=10, sep_token_id=sep_token_id
        )

        train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)

        # 3. Train for 2 epochs
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        results = []

        for epoch in range(1, 3):
            # Train
            model.train()
            train_losses = []

            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, labels=labels)
                loss = outputs['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = sum(train_losses) / len(train_losses)

            # Evaluate
            evaluator = Evaluator(model, device=device)
            val_metrics = evaluator.evaluate(val_loader)

            results.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': val_metrics['loss'],
                'val_perplexity': val_metrics['perplexity']
            })

        # 4. Save checkpoint
        checkpoint_path = temp_dir / "final_checkpoint.pt"
        model.save_checkpoint(
            checkpoint_path,
            optimizer=optimizer,
            epoch=2,
            loss=results[-1]['val_loss']
        )

        # 5. Save experiment
        results_df = pd.DataFrame(results)
        save_experiment('test_full_pipeline', results_df, output_dir=temp_dir)

        # 6. Load checkpoint and verify
        new_model = ReferenceTransformer(**tiny_config, pad_token_id=0)
        metadata = new_model.load_checkpoint(checkpoint_path)

        assert metadata['epoch'] == 2

        # 7. Verify loss decreased
        assert results[-1]['train_loss'] < results[0]['train_loss']

        print(f"✓ Full training pipeline test passed")
        print(f"  Initial loss: {results[0]['train_loss']:.4f}")
        print(f"  Final loss: {results[-1]['train_loss']:.4f}")
        print(f"  Final perplexity: {results[-1]['val_perplexity']:.2f}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
