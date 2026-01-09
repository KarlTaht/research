"""Modular arithmetic dataset for grokking experiments."""

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Literal


@dataclass
class DatasetSplit:
    """Container for train/test split tensors."""
    inputs: Tensor  # [N, 2*p] one-hot encoded (a, b)
    targets: Tensor  # [N] target class indices
    pairs: Tensor  # [N, 2] original (a, b) pairs for analysis


class ModularArithmeticDataset(Dataset):
    """
    Dataset for modular arithmetic operations.

    Generates all p² pairs (a, b) for a prime p and computes
    the result of the specified operation mod p.

    Args:
        p: Prime modulus
        operation: "add" for (a + b) mod p, "multiply" for (a * b) mod p
        train_frac: Fraction of data to use for training
        seed: Random seed for reproducible splits
    """

    def __init__(
        self,
        p: int,
        operation: Literal["add", "multiply"] = "add",
        train_frac: float = 0.3,
        seed: int = 42,
    ):
        self.p = p
        self.operation = operation
        self.train_frac = train_frac
        self.seed = seed

        # Generate all p² pairs
        self._generate_data()
        self._create_split()

    def _generate_data(self):
        """Generate all (a, b) pairs and compute targets."""
        # Create all pairs
        a_vals = torch.arange(self.p)
        b_vals = torch.arange(self.p)
        aa, bb = torch.meshgrid(a_vals, b_vals, indexing='ij')

        self.all_pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)  # [p², 2]

        # Compute targets based on operation
        a = self.all_pairs[:, 0]
        b = self.all_pairs[:, 1]

        if self.operation == "add":
            self.all_targets = (a + b) % self.p
        elif self.operation == "multiply":
            self.all_targets = (a * b) % self.p
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        # One-hot encode inputs: concatenate one-hot(a) and one-hot(b)
        self.all_inputs = self._one_hot_encode(self.all_pairs)

    def _one_hot_encode(self, pairs: Tensor) -> Tensor:
        """One-hot encode (a, b) pairs into 2p-dimensional vectors."""
        batch_size = pairs.shape[0]
        encoded = torch.zeros(batch_size, 2 * self.p)

        # First p dimensions for a, next p for b
        encoded[torch.arange(batch_size), pairs[:, 0]] = 1.0
        encoded[torch.arange(batch_size), self.p + pairs[:, 1]] = 1.0

        return encoded

    def _create_split(self):
        """Create train/test split."""
        generator = torch.Generator().manual_seed(self.seed)
        n_total = self.p * self.p
        n_train = int(n_total * self.train_frac)

        # Random permutation
        perm = torch.randperm(n_total, generator=generator)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        self.train_split = DatasetSplit(
            inputs=self.all_inputs[train_idx],
            targets=self.all_targets[train_idx],
            pairs=self.all_pairs[train_idx],
        )

        self.test_split = DatasetSplit(
            inputs=self.all_inputs[test_idx],
            targets=self.all_targets[test_idx],
            pairs=self.all_pairs[test_idx],
        )

    def get_train(self) -> DatasetSplit:
        """Get training data split."""
        return self.train_split

    def get_test(self) -> DatasetSplit:
        """Get test data split."""
        return self.test_split

    def get_all(self) -> DatasetSplit:
        """Get all data (useful for evaluation on full input space)."""
        return DatasetSplit(
            inputs=self.all_inputs,
            targets=self.all_targets,
            pairs=self.all_pairs,
        )

    @property
    def input_dim(self) -> int:
        """Input dimension (2 * p for one-hot encoding)."""
        return 2 * self.p

    @property
    def output_dim(self) -> int:
        """Output dimension (p classes)."""
        return self.p

    def __len__(self) -> int:
        """Total number of examples."""
        return self.p * self.p

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single example."""
        return self.all_inputs[idx], self.all_targets[idx]


class ModularArithmeticLoader:
    """
    Convenience class for creating dataloaders.

    For grokking experiments, typically use full-batch training.
    """

    def __init__(
        self,
        p: int,
        operation: Literal["add", "multiply"] = "add",
        train_frac: float = 0.3,
        batch_size: int = -1,  # -1 means full batch
        seed: int = 42,
    ):
        self.dataset = ModularArithmeticDataset(p, operation, train_frac, seed)
        self.batch_size = batch_size

    def get_train_loader(self) -> DataLoader:
        """Get training dataloader."""
        split = self.dataset.get_train()
        bs = len(split.inputs) if self.batch_size == -1 else self.batch_size

        dataset = torch.utils.data.TensorDataset(split.inputs, split.targets)
        return DataLoader(dataset, batch_size=bs, shuffle=True)

    def get_test_loader(self) -> DataLoader:
        """Get test dataloader."""
        split = self.dataset.get_test()
        bs = len(split.inputs) if self.batch_size == -1 else self.batch_size

        dataset = torch.utils.data.TensorDataset(split.inputs, split.targets)
        return DataLoader(dataset, batch_size=bs, shuffle=False)

    def get_full_loader(self) -> DataLoader:
        """Get loader for all data (useful for visualization)."""
        split = self.dataset.get_all()
        dataset = torch.utils.data.TensorDataset(split.inputs, split.targets)
        return DataLoader(dataset, batch_size=len(split.inputs), shuffle=False)

    @property
    def input_dim(self) -> int:
        return self.dataset.input_dim

    @property
    def output_dim(self) -> int:
        return self.dataset.output_dim


def get_pair_indices(p: int, pairs: list[tuple[int, int]]) -> Tensor:
    """
    Get indices of specific (a, b) pairs in the flattened dataset.

    Useful for analyzing specific inputs like (a, 0), (0, b), etc.
    """
    indices = []
    for a, b in pairs:
        idx = a * p + b
        indices.append(idx)
    return torch.tensor(indices)


@dataclass
class SequenceDatasetSplit:
    """Container for sequence-based train/test split tensors."""
    input_ids: Tensor  # [N, seq_len] token sequences
    targets: Tensor  # [N] target result tokens
    pairs: Tensor  # [N, 2] original (a, b) pairs for analysis


class SequenceArithmeticDataset(Dataset):
    """
    Sequence-based dataset for transformer grokking experiments.

    Produces sequences in the format [a, op, b, =] where:
    - a, b are residue tokens (0 to p-1)
    - op is the operation token (token id = p)
    - = is the equals token (token id = p+1)

    The target is the result token (0 to p-1).

    This matches the original grokking paper's transformer setup.

    Args:
        p: Prime modulus
        operation: "add" for (a + b) mod p, "multiply" for (a * b) mod p
        train_frac: Fraction of data to use for training
        seed: Random seed for reproducible splits
    """

    # Special token offsets
    OP_TOKEN_OFFSET = 0  # op token is at index p
    EQ_TOKEN_OFFSET = 1  # = token is at index p + 1

    def __init__(
        self,
        p: int,
        operation: Literal["add", "multiply"] = "add",
        train_frac: float = 0.3,
        seed: int = 42,
    ):
        self.p = p
        self.operation = operation
        self.train_frac = train_frac
        self.seed = seed

        # Token vocabulary
        self.op_token = p  # operation token
        self.eq_token = p + 1  # equals token
        self.vocab_size = p + 2  # p residues + op + equals

        # Generate data
        self._generate_data()
        self._create_split()

    def _generate_data(self):
        """Generate all (a, b) pairs as sequences."""
        # Create all pairs
        a_vals = torch.arange(self.p)
        b_vals = torch.arange(self.p)
        aa, bb = torch.meshgrid(a_vals, b_vals, indexing='ij')

        self.all_pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)  # [p², 2]
        n_examples = self.all_pairs.shape[0]

        # Compute targets based on operation
        a = self.all_pairs[:, 0]
        b = self.all_pairs[:, 1]

        if self.operation == "add":
            self.all_targets = ((a + b) % self.p).long()
        elif self.operation == "multiply":
            self.all_targets = ((a * b) % self.p).long()
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        # Create sequences: [a, op, b, =]
        # Shape: [n_examples, 4]
        # Explicitly cast to long for embedding layer compatibility
        self.all_input_ids = torch.stack([
            a.long(),
            torch.full((n_examples,), self.op_token, dtype=torch.long),
            b.long(),
            torch.full((n_examples,), self.eq_token, dtype=torch.long),
        ], dim=1)

    def _create_split(self):
        """Create train/test split."""
        generator = torch.Generator().manual_seed(self.seed)
        n_total = self.p * self.p
        n_train = int(n_total * self.train_frac)

        # Random permutation
        perm = torch.randperm(n_total, generator=generator)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]

        self.train_split = SequenceDatasetSplit(
            input_ids=self.all_input_ids[train_idx],
            targets=self.all_targets[train_idx],
            pairs=self.all_pairs[train_idx],
        )

        self.test_split = SequenceDatasetSplit(
            input_ids=self.all_input_ids[test_idx],
            targets=self.all_targets[test_idx],
            pairs=self.all_pairs[test_idx],
        )

    def get_train(self) -> SequenceDatasetSplit:
        """Get training data split."""
        return self.train_split

    def get_test(self) -> SequenceDatasetSplit:
        """Get test data split."""
        return self.test_split

    def get_all(self) -> SequenceDatasetSplit:
        """Get all data (useful for evaluation on full input space)."""
        return SequenceDatasetSplit(
            input_ids=self.all_input_ids,
            targets=self.all_targets,
            pairs=self.all_pairs,
        )

    @property
    def input_dim(self) -> int:
        """For compatibility - returns vocab size."""
        return self.vocab_size

    @property
    def output_dim(self) -> int:
        """Output dimension (p classes for result prediction)."""
        return self.p

    @property
    def seq_len(self) -> int:
        """Sequence length (4 tokens: a, op, b, =)."""
        return 4

    def __len__(self) -> int:
        """Total number of examples."""
        return self.p * self.p

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single example."""
        return self.all_input_ids[idx], self.all_targets[idx]


class MultiOpSequenceDataset(Dataset):
    """
    Multi-operation sequence dataset for transformer grokking experiments.

    Combines both addition and multiplication operations in a single dataset.
    This tests whether routing can specialize heads per operation.

    Produces sequences in the format [a, op, b, =] where:
    - a, b are residue tokens (0 to p-1)
    - op is either add_token (p) or mul_token (p+1)
    - = is the equals token (p+2)

    The target is the result token (0 to p-1).

    Args:
        p: Prime modulus
        train_frac: Fraction of data to use for training (applied per operation)
        seed: Random seed for reproducible splits
    """

    def __init__(
        self,
        p: int,
        train_frac: float = 0.3,
        seed: int = 42,
    ):
        self.p = p
        self.train_frac = train_frac
        self.seed = seed

        # Token vocabulary
        self.add_token = p      # addition operation token
        self.mul_token = p + 1  # multiplication operation token
        self.eq_token = p + 2   # equals token
        self.vocab_size = p + 3  # p residues + add + mul + equals

        # Generate data
        self._generate_data()
        self._create_split()

    def _generate_data(self):
        """Generate all (a, b) pairs for both operations."""
        # Create all pairs
        a_vals = torch.arange(self.p)
        b_vals = torch.arange(self.p)
        aa, bb = torch.meshgrid(a_vals, b_vals, indexing='ij')

        pairs = torch.stack([aa.flatten(), bb.flatten()], dim=1)  # [p², 2]
        n_pairs = pairs.shape[0]

        a = pairs[:, 0]
        b = pairs[:, 1]

        # Addition data
        add_targets = ((a + b) % self.p).long()
        add_input_ids = torch.stack([
            a.long(),
            torch.full((n_pairs,), self.add_token, dtype=torch.long),
            b.long(),
            torch.full((n_pairs,), self.eq_token, dtype=torch.long),
        ], dim=1)
        add_ops = torch.zeros(n_pairs, dtype=torch.long)  # 0 = add

        # Multiplication data
        mul_targets = ((a * b) % self.p).long()
        mul_input_ids = torch.stack([
            a.long(),
            torch.full((n_pairs,), self.mul_token, dtype=torch.long),
            b.long(),
            torch.full((n_pairs,), self.eq_token, dtype=torch.long),
        ], dim=1)
        mul_ops = torch.ones(n_pairs, dtype=torch.long)  # 1 = mul

        # Combine both operations
        self.all_pairs = torch.cat([pairs, pairs], dim=0)  # [2p², 2]
        self.all_targets = torch.cat([add_targets, mul_targets], dim=0)
        self.all_input_ids = torch.cat([add_input_ids, mul_input_ids], dim=0)
        self.all_ops = torch.cat([add_ops, mul_ops], dim=0)  # Track which operation

    def _create_split(self):
        """Create train/test split (stratified by operation)."""
        generator = torch.Generator().manual_seed(self.seed)
        n_per_op = self.p * self.p
        n_train_per_op = int(n_per_op * self.train_frac)

        # Split each operation separately for balanced train/test
        perm_add = torch.randperm(n_per_op, generator=generator)
        perm_mul = torch.randperm(n_per_op, generator=generator) + n_per_op

        train_idx = torch.cat([perm_add[:n_train_per_op], perm_mul[:n_train_per_op]])
        test_idx = torch.cat([perm_add[n_train_per_op:], perm_mul[n_train_per_op:]])

        # Shuffle combined indices
        train_perm = torch.randperm(len(train_idx), generator=generator)
        test_perm = torch.randperm(len(test_idx), generator=generator)
        train_idx = train_idx[train_perm]
        test_idx = test_idx[test_perm]

        self.train_split = SequenceDatasetSplit(
            input_ids=self.all_input_ids[train_idx],
            targets=self.all_targets[train_idx],
            pairs=self.all_pairs[train_idx],
        )
        self.train_ops = self.all_ops[train_idx]

        self.test_split = SequenceDatasetSplit(
            input_ids=self.all_input_ids[test_idx],
            targets=self.all_targets[test_idx],
            pairs=self.all_pairs[test_idx],
        )
        self.test_ops = self.all_ops[test_idx]

    def get_train(self) -> SequenceDatasetSplit:
        """Get training data split."""
        return self.train_split

    def get_test(self) -> SequenceDatasetSplit:
        """Get test data split."""
        return self.test_split

    def get_all(self) -> SequenceDatasetSplit:
        """Get all data."""
        return SequenceDatasetSplit(
            input_ids=self.all_input_ids,
            targets=self.all_targets,
            pairs=self.all_pairs,
        )

    @property
    def input_dim(self) -> int:
        """Returns vocab size."""
        return self.vocab_size

    @property
    def output_dim(self) -> int:
        """Output dimension (p classes)."""
        return self.p

    @property
    def seq_len(self) -> int:
        """Sequence length (4 tokens: a, op, b, =)."""
        return 4

    def __len__(self) -> int:
        """Total number of examples (2 * p²)."""
        return 2 * self.p * self.p

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a single example."""
        return self.all_input_ids[idx], self.all_targets[idx]


def get_special_pairs(p: int) -> dict[str, list[tuple[int, int]]]:
    """
    Get lists of special (a, b) pairs for analysis.

    Returns pairs grouped by their "type":
    - zero_pairs: (a, 0) and (0, b)
    - identity_pairs: (a, a)
    - small_pairs: both a, b < p/4
    """
    zero_pairs = [(a, 0) for a in range(p)] + [(0, b) for b in range(1, p)]
    identity_pairs = [(a, a) for a in range(p)]
    small_pairs = [(a, b) for a in range(p // 4) for b in range(p // 4)]

    return {
        "zero_pairs": zero_pairs,
        "identity_pairs": identity_pairs,
        "small_pairs": small_pairs,
    }
