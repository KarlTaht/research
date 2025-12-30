"""Tests for data.py - Dataset classes and utilities."""

import pytest
import torch
from projects.least_action_learning.src.data import (
    ModularArithmeticDataset,
    SequenceArithmeticDataset,
    ModularArithmeticLoader,
    DatasetSplit,
    SequenceDatasetSplit,
    get_pair_indices,
    get_special_pairs,
)


class TestModularArithmeticDataset:
    """Tests for ModularArithmeticDataset."""

    def test_creation_default_params(self, small_p):
        """Test dataset creation with default parameters."""
        dataset = ModularArithmeticDataset(p=small_p)
        assert dataset.p == small_p
        assert dataset.operation == "add"
        assert dataset.train_frac == 0.3
        assert len(dataset) == small_p * small_p

    def test_input_dim(self, modular_dataset, small_p):
        """Test input dimension is 2*p for one-hot encoding."""
        assert modular_dataset.input_dim == 2 * small_p

    def test_output_dim(self, modular_dataset, small_p):
        """Test output dimension equals p classes."""
        assert modular_dataset.output_dim == small_p

    def test_total_examples(self, modular_dataset, small_p):
        """Test total number of examples is p^2."""
        assert len(modular_dataset) == small_p * small_p

    def test_train_test_split_sizes(self, modular_dataset, small_p):
        """Test train/test split respects train_frac."""
        train_split = modular_dataset.get_train()
        test_split = modular_dataset.get_test()

        total = small_p * small_p
        expected_train = int(total * 0.3)

        assert len(train_split.inputs) == expected_train
        assert len(test_split.inputs) == total - expected_train

    def test_train_test_no_overlap(self, modular_dataset):
        """Test train and test sets don't overlap."""
        train_pairs = set(map(tuple, modular_dataset.train_split.pairs.tolist()))
        test_pairs = set(map(tuple, modular_dataset.test_split.pairs.tolist()))
        assert train_pairs.isdisjoint(test_pairs)

    def test_train_test_complete(self, modular_dataset):
        """Test train + test covers all examples."""
        train_pairs = set(map(tuple, modular_dataset.train_split.pairs.tolist()))
        test_pairs = set(map(tuple, modular_dataset.test_split.pairs.tolist()))
        all_pairs = set(map(tuple, modular_dataset.all_pairs.tolist()))
        assert train_pairs.union(test_pairs) == all_pairs

    def test_one_hot_encoding_valid(self, modular_dataset, small_p):
        """Test one-hot encoding is valid (exactly 2 ones per row)."""
        all_data = modular_dataset.get_all()
        inputs = all_data.inputs

        # Each row should have exactly 2 ones
        row_sums = inputs.sum(dim=1)
        assert torch.allclose(row_sums, torch.full_like(row_sums, 2.0))

    def test_one_hot_encoding_positions(self, small_p):
        """Test one-hot encoding puts ones in correct positions."""
        dataset = ModularArithmeticDataset(p=small_p, seed=0)
        all_data = dataset.get_all()

        for i in range(len(all_data.pairs)):
            a, b = all_data.pairs[i].tolist()
            inputs = all_data.inputs[i]

            # Check first p dims for 'a', next p dims for 'b'
            assert inputs[a] == 1.0
            assert inputs[small_p + b] == 1.0

    def test_add_operation_correctness(self, small_p):
        """Test addition operation computes (a + b) mod p correctly."""
        dataset = ModularArithmeticDataset(p=small_p, operation="add")
        all_data = dataset.get_all()

        for i in range(len(all_data.pairs)):
            a, b = all_data.pairs[i].tolist()
            expected = (a + b) % small_p
            assert all_data.targets[i].item() == expected

    def test_multiply_operation_correctness(self, multiply_dataset, small_p):
        """Test multiplication operation computes (a * b) mod p correctly."""
        all_data = multiply_dataset.get_all()

        for i in range(len(all_data.pairs)):
            a, b = all_data.pairs[i].tolist()
            expected = (a * b) % small_p
            assert all_data.targets[i].item() == expected

    def test_invalid_operation_raises(self, small_p):
        """Test that invalid operation raises ValueError."""
        with pytest.raises(ValueError, match="Unknown operation"):
            ModularArithmeticDataset(p=small_p, operation="divide")

    def test_reproducibility_with_seed(self, small_p):
        """Test same seed produces identical splits."""
        ds1 = ModularArithmeticDataset(p=small_p, seed=123)
        ds2 = ModularArithmeticDataset(p=small_p, seed=123)

        assert torch.equal(ds1.train_split.inputs, ds2.train_split.inputs)
        assert torch.equal(ds1.test_split.inputs, ds2.test_split.inputs)
        assert torch.equal(ds1.train_split.targets, ds2.train_split.targets)
        assert torch.equal(ds1.test_split.targets, ds2.test_split.targets)

    def test_different_seeds_produce_different_splits(self, small_p):
        """Test different seeds produce different splits."""
        ds1 = ModularArithmeticDataset(p=small_p, seed=123)
        ds2 = ModularArithmeticDataset(p=small_p, seed=456)

        # Train inputs should differ (different random permutation)
        assert not torch.equal(ds1.train_split.inputs, ds2.train_split.inputs)

    def test_getitem(self, modular_dataset):
        """Test __getitem__ returns input-target pair."""
        inputs, targets = modular_dataset[0]
        assert inputs.shape == (modular_dataset.input_dim,)
        assert targets.shape == ()  # Scalar

    def test_getitem_consistency(self, modular_dataset):
        """Test __getitem__ returns data consistent with get_all."""
        all_data = modular_dataset.get_all()
        for i in range(min(10, len(modular_dataset))):
            inputs_item, targets_item = modular_dataset[i]
            assert torch.equal(inputs_item, all_data.inputs[i])
            assert torch.equal(targets_item, all_data.targets[i])

    def test_get_all_returns_all_data(self, modular_dataset, small_p):
        """Test get_all returns all p^2 examples."""
        all_data = modular_dataset.get_all()
        assert all_data.inputs.shape[0] == small_p * small_p
        assert all_data.targets.shape[0] == small_p * small_p
        assert all_data.pairs.shape[0] == small_p * small_p

    def test_edge_case_small_prime(self):
        """Test with smallest prime p=2."""
        dataset = ModularArithmeticDataset(p=2, train_frac=0.5)
        assert len(dataset) == 4
        assert dataset.input_dim == 4
        assert dataset.output_dim == 2

    def test_train_frac_zero(self, small_p):
        """Test train_frac=0.0 gives empty training set."""
        dataset = ModularArithmeticDataset(p=small_p, train_frac=0.0)
        train_split = dataset.get_train()
        test_split = dataset.get_test()

        assert len(train_split.inputs) == 0
        assert len(test_split.inputs) == small_p * small_p

    def test_train_frac_one(self, small_p):
        """Test train_frac=1.0 gives empty test set."""
        dataset = ModularArithmeticDataset(p=small_p, train_frac=1.0)
        train_split = dataset.get_train()
        test_split = dataset.get_test()

        assert len(train_split.inputs) == small_p * small_p
        assert len(test_split.inputs) == 0


class TestSequenceArithmeticDataset:
    """Tests for SequenceArithmeticDataset."""

    def test_creation(self, small_p):
        """Test sequence dataset creation."""
        dataset = SequenceArithmeticDataset(p=small_p)
        assert dataset.p == small_p
        assert dataset.vocab_size == small_p + 2  # p residues + op + equals

    def test_special_tokens(self, sequence_dataset, small_p):
        """Test special token values."""
        assert sequence_dataset.op_token == small_p
        assert sequence_dataset.eq_token == small_p + 1

    def test_vocab_size(self, sequence_dataset, small_p):
        """Test vocabulary size is p + 2."""
        assert sequence_dataset.vocab_size == small_p + 2

    def test_sequence_length(self, sequence_dataset):
        """Test sequence length is 4 (a, op, b, =)."""
        assert sequence_dataset.seq_len == 4

    def test_sequence_format(self, sequence_dataset, small_p):
        """Test sequences have format [a, op, b, =]."""
        all_data = sequence_dataset.get_all()
        input_ids = all_data.input_ids

        # Check all op tokens are correct (position 1)
        assert (input_ids[:, 1] == small_p).all()
        # Check all = tokens are correct (position 3)
        assert (input_ids[:, 3] == small_p + 1).all()
        # Check a and b are in valid range
        assert (input_ids[:, 0] < small_p).all()
        assert (input_ids[:, 2] < small_p).all()
        assert (input_ids[:, 0] >= 0).all()
        assert (input_ids[:, 2] >= 0).all()

    def test_dtype_is_long(self, sequence_dataset):
        """Test input_ids are torch.long (required for embedding)."""
        all_data = sequence_dataset.get_all()
        assert all_data.input_ids.dtype == torch.long

    def test_targets_dtype_is_long(self, sequence_dataset):
        """Test targets are torch.long."""
        all_data = sequence_dataset.get_all()
        assert all_data.targets.dtype == torch.long

    def test_targets_match_pairs(self, sequence_dataset, small_p):
        """Test targets match the operation on pairs."""
        all_data = sequence_dataset.get_all()

        for i in range(len(all_data.pairs)):
            a, b = all_data.pairs[i].tolist()
            expected = (a + b) % small_p
            assert all_data.targets[i].item() == expected

    def test_input_ids_shape(self, sequence_dataset, small_p):
        """Test input_ids shape is [N, 4]."""
        all_data = sequence_dataset.get_all()
        assert all_data.input_ids.shape == (small_p * small_p, 4)

    def test_train_test_split(self, sequence_dataset, small_p):
        """Test train/test split sizes."""
        train_split = sequence_dataset.get_train()
        test_split = sequence_dataset.get_test()

        total = small_p * small_p
        expected_train = int(total * 0.3)

        assert len(train_split.input_ids) == expected_train
        assert len(test_split.input_ids) == total - expected_train

    def test_input_dim_returns_vocab_size(self, sequence_dataset, small_p):
        """Test input_dim property returns vocab_size for compatibility."""
        assert sequence_dataset.input_dim == small_p + 2

    def test_output_dim(self, sequence_dataset, small_p):
        """Test output_dim equals p."""
        assert sequence_dataset.output_dim == small_p

    def test_getitem(self, sequence_dataset):
        """Test __getitem__ returns (input_ids, target)."""
        input_ids, target = sequence_dataset[0]
        assert input_ids.shape == (4,)  # seq_len
        assert target.shape == ()  # scalar

    def test_multiply_operation(self, small_p):
        """Test multiplication operation for sequence dataset."""
        dataset = SequenceArithmeticDataset(p=small_p, operation="multiply")
        all_data = dataset.get_all()

        for i in range(len(all_data.pairs)):
            a, b = all_data.pairs[i].tolist()
            expected = (a * b) % small_p
            assert all_data.targets[i].item() == expected


class TestModularArithmeticLoader:
    """Tests for ModularArithmeticLoader."""

    def test_train_loader_creation(self, small_p):
        """Test train loader creation."""
        loader = ModularArithmeticLoader(p=small_p, batch_size=8)
        train_loader = loader.get_train_loader()

        batch = next(iter(train_loader))
        assert len(batch) == 2  # inputs, targets
        assert batch[0].shape[0] <= 8

    def test_full_batch_mode(self, small_p):
        """Test full batch mode (batch_size=-1)."""
        loader = ModularArithmeticLoader(p=small_p, batch_size=-1)
        train_loader = loader.get_train_loader()

        batch = next(iter(train_loader))
        expected_size = int(small_p * small_p * 0.3)
        assert batch[0].shape[0] == expected_size

    def test_test_loader_not_shuffled(self, small_p):
        """Test test loader returns same order across iterations."""
        loader = ModularArithmeticLoader(p=small_p, batch_size=-1)

        test1 = list(loader.get_test_loader())
        test2 = list(loader.get_test_loader())

        assert torch.equal(test1[0][0], test2[0][0])

    def test_input_dim_property(self, modular_loader, small_p):
        """Test input_dim property returns 2*p."""
        assert modular_loader.input_dim == 2 * small_p

    def test_output_dim_property(self, modular_loader, small_p):
        """Test output_dim property returns p."""
        assert modular_loader.output_dim == small_p

    def test_full_loader(self, small_p):
        """Test get_full_loader returns all data."""
        loader = ModularArithmeticLoader(p=small_p)
        full_loader = loader.get_full_loader()

        batch = next(iter(full_loader))
        assert batch[0].shape[0] == small_p * small_p


class TestGetPairIndices:
    """Tests for get_pair_indices helper function."""

    def test_single_pair(self, small_p):
        """Test get_pair_indices with single pair."""
        indices = get_pair_indices(small_p, [(0, 0)])
        assert indices.shape == (1,)
        assert indices[0].item() == 0

    def test_multiple_pairs(self, small_p):
        """Test get_pair_indices with multiple pairs."""
        pairs = [(0, 0), (0, 1), (1, 0)]
        indices = get_pair_indices(small_p, pairs)

        expected = torch.tensor([0, 1, small_p])
        assert torch.equal(indices, expected)

    def test_corner_pair(self, small_p):
        """Test get_pair_indices with corner case (p-1, p-1)."""
        pairs = [(small_p - 1, small_p - 1)]
        indices = get_pair_indices(small_p, pairs)

        expected = (small_p - 1) * small_p + (small_p - 1)
        assert indices[0].item() == expected

    def test_index_formula(self, small_p):
        """Test index formula: idx = a * p + b."""
        for a in range(small_p):
            for b in range(small_p):
                indices = get_pair_indices(small_p, [(a, b)])
                expected = a * small_p + b
                assert indices[0].item() == expected


class TestGetSpecialPairs:
    """Tests for get_special_pairs helper function."""

    def test_returns_dict_with_required_keys(self, small_p):
        """Test get_special_pairs returns expected categories."""
        special = get_special_pairs(small_p)

        assert "zero_pairs" in special
        assert "identity_pairs" in special
        assert "small_pairs" in special

    def test_zero_pairs_content(self, small_p):
        """Test zero_pairs include (a, 0) and (0, b)."""
        special = get_special_pairs(small_p)
        zero_pairs = special["zero_pairs"]

        # Should include (0, 0)
        assert (0, 0) in zero_pairs
        # Should include (a, 0) for all a
        for a in range(small_p):
            assert (a, 0) in zero_pairs
        # Should include (0, b) for all b > 0
        for b in range(1, small_p):
            assert (0, b) in zero_pairs

    def test_identity_pairs_content(self, small_p):
        """Test identity_pairs are (a, a) for all a."""
        special = get_special_pairs(small_p)
        identity_pairs = special["identity_pairs"]

        assert len(identity_pairs) == small_p
        for a in range(small_p):
            assert (a, a) in identity_pairs

    def test_small_pairs_content(self, small_p):
        """Test small_pairs have both a, b < p/4."""
        special = get_special_pairs(small_p)
        small_pairs = special["small_pairs"]

        quarter = small_p // 4
        for a, b in small_pairs:
            assert a < quarter
            assert b < quarter


class TestDatasetSplit:
    """Tests for DatasetSplit dataclass."""

    def test_dataclass_fields(self, modular_dataset):
        """Test DatasetSplit has required fields."""
        split = modular_dataset.get_train()

        assert hasattr(split, "inputs")
        assert hasattr(split, "targets")
        assert hasattr(split, "pairs")

    def test_tensor_types(self, modular_dataset):
        """Test DatasetSplit fields are tensors."""
        split = modular_dataset.get_train()

        assert isinstance(split.inputs, torch.Tensor)
        assert isinstance(split.targets, torch.Tensor)
        assert isinstance(split.pairs, torch.Tensor)


class TestSequenceDatasetSplit:
    """Tests for SequenceDatasetSplit dataclass."""

    def test_dataclass_fields(self, sequence_dataset):
        """Test SequenceDatasetSplit has required fields."""
        split = sequence_dataset.get_train()

        assert hasattr(split, "input_ids")
        assert hasattr(split, "targets")
        assert hasattr(split, "pairs")

    def test_tensor_types(self, sequence_dataset):
        """Test SequenceDatasetSplit fields are tensors."""
        split = sequence_dataset.get_train()

        assert isinstance(split.input_ids, torch.Tensor)
        assert isinstance(split.targets, torch.Tensor)
        assert isinstance(split.pairs, torch.Tensor)

    def test_input_ids_dtype(self, sequence_dataset):
        """Test input_ids dtype is long."""
        split = sequence_dataset.get_train()
        assert split.input_ids.dtype == torch.long
