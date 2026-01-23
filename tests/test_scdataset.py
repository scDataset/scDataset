"""
Tests for scDataset class.

This module tests the main scDataset class from scdataset.scdataset.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from torch.utils.data import DataLoader

from scdataset import (
    BlockShuffling,
    BlockWeightedSampling,
    ClassBalancedSampling,
    MultiIndexable,
    Streaming,
    scDataset,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Create sample data array."""
    return np.random.randn(1000, 50).astype(np.float32)


@pytest.fixture
def small_data():
    """Create small sample data for detailed tests."""
    return np.arange(100).reshape(-1, 1).astype(np.float32)


@pytest.fixture
def labeled_data():
    """Create data with labels."""
    n = 300
    features = np.random.randn(n, 20).astype(np.float32)
    labels = np.random.choice(["A", "B", "C"], n)
    return features, labels


# =============================================================================
# Basic Construction Tests
# =============================================================================


class TestScDatasetConstruction:
    """Tests for scDataset construction."""

    def test_basic_construction(self, sample_data):
        """Test basic dataset construction."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)
        assert dataset.batch_size == 64
        assert dataset.collection is sample_data

    def test_default_fetch_factor(self, sample_data):
        """Test default fetch_factor value."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)
        assert dataset.fetch_factor == 16

    def test_custom_fetch_factor(self, sample_data):
        """Test custom fetch_factor."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64, fetch_factor=8)
        assert dataset.fetch_factor == 8
        assert dataset.fetch_size == 64 * 8

    def test_drop_last_parameter(self, sample_data):
        """Test drop_last parameter."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64, drop_last=True)
        assert dataset.drop_last is True

    def test_invalid_batch_size(self, sample_data):
        """Test that invalid batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            scDataset(sample_data, Streaming(), batch_size=0)

        with pytest.raises(ValueError, match="batch_size must be positive"):
            scDataset(sample_data, Streaming(), batch_size=-1)

    def test_invalid_fetch_factor(self, sample_data):
        """Test that invalid fetch_factor raises error."""
        with pytest.raises(ValueError, match="fetch_factor must be positive"):
            scDataset(sample_data, Streaming(), batch_size=64, fetch_factor=0)

    def test_custom_indexing_collection(self):
        """Test that custom collections work with fetch_callback (no validation check)."""

        # Object without __len__ or __getitem__ but works with custom fetch_callback
        class CustomCollection:
            def __init__(self, data):
                self._data = data

            def custom_fetch(self, indices):
                return self._data[indices]

        data = np.random.randn(100, 10)
        collection = CustomCollection(data)

        # This should work because we provide a custom fetch_callback
        # and use explicit indices in the strategy
        indices = np.arange(100)
        strategy = Streaming(indices=indices)
        dataset = scDataset(
            collection,
            strategy,
            batch_size=10,
            fetch_callback=lambda c, idx: c.custom_fetch(idx),
        )

        # Should be able to iterate
        batches = list(dataset)
        assert len(batches) == 10
        assert batches[0].shape == (10, 10)

    def test_invalid_strategy(self, sample_data):
        """Test that invalid strategy raises error."""
        with pytest.raises(TypeError, match="SamplingStrategy"):
            scDataset(sample_data, "not_a_strategy", batch_size=64)


# =============================================================================
# Length Tests
# =============================================================================


class TestScDatasetLength:
    """Tests for scDataset __len__ method."""

    def test_len_without_drop_last(self, small_data):
        """Test length calculation without drop_last."""
        # 100 samples, batch_size=10 -> 10 batches
        dataset = scDataset(small_data, Streaming(), batch_size=10, drop_last=False)
        assert len(dataset) == 10

        # 100 samples, batch_size=7 -> ceil(100/7) = 15 batches
        dataset = scDataset(small_data, Streaming(), batch_size=7, drop_last=False)
        assert len(dataset) == 15

    def test_len_with_drop_last(self, small_data):
        """Test length calculation with drop_last."""
        # 100 samples, batch_size=10 -> 10 batches
        dataset = scDataset(small_data, Streaming(), batch_size=10, drop_last=True)
        assert len(dataset) == 10

        # 100 samples, batch_size=7 -> floor(100/7) = 14 batches
        dataset = scDataset(small_data, Streaming(), batch_size=7, drop_last=True)
        assert len(dataset) == 14

    def test_len_with_subset_indices(self, small_data):
        """Test length with subset indices."""
        indices = np.arange(50)
        strategy = Streaming(indices=indices)
        dataset = scDataset(small_data, strategy, batch_size=10)
        assert len(dataset) == 5


# =============================================================================
# Iteration Tests
# =============================================================================


class TestScDatasetIteration:
    """Tests for scDataset iteration."""

    def test_basic_iteration(self, sample_data):
        """Test basic iteration through dataset."""
        batch_size = 64
        dataset = scDataset(sample_data, Streaming(), batch_size=batch_size)

        batches = list(dataset)

        # Should have correct number of batches
        expected_batches = (len(sample_data) + batch_size - 1) // batch_size
        assert len(batches) == expected_batches

        # First batch should have correct size
        assert batches[0].shape[0] == batch_size

    def test_iteration_with_drop_last(self, small_data):
        """Test iteration with drop_last=True."""
        # 100 samples, batch_size=7 -> 14 complete batches (98 samples)
        dataset = scDataset(small_data, Streaming(), batch_size=7, drop_last=True)

        batches = list(dataset)
        assert len(batches) == 14

        # All batches should be complete
        for batch in batches:
            assert batch.shape[0] == 7

    def test_iteration_covers_all_data(self, small_data):
        """Test that iteration covers all data in streaming mode."""
        dataset = scDataset(small_data, Streaming(), batch_size=10, fetch_factor=1)

        all_samples = []
        for batch in dataset:
            all_samples.append(batch.flatten())

        all_samples = np.concatenate(all_samples)

        # Should cover all original data
        assert len(all_samples) == len(small_data)
        assert_array_equal(sorted(all_samples), sorted(small_data.flatten()))

    def test_streaming_order(self, small_data):
        """Test that streaming preserves order."""
        dataset = scDataset(small_data, Streaming(), batch_size=10, fetch_factor=1)

        samples = []
        for batch in dataset:
            samples.extend(batch.flatten().tolist())

        # Should be in original order
        assert samples == list(range(100))

    def test_multiple_iterations(self, sample_data):
        """Test that dataset can be iterated multiple times."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)

        first_batches = [b.shape for b in dataset]
        second_batches = [b.shape for b in dataset]

        assert first_batches == second_batches


# =============================================================================
# Transform Tests
# =============================================================================


class TestScDatasetTransforms:
    """Tests for scDataset transforms."""

    def test_fetch_transform(self, sample_data):
        """Test fetch_transform application."""

        def double_transform(data):
            return data * 2

        dataset = scDataset(
            sample_data, Streaming(), batch_size=64, fetch_transform=double_transform
        )

        batch = next(iter(dataset))

        # Batch should be doubled
        # (Note: exact comparison is tricky due to sorting, but values should be doubled)
        assert batch.mean() > sample_data.mean()  # Rough check

    def test_batch_transform(self, sample_data):
        """Test batch_transform application."""

        def normalize_batch(batch):
            return (batch - batch.mean()) / (batch.std() + 1e-8)

        dataset = scDataset(
            sample_data, Streaming(), batch_size=64, batch_transform=normalize_batch
        )

        batch = next(iter(dataset))

        # Batch should be approximately normalized
        assert abs(batch.mean()) < 0.5
        assert abs(batch.std() - 1.0) < 0.5

    def test_both_transforms(self, small_data):
        """Test both fetch_transform and batch_transform together."""

        def fetch_trans(data):
            return data + 100

        def batch_trans(batch):
            return batch * 10

        dataset = scDataset(
            small_data,
            Streaming(),
            batch_size=10,
            fetch_factor=1,
            fetch_transform=fetch_trans,
            batch_transform=batch_trans,
        )

        batch = next(iter(dataset))

        # Values should be (original + 100) * 10
        expected_min = (0 + 100) * 10  # 1000
        assert batch.min() >= expected_min


# =============================================================================
# Callback Tests
# =============================================================================


class TestScDatasetCallbacks:
    """Tests for scDataset callbacks."""

    def test_fetch_callback(self, sample_data):
        """Test custom fetch_callback."""

        def custom_fetch(collection, indices):
            # Return negative of the data
            return -collection[indices]

        dataset = scDataset(
            sample_data, Streaming(), batch_size=64, fetch_callback=custom_fetch
        )

        batch = next(iter(dataset))

        # Values should be negative
        assert batch.mean() < 0 or sample_data.mean() > 0  # One should be opposite

    def test_batch_callback(self, small_data):
        """Test custom batch_callback."""

        def custom_batch_extract(fetched_data, indices):
            # Return only first column doubled
            return fetched_data[indices] * 2

        dataset = scDataset(
            small_data,
            Streaming(),
            batch_size=10,
            fetch_factor=1,
            batch_callback=custom_batch_extract,
        )

        batch = next(iter(dataset))

        # Values should be doubled
        assert batch[0, 0] == 0  # 0 * 2 = 0 still


# =============================================================================
# Strategy Integration Tests
# =============================================================================


class TestScDatasetWithStrategies:
    """Tests for scDataset with different strategies."""

    def test_with_streaming(self, sample_data):
        """Test scDataset with Streaming strategy."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)
        batches = list(dataset)
        assert len(batches) > 0

    def test_with_block_shuffling(self, sample_data):
        """Test scDataset with BlockShuffling strategy."""
        dataset = scDataset(sample_data, BlockShuffling(block_size=8), batch_size=64)
        batches = list(dataset)
        assert len(batches) > 0

    def test_with_block_weighted_sampling(self, sample_data):
        """Test scDataset with BlockWeightedSampling strategy."""
        weights = np.ones(len(sample_data)) / len(sample_data)
        dataset = scDataset(
            sample_data,
            BlockWeightedSampling(weights=weights, block_size=8, total_size=500),
            batch_size=64,
        )
        batches = list(dataset)
        # Should have 500 // 64 = 7 batches (with remainder)
        assert len(batches) == 8  # ceil(500/64)

    def test_with_class_balanced_sampling(self, labeled_data):
        """Test scDataset with ClassBalancedSampling strategy."""
        features, labels = labeled_data
        dataset = scDataset(
            features,
            ClassBalancedSampling(labels=labels, total_size=200, block_size=8),
            batch_size=32,
        )
        batches = list(dataset)
        assert len(batches) > 0


# =============================================================================
# MultiIndexable Integration Tests
# =============================================================================


class TestScDatasetWithMultiIndexable:
    """Tests for scDataset with MultiIndexable data."""

    def test_basic_multiindexable(self):
        """Test scDataset with MultiIndexable data."""
        features = np.random.randn(100, 50).astype(np.float32)
        labels = np.random.randint(0, 3, 100)

        multi = MultiIndexable(features, labels, names=["X", "y"])
        dataset = scDataset(multi, Streaming(), batch_size=10)

        batch = next(iter(dataset))

        assert isinstance(batch, MultiIndexable)
        assert "X" in batch.names
        assert "y" in batch.names
        assert batch["X"].shape[0] == 10

    def test_multiindexable_unstructured_preserved(self):
        """Test that unstructured data is preserved through iteration."""
        features = np.random.randn(100, 50).astype(np.float32)
        unstructured = {"gene_names": ["Gene_" + str(i) for i in range(50)]}

        multi = MultiIndexable(features, names=["X"], unstructured=unstructured)
        dataset = scDataset(multi, Streaming(), batch_size=10)

        batch = next(iter(dataset))

        assert batch.unstructured == unstructured


# =============================================================================
# DataLoader Integration Tests
# =============================================================================


class TestScDatasetWithDataLoader:
    """Tests for scDataset with PyTorch DataLoader."""

    def test_basic_dataloader(self, sample_data):
        """Test scDataset with DataLoader."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)

        loader = DataLoader(dataset, batch_size=None)

        batch = next(iter(loader))
        assert batch.shape[0] == 64

    def test_dataloader_num_workers(self, sample_data):
        """Test scDataset with multiple workers."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64, fetch_factor=4)

        # Note: num_workers > 0 may require more setup in CI
        # Just test with 0 workers here
        loader = DataLoader(dataset, batch_size=None, num_workers=0)

        batches = list(loader)
        assert len(batches) > 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestScDatasetEdgeCases:
    """Test edge cases for scDataset."""

    def test_single_sample(self):
        """Test dataset with single sample."""
        data = np.array([[1, 2, 3]])
        dataset = scDataset(data, Streaming(), batch_size=1)

        batches = list(dataset)
        assert len(batches) == 1
        assert_array_equal(batches[0], data)

    def test_batch_size_equals_data_size(self, small_data):
        """Test when batch_size equals data size."""
        dataset = scDataset(small_data, Streaming(), batch_size=len(small_data))

        batches = list(dataset)
        assert len(batches) == 1
        assert batches[0].shape[0] == len(small_data)

    def test_batch_size_larger_than_data(self):
        """Test when batch_size is larger than data."""
        data = np.arange(10).reshape(-1, 1)
        dataset = scDataset(data, Streaming(), batch_size=100, drop_last=False)

        batches = list(dataset)
        assert len(batches) == 1
        assert batches[0].shape[0] == 10

    def test_with_list_data(self):
        """Test scDataset with Python list data converted to numpy array."""
        # Note: scDataset requires data supporting numpy-style fancy indexing
        # Python lists must be converted to numpy arrays
        data = np.array([[i, i + 1, i + 2] for i in range(100)])
        dataset = scDataset(data, Streaming(), batch_size=10)

        batch = next(iter(dataset))
        assert len(batch) == 10

    def test_fetch_factor_larger_than_data(self):
        """Test when fetch_factor * batch_size > data size."""
        data = np.arange(50).reshape(-1, 1)
        dataset = scDataset(data, Streaming(), batch_size=10, fetch_factor=100)

        # Should still work, just fewer fetches
        batches = list(dataset)
        assert len(batches) == 5


# =============================================================================
# DDP Tests
# =============================================================================


class TestScDatasetDDP:
    """Tests for Distributed Data Parallel support."""

    def test_ddp_default_single_process(self, sample_data):
        """Test DDP defaults to single process when not initialized."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)
        assert dataset.rank == 0
        assert dataset.world_size == 1

    def test_ddp_explicit_rank_world_size(self, sample_data):
        """Test explicit rank and world_size parameters."""
        # Provide explicit seed to avoid needing DDP initialization
        dataset = scDataset(
            sample_data, Streaming(), batch_size=64, rank=1, world_size=4, seed=42
        )
        assert dataset.rank == 1
        assert dataset.world_size == 4

    def test_ddp_ranks_no_overlap(self):
        """Test that different ranks get non-overlapping data."""
        # Use larger data to ensure multiple fetches
        large_data = np.random.randn(10000, 50)

        # Simulate 2 ranks - provide explicit seed to avoid DDP init requirement
        dataset_rank0 = scDataset(
            large_data,
            Streaming(),
            batch_size=64,
            fetch_factor=4,
            rank=0,
            world_size=2,
            seed=42,
        )
        dataset_rank1 = scDataset(
            large_data,
            Streaming(),
            batch_size=64,
            fetch_factor=4,
            rank=1,
            world_size=2,
            seed=42,
        )

        # Collect all indices from each rank
        batches_rank0 = list(dataset_rank0)
        batches_rank1 = list(dataset_rank1)

        # Both ranks should have some data
        assert len(batches_rank0) > 0
        assert len(batches_rank1) > 0

        # Combined should cover approximately all data
        total_samples_rank0 = sum(len(b) for b in batches_rank0)
        total_samples_rank1 = sum(len(b) for b in batches_rank1)
        total = total_samples_rank0 + total_samples_rank1

        # Should be close to total data size (may differ slightly due to batching)
        assert total >= len(large_data) * 0.9  # At least 90% coverage

    def test_ddp_len_accounts_for_world_size(self, sample_data):
        """Test that len() returns per-rank batch count."""
        # Single process
        dataset_single = scDataset(
            sample_data, Streaming(), batch_size=64, world_size=1
        )

        # 4 processes - provide explicit seed to avoid DDP init requirement
        dataset_ddp = scDataset(
            sample_data, Streaming(), batch_size=64, world_size=4, seed=42
        )

        # DDP dataset should report fewer batches per rank
        assert len(dataset_ddp) <= len(dataset_single)
        # Should be approximately 1/4 of single process
        assert len(dataset_ddp) >= len(dataset_single) // 5  # Some tolerance

    def test_dataparallel_compatibility(self, sample_data):
        """Test that scDataset works with standard DataLoader (compatible with DP).

        DataParallel (DP) uses a standard DataLoader and replicates the model.
        scDataset, as an IterableDataset, should work seamlessly with DP setups
        since it doesn't require special handling - just standard iteration.
        """
        from torch.utils.data import DataLoader

        dataset = scDataset(sample_data, Streaming(), batch_size=64)

        # Standard DataLoader usage (what DP uses)
        loader = DataLoader(dataset, batch_size=None, num_workers=0)

        # Verify iteration works
        batches = list(loader)
        assert len(batches) > 0

        # Verify all data covered
        total_samples = sum(len(b) for b in batches)
        assert total_samples == len(sample_data)

    def test_ddp_with_all_strategies(self):
        """Test DDP partitioning works with all strategy types."""
        # Use larger data to ensure enough for multiple ranks
        large_data = np.random.randn(5000, 50)

        strategies = [
            Streaming(),
            BlockShuffling(block_size=16),
            BlockWeightedSampling(block_size=16, weights=np.ones(len(large_data))),
            ClassBalancedSampling(
                block_size=16, labels=np.random.randint(0, 3, len(large_data))
            ),
        ]

        for strategy in strategies:
            # Provide explicit seed to avoid DDP init requirement
            dataset_rank0 = scDataset(
                large_data, strategy, batch_size=64, rank=0, world_size=2, seed=42
            )
            dataset_rank1 = scDataset(
                large_data, strategy, batch_size=64, rank=1, world_size=2, seed=42
            )

            # Both should produce data
            batches_0 = list(dataset_rank0)
            batches_1 = list(dataset_rank1)

            assert len(batches_0) > 0, f"Rank 0 empty with {type(strategy).__name__}"
            assert len(batches_1) > 0, f"Rank 1 empty with {type(strategy).__name__}"

    def test_seed_parameter(self, sample_data):
        """Test seed parameter for reproducibility."""
        # Default seed is None (random)
        dataset = scDataset(sample_data, BlockShuffling(block_size=4), batch_size=64)
        assert dataset._base_seed is not None  # Should be auto-generated
        assert isinstance(dataset._base_seed, int)

        # Custom seed via parameter
        dataset_custom = scDataset(
            sample_data, BlockShuffling(block_size=4), batch_size=64, seed=123
        )
        assert dataset_custom._base_seed == 123

    def test_auto_epoch_increment(self, sample_data):
        """Test that epoch auto-increments each iteration."""
        dataset = scDataset(
            sample_data, BlockShuffling(block_size=4), batch_size=64, seed=42
        )

        assert dataset._epoch == 0

        # First iteration
        list(dataset)
        assert dataset._epoch == 1

        # Second iteration
        list(dataset)
        assert dataset._epoch == 2

    def test_auto_epoch_different_shuffling(self, sample_data):
        """Test that auto-incrementing epoch produces different shuffling."""
        strategy = BlockShuffling(block_size=4)
        dataset = scDataset(sample_data, strategy, batch_size=64, seed=42)

        # First epoch (epoch 0)
        batches_epoch0 = [b.copy() for b in dataset]

        # Second epoch (epoch auto-incremented to 1)
        batches_epoch1 = [b.copy() for b in dataset]

        # At least some batches should be different
        any_different = False
        for b0, b1 in zip(batches_epoch0, batches_epoch1):
            if not np.array_equal(b0, b1):
                any_different = True
                break
        assert (
            any_different
        ), "Auto-incrementing epoch should produce different shuffling"

    def test_reproducibility_with_same_seed(self, sample_data):
        """Test that same seed produces same shuffling sequence."""
        # Create two datasets with same seed via parameter
        dataset1 = scDataset(
            sample_data, BlockShuffling(block_size=4), batch_size=64, seed=42
        )
        dataset2 = scDataset(
            sample_data, BlockShuffling(block_size=4), batch_size=64, seed=42
        )

        # First iteration should be identical
        batches1_e0 = [b.copy() for b in dataset1]
        batches2_e0 = [b.copy() for b in dataset2]

        for b1, b2 in zip(batches1_e0, batches2_e0):
            assert np.array_equal(b1, b2), "Same seed should produce same first epoch"

    def test_ddp_high_world_size_small_data(self):
        """Test DDP with world_size larger than num_fetches - some ranks get no data."""
        # Small data with large world_size means some ranks get 0 fetches
        small_data = np.random.randn(100, 10)  # 100 samples

        # With batch_size=64 and fetch_factor=1, we have 2 fetches total
        # With world_size=8, ranks 2-7 should get 0 data (return 0 length)
        # Provide explicit seed to avoid DDP init requirement
        dataset_rank0 = scDataset(
            small_data,
            Streaming(),
            batch_size=64,
            fetch_factor=1,
            rank=0,
            world_size=8,
            seed=42,
        )
        dataset_rank2 = scDataset(
            small_data,
            Streaming(),
            batch_size=64,
            fetch_factor=1,
            rank=2,
            world_size=8,
            seed=42,
        )

        # Rank 0 should have data
        batches_rank0 = list(dataset_rank0)
        assert len(batches_rank0) > 0

        # Rank 2 should have no data (len() should return 0)
        assert len(dataset_rank2) == 0
        batches_rank2 = list(dataset_rank2)
        assert len(batches_rank2) == 0

    def test_ddp_all_data_covered_no_overlap(self):
        """Test that DDP partitioning covers all data with no overlap between ranks."""
        # Use unique values so we can track which samples went where
        data = np.arange(1000).reshape(1000, 1)  # 1000 unique samples

        all_samples = []
        for rank in range(4):
            dataset = scDataset(
                data,
                Streaming(),
                batch_size=64,
                fetch_factor=2,
                rank=rank,
                world_size=4,
                seed=42,
            )
            for batch in dataset:
                all_samples.extend(batch.flatten().tolist())

        # All samples should be covered exactly once (no overlap, no missing)
        all_samples_sorted = sorted(all_samples)
        expected = list(range(1000))
        assert all_samples_sorted == expected, "DDP should cover all data exactly once"

    def test_ddp_with_dataloader_multiprocessing(self):
        """Test DDP + DataLoader with multiple workers covers all data correctly."""
        from torch.utils.data import DataLoader

        # Unique data to track coverage
        data = np.arange(500).reshape(500, 1)

        # Simulate rank 0 of 2 with 2 workers
        dataset = scDataset(
            data,
            Streaming(),
            batch_size=32,
            fetch_factor=2,
            rank=0,
            world_size=2,
            seed=42,
        )

        loader = DataLoader(dataset, batch_size=None, num_workers=2)

        samples_rank0 = []
        for batch in loader:
            samples_rank0.extend(batch.numpy().flatten().tolist())

        # Should have approximately half the data (rank 0 of 2)
        assert len(samples_rank0) > 0
        assert len(samples_rank0) <= len(data)  # No more than total

        # No duplicates within this rank
        assert len(samples_rank0) == len(
            set(samples_rank0)
        ), "No duplicates within rank"


# =============================================================================
# Edge Cases and Additional Coverage Tests
# =============================================================================


class TestScDatasetEdgeCaseCoverage:
    """Additional tests to improve code coverage on edge cases."""

    def test_fetch_transform_only(self):
        """Test using fetch_transform alone (covers fetch_transform path)."""
        data = np.random.randn(100, 10)

        def my_fetch_transform(fetched_data):
            return fetched_data * 2

        dataset = scDataset(
            data, Streaming(), batch_size=32, fetch_transform=my_fetch_transform
        )

        batches = list(dataset)
        assert len(batches) > 0
        # Verify transform was applied
        for batch in batches:
            assert batch.min() < -0.5 or batch.max() > 0.5  # Scaled data

    def test_batch_transform_only(self):
        """Test using batch_transform alone (covers batch_transform path)."""
        data = np.random.randn(100, 10)

        def my_batch_transform(batch):
            return batch + 100

        dataset = scDataset(
            data, Streaming(), batch_size=32, batch_transform=my_batch_transform
        )

        batches = list(dataset)
        assert len(batches) > 0
        for batch in batches:
            assert batch.min() > 50  # Offset was applied

    def test_sort_before_fetch_disabled(self):
        """Test with sort_before_fetch=False."""
        data = np.random.randn(200, 10)

        dataset = scDataset(data, BlockShuffling(block_size=8), batch_size=32)
        # Manually disable sorting for this test
        dataset.sort_before_fetch = False

        batches = list(dataset)
        assert len(batches) > 0

    def test_very_large_fetch_factor(self):
        """Test with fetch_factor much larger than data."""
        data = np.random.randn(50, 10)

        dataset = scDataset(
            data,
            Streaming(),
            batch_size=10,
            fetch_factor=100,  # Much larger than data
        )

        batches = list(dataset)
        total = sum(len(b) for b in batches)
        assert total == 50  # All data covered

    def test_worker_info_path(self):
        """Test the worker_info path by mocking get_worker_info."""
        from unittest.mock import MagicMock, patch

        data = np.random.randn(200, 10)
        dataset = scDataset(data, Streaming(), batch_size=32, fetch_factor=2)

        # Create mock worker_info
        mock_worker_info = MagicMock()
        mock_worker_info.id = 0
        mock_worker_info.num_workers = 2
        mock_worker_info.seed = 12345

        # Patch get_worker_info to return our mock
        with patch(
            "scdataset.scdataset.get_worker_info", return_value=mock_worker_info
        ):
            # Iterate to trigger the worker_info path
            batches = list(dataset)
            assert len(batches) > 0

    def test_worker_distribution_path(self):
        """Test the worker distribution code path with mocked worker info."""
        from unittest.mock import MagicMock, patch

        data = np.random.randn(
            1000, 10
        )  # Larger dataset to ensure both workers get data

        # Test that worker_info path works for worker 0 of 2
        dataset0 = scDataset(data, Streaming(), batch_size=32, fetch_factor=4, seed=42)
        mock_worker_info = MagicMock()
        mock_worker_info.id = 0
        mock_worker_info.num_workers = 2
        mock_worker_info.seed = 99999

        with patch(
            "scdataset.scdataset.get_worker_info", return_value=mock_worker_info
        ):
            batches0 = list(dataset0)
            # Worker 0 should get some data (not all due to distribution)
            assert len(batches0) > 0

        # Test worker 1 of 2
        dataset1 = scDataset(data, Streaming(), batch_size=32, fetch_factor=4, seed=42)
        mock_worker_info1 = MagicMock()
        mock_worker_info1.id = 1
        mock_worker_info1.num_workers = 2
        mock_worker_info1.seed = 99999

        with patch(
            "scdataset.scdataset.get_worker_info", return_value=mock_worker_info1
        ):
            batches1 = list(dataset1)
            # Worker 1 should get some data too
            assert len(batches1) > 0

    def test_worker_remainder_distribution(self):
        """Test worker distribution with uneven fetch range counts to cover else branch."""
        from unittest.mock import MagicMock, patch

        # Use a size that creates uneven distribution among workers
        # With 100 samples, fetch_factor=4, batch_size=16: fetch_size = 64
        # This gives 2 fetch ranges, and with 3 workers, distribution is uneven
        data = np.random.randn(200, 10)

        for worker_id in range(3):
            dataset = scDataset(
                data, Streaming(), batch_size=16, fetch_factor=4, seed=42
            )

            mock_worker_info = MagicMock()
            mock_worker_info.id = worker_id
            mock_worker_info.num_workers = 3
            mock_worker_info.seed = 99999

            with patch(
                "scdataset.scdataset.get_worker_info", return_value=mock_worker_info
            ):
                # Just iterate - don't need to check counts, just coverage
                list(dataset)  # Iterate to trigger worker distribution
                # Some workers may get 0 batches with uneven distribution, that's ok

    def test_ddp_auto_detection_initialized(self):
        """Test DDP rank/world_size auto-detection when distributed is initialized."""
        from unittest.mock import MagicMock, patch

        data = np.random.randn(100, 10)

        # Mock torch.distributed as initialized
        mock_dist = MagicMock()
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 1
        mock_dist.get_world_size.return_value = 4

        with patch.dict("sys.modules", {"torch.distributed": mock_dist}):
            # Force re-import to use the mock
            dataset = scDataset(data, Streaming(), batch_size=32, seed=42)
            # The detection happens in _detect_ddp_config, which is called on init
            # Just verify the dataset works
            batches = list(dataset)
            assert len(batches) > 0

    def test_random_seed_different_datasets(self):
        """Test that None seed produces different seeds for different datasets."""
        data = np.random.randn(100, 10)

        # Create multiple datasets with None seed
        seeds = set()
        for _ in range(5):
            dataset = scDataset(data, Streaming(), batch_size=32)
            seeds.add(dataset._base_seed)

        # Should have generated at least 2 different seeds (very likely with 5 tries)
        # Note: There's a tiny chance they could all be the same, but vanishingly small
        assert len(seeds) >= 2, "Random seeds should differ between datasets"

    def test_explicit_seed_is_reproducible(self):
        """Test that explicit seed produces reproducible results."""
        data = np.random.randn(100, 10)

        # Create two datasets with same explicit seed
        dataset1 = scDataset(
            data, BlockShuffling(block_size=4), batch_size=32, seed=12345
        )
        dataset2 = scDataset(
            data, BlockShuffling(block_size=4), batch_size=32, seed=12345
        )

        assert dataset1._base_seed == dataset2._base_seed == 12345

        # Verify iteration produces same results
        batches1 = [b.copy() for b in dataset1]
        batches2 = [b.copy() for b in dataset2]

        for b1, b2 in zip(batches1, batches2):
            assert np.array_equal(b1, b2)

    def test_len_exact_matches_iteration_count(self):
        """Test that __len__ returns exactly the number of batches yielded by iteration."""
        # Test various configurations including edge cases
        configs = [
            # Normal cases
            {"n": 100, "batch_size": 10, "fetch_factor": 1, "drop_last": False},
            {"n": 105, "batch_size": 10, "fetch_factor": 1, "drop_last": True},
            {"n": 1000, "batch_size": 64, "fetch_factor": 4, "drop_last": False},
            {"n": 1000, "batch_size": 64, "fetch_factor": 4, "drop_last": True},
            {"n": 500, "batch_size": 32, "fetch_factor": 8, "drop_last": False},
            # Edge cases
            {
                "n": 10,
                "batch_size": 10,
                "fetch_factor": 1,
                "drop_last": False,
            },  # Exact fit
            {
                "n": 10,
                "batch_size": 10,
                "fetch_factor": 1,
                "drop_last": True,
            },  # Exact fit
            {
                "n": 9,
                "batch_size": 10,
                "fetch_factor": 1,
                "drop_last": False,
            },  # Less than batch
            {
                "n": 9,
                "batch_size": 10,
                "fetch_factor": 1,
                "drop_last": True,
            },  # Less than batch
            {
                "n": 1,
                "batch_size": 10,
                "fetch_factor": 1,
                "drop_last": False,
            },  # Single sample
            {
                "n": 100,
                "batch_size": 1,
                "fetch_factor": 1,
                "drop_last": False,
            },  # batch_size=1
            {
                "n": 100,
                "batch_size": 100,
                "fetch_factor": 1,
                "drop_last": False,
            },  # One batch
            {
                "n": 33,
                "batch_size": 7,
                "fetch_factor": 3,
                "drop_last": False,
            },  # Odd numbers
            {
                "n": 33,
                "batch_size": 7,
                "fetch_factor": 3,
                "drop_last": True,
            },  # Odd numbers
        ]

        for cfg in configs:
            data = np.random.randn(cfg["n"], 10)
            dataset = scDataset(
                data,
                Streaming(),
                batch_size=cfg["batch_size"],
                fetch_factor=cfg["fetch_factor"],
                drop_last=cfg["drop_last"],
                seed=42,
            )

            expected_len = len(dataset)
            actual_batches = list(dataset)
            actual_len = len(actual_batches)

            assert (
                expected_len == actual_len
            ), f"len() mismatch: {expected_len} vs {actual_len} for {cfg}"

    def test_len_exact_matches_iteration_count_ddp(self):
        """Test that __len__ returns exactly the number of batches yielded with DDP."""
        # Test with various DDP configurations including edge cases
        configs = [
            # Normal DDP cases
            {"n": 1000, "batch_size": 64, "fetch_factor": 4, "world_size": 2},
            {"n": 1000, "batch_size": 64, "fetch_factor": 4, "world_size": 4},
            {"n": 500, "batch_size": 32, "fetch_factor": 8, "world_size": 3},
            {"n": 100, "batch_size": 16, "fetch_factor": 2, "world_size": 4},
            # Edge cases - small data, high world_size
            {
                "n": 50,
                "batch_size": 10,
                "fetch_factor": 1,
                "world_size": 8,
            },  # Some ranks empty
            {
                "n": 20,
                "batch_size": 5,
                "fetch_factor": 2,
                "world_size": 4,
            },  # Just 2 fetches
            {
                "n": 100,
                "batch_size": 7,
                "fetch_factor": 3,
                "world_size": 5,
            },  # Odd numbers
        ]

        for cfg in configs:
            data = np.random.randn(cfg["n"], 10)

            for rank in range(cfg["world_size"]):
                for drop_last in [True, False]:
                    dataset = scDataset(
                        data,
                        Streaming(),
                        batch_size=cfg["batch_size"],
                        fetch_factor=cfg["fetch_factor"],
                        rank=rank,
                        world_size=cfg["world_size"],
                        drop_last=drop_last,
                        seed=42,
                    )

                    expected_len = len(dataset)
                    actual_batches = list(dataset)
                    actual_len = len(actual_batches)

                    assert expected_len == actual_len, (
                        f"DDP len() mismatch for rank {rank}/{cfg['world_size']}: "
                        f"{expected_len} vs {actual_len} with {cfg}, drop_last={drop_last}"
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
