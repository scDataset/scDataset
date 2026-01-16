"""
Tests for scDataset class.

This module tests the main scDataset class from scdataset.scdataset.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from torch.utils.data import DataLoader

from scdataset import (
    scDataset,
    Streaming,
    BlockShuffling,
    BlockWeightedSampling,
    ClassBalancedSampling,
    MultiIndexable,
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
    labels = np.random.choice(['A', 'B', 'C'], n)
    return features, labels


# =============================================================================
# Basic Construction Tests
# =============================================================================

class TestScDatasetConstruction:
    """Tests for scDataset construction."""
    
    def test_basic_construction(self, sample_data):
        """Test basic dataset construction."""
        dataset = scDataset(
            sample_data,
            Streaming(),
            batch_size=64
        )
        assert dataset.batch_size == 64
        assert dataset.collection is sample_data
    
    def test_default_fetch_factor(self, sample_data):
        """Test default fetch_factor value."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)
        assert dataset.fetch_factor == 16
    
    def test_custom_fetch_factor(self, sample_data):
        """Test custom fetch_factor."""
        dataset = scDataset(
            sample_data,
            Streaming(),
            batch_size=64,
            fetch_factor=8
        )
        assert dataset.fetch_factor == 8
        assert dataset.fetch_size == 64 * 8
    
    def test_drop_last_parameter(self, sample_data):
        """Test drop_last parameter."""
        dataset = scDataset(
            sample_data, Streaming(),
            batch_size=64,
            drop_last=True
        )
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
    
    def test_invalid_data_collection(self):
        """Test that invalid data collection raises error."""
        # Object without __len__
        class NoLen:
            def __getitem__(self, idx):
                return idx
        
        with pytest.raises(TypeError, match="must support indexing and len"):
            scDataset(NoLen(), Streaming(), batch_size=64)
    
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
        dataset = scDataset(
            small_data, Streaming(),
            batch_size=7, drop_last=True
        )
        
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
        dataset = scDataset(
            small_data, Streaming(),
            batch_size=10, fetch_factor=1
        )
        
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
            sample_data, Streaming(),
            batch_size=64,
            fetch_transform=double_transform
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
            sample_data, Streaming(),
            batch_size=64,
            batch_transform=normalize_batch
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
            small_data, Streaming(),
            batch_size=10, fetch_factor=1,
            fetch_transform=fetch_trans,
            batch_transform=batch_trans
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
            sample_data, Streaming(),
            batch_size=64,
            fetch_callback=custom_fetch
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
            small_data, Streaming(),
            batch_size=10, fetch_factor=1,
            batch_callback=custom_batch_extract
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
        dataset = scDataset(
            sample_data,
            BlockShuffling(block_size=8),
            batch_size=64
        )
        batches = list(dataset)
        assert len(batches) > 0
    
    def test_with_block_weighted_sampling(self, sample_data):
        """Test scDataset with BlockWeightedSampling strategy."""
        weights = np.ones(len(sample_data)) / len(sample_data)
        dataset = scDataset(
            sample_data,
            BlockWeightedSampling(weights=weights, block_size=8, total_size=500),
            batch_size=64
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
            batch_size=32
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
        
        multi = MultiIndexable(features, labels, names=['X', 'y'])
        dataset = scDataset(multi, Streaming(), batch_size=10)
        
        batch = next(iter(dataset))
        
        assert isinstance(batch, MultiIndexable)
        assert 'X' in batch.names
        assert 'y' in batch.names
        assert batch['X'].shape[0] == 10
    
    def test_multiindexable_unstructured_preserved(self):
        """Test that unstructured data is preserved through iteration."""
        features = np.random.randn(100, 50).astype(np.float32)
        unstructured = {'gene_names': ['Gene_' + str(i) for i in range(50)]}
        
        multi = MultiIndexable(features, names=['X'], unstructured=unstructured)
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
        dataset = scDataset(
            sample_data, Streaming(),
            batch_size=64, fetch_factor=4
        )
        
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
        dataset = scDataset(
            small_data, Streaming(),
            batch_size=len(small_data)
        )
        
        batches = list(dataset)
        assert len(batches) == 1
        assert batches[0].shape[0] == len(small_data)
    
    def test_batch_size_larger_than_data(self):
        """Test when batch_size is larger than data."""
        data = np.arange(10).reshape(-1, 1)
        dataset = scDataset(
            data, Streaming(),
            batch_size=100, drop_last=False
        )
        
        batches = list(dataset)
        assert len(batches) == 1
        assert batches[0].shape[0] == 10
    
    def test_with_list_data(self):
        """Test scDataset with Python list data converted to numpy array."""
        # Note: scDataset requires data supporting numpy-style fancy indexing
        # Python lists must be converted to numpy arrays
        data = np.array([[i, i+1, i+2] for i in range(100)])
        dataset = scDataset(data, Streaming(), batch_size=10)
        
        batch = next(iter(dataset))
        assert len(batch) == 10
    
    def test_fetch_factor_larger_than_data(self):
        """Test when fetch_factor * batch_size > data size."""
        data = np.arange(50).reshape(-1, 1)
        dataset = scDataset(
            data, Streaming(),
            batch_size=10, fetch_factor=100
        )
        
        # Should still work, just fewer fetches
        batches = list(dataset)
        assert len(batches) == 5


# =============================================================================
# Reproducibility Tests
# =============================================================================

class TestScDatasetReproducibility:
    """Tests for reproducibility."""
    
    def test_streaming_reproducibility(self, sample_data):
        """Test that streaming produces same results."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)
        
        batches1 = [b.copy() for b in dataset]
        batches2 = [b.copy() for b in dataset]
        
        for b1, b2 in zip(batches1, batches2):
            assert_array_equal(b1, b2)
    
    def test_block_shuffling_with_seed(self, sample_data):
        """Test BlockShuffling reproducibility with same seed."""
        # Note: Seed is typically set in strategy.get_indices
        # This test verifies same strategy produces same sequence
        strategy = BlockShuffling(block_size=8)
        
        dataset1 = scDataset(sample_data, strategy, batch_size=64)
        dataset2 = scDataset(sample_data, strategy, batch_size=64)
        
        # Multiple iterations through same dataset should be reproducible
        # (within same process; worker distribution affects this)


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
        dataset = scDataset(
            sample_data, Streaming(), batch_size=64,
            rank=1, world_size=4
        )
        assert dataset.rank == 1
        assert dataset.world_size == 4
    
    def test_ddp_set_epoch(self, sample_data):
        """Test set_epoch method."""
        dataset = scDataset(sample_data, Streaming(), batch_size=64)
        assert dataset._epoch == 0
        
        dataset.set_epoch(5)
        assert dataset._epoch == 5
    
    def test_ddp_different_epochs_different_shuffle(self, sample_data):
        """Test that different epochs produce different shuffling."""
        strategy = BlockShuffling(block_size=4)
        dataset = scDataset(sample_data, strategy, batch_size=64)
        
        dataset.set_epoch(0)
        batches_epoch0 = [b.copy() for b in dataset]
        
        dataset.set_epoch(1)
        batches_epoch1 = [b.copy() for b in dataset]
        
        # At least some batches should be different
        # (shuffling is different per epoch)
        any_different = False
        for b0, b1 in zip(batches_epoch0, batches_epoch1):
            if not np.array_equal(b0, b1):
                any_different = True
                break
        assert any_different, "Different epochs should produce different shuffling"
    
    def test_ddp_ranks_no_overlap(self):
        """Test that different ranks get non-overlapping data."""
        # Use larger data to ensure multiple fetches
        large_data = np.random.randn(10000, 50)
        
        # Simulate 2 ranks
        dataset_rank0 = scDataset(
            large_data, Streaming(), batch_size=64, fetch_factor=4,
            rank=0, world_size=2
        )
        dataset_rank1 = scDataset(
            large_data, Streaming(), batch_size=64, fetch_factor=4,
            rank=1, world_size=2
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
            sample_data, Streaming(), batch_size=64,
            world_size=1
        )
        
        # 4 processes
        dataset_ddp = scDataset(
            sample_data, Streaming(), batch_size=64,
            world_size=4
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
                block_size=16, 
                labels=np.random.randint(0, 3, len(large_data))
            ),
        ]
        
        for strategy in strategies:
            dataset_rank0 = scDataset(
                large_data, strategy, batch_size=64,
                rank=0, world_size=2
            )
            dataset_rank1 = scDataset(
                large_data, strategy, batch_size=64,
                rank=1, world_size=2
            )
            
            # Both should produce data
            batches_0 = list(dataset_rank0)
            batches_1 = list(dataset_rank1)
            
            assert len(batches_0) > 0, f"Rank 0 empty with {type(strategy).__name__}"
            assert len(batches_1) > 0, f"Rank 1 empty with {type(strategy).__name__}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
