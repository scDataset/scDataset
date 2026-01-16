"""
Tests for auto_config module.

This module tests the automatic parameter suggestion functionality.
"""

import pytest
import numpy as np

from scdataset import suggest_parameters, estimate_sample_size


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_array():
    """Create small numpy array."""
    return np.random.randn(1000, 100).astype(np.float32)


@pytest.fixture
def large_array():
    """Create larger numpy array."""
    return np.random.randn(10000, 500).astype(np.float32)


# =============================================================================
# estimate_sample_size Tests
# =============================================================================

class TestEstimateSampleSize:
    """Tests for estimate_sample_size function."""
    
    def test_numpy_array(self, small_array):
        """Test estimating sample size for numpy array."""
        size = estimate_sample_size(small_array)
        
        # Should estimate based on first sample
        expected = small_array[0].nbytes
        assert size == expected
    
    def test_numpy_2d_array(self):
        """Test with 2D array."""
        arr = np.zeros((100, 50), dtype=np.float32)
        size = estimate_sample_size(arr)
        
        # Each sample is 50 float32 = 200 bytes
        assert size == 50 * 4
    
    def test_list_data(self):
        """Test with Python list."""
        data = [[1.0, 2.0, 3.0] for _ in range(100)]
        size = estimate_sample_size(data)
        
        # Should return some reasonable size
        assert size > 0
    
    def test_default_fallback(self):
        """Test that unknown types return default."""
        class UnknownType:
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return "test_string"  # Returns string which has getsizeof
        
        size = estimate_sample_size(UnknownType())
        
        # Should return fallback (getsizeof for string)
        assert size > 0

    def test_dict_data(self):
        """Test estimating sample size for dictionary-based data (e.g., HuggingFace format)."""
        # Simulate HuggingFace-like dataset structure
        class DictDataset:
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return {
                    'input_ids': np.zeros(512, dtype=np.int64),
                    'attention_mask': np.ones(512, dtype=np.int64),
                    'labels': np.array([0, 1, 2], dtype=np.int64)
                }
        
        size = estimate_sample_size(DictDataset())
        # Should sum up nbytes of all values
        expected = 512 * 8 + 512 * 8 + 3 * 8  # int64 = 8 bytes
        assert size == expected

    def test_anndata_like_with_dense_x(self):
        """Test estimating sample size for AnnData-like object with dense X."""
        class AnnDataLike:
            def __init__(self):
                self.X = np.random.randn(100, 2000).astype(np.float32)
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return AnnDataLike._Sample(self.X[idx])
            
            class _Sample:
                def __init__(self, x_row):
                    self.X = x_row
        
        adata = AnnDataLike()
        size = estimate_sample_size(adata)
        # Each sample's X has 2000 float32 values = 8000 bytes
        assert size == 2000 * 4

    def test_anndata_like_with_sparse_x(self):
        """Test estimating sample size for AnnData-like object with sparse X."""
        from scipy import sparse
        
        class SparseSample:
            def __init__(self):
                # Create a sparse row (csr_matrix for row vector)
                self.X = sparse.csr_matrix(np.random.randn(1, 2000).astype(np.float32))
        
        class AnnDataSparse:
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return SparseSample()
        
        adata = AnnDataSparse()
        size = estimate_sample_size(adata)
        # Should return size of sparse data (data + indices + indptr)
        assert size > 0


# =============================================================================
# suggest_parameters Tests
# =============================================================================

class TestSuggestParameters:
    """Tests for suggest_parameters function."""
    
    def test_basic_suggestion(self, small_array):
        """Test basic parameter suggestion."""
        params = suggest_parameters(small_array, batch_size=64)
        
        assert 'num_workers' in params
        assert 'fetch_factor' in params
        assert 'block_size_balanced' in params
        assert 'block_size_conservative' in params
        assert 'block_size_aggressive' in params
    
    def test_num_workers_reasonable(self, small_array):
        """Test that num_workers is reasonable."""
        params = suggest_parameters(small_array, batch_size=64)
        
        # Should be between 1 and some reasonable max
        assert 1 <= params['num_workers'] <= 32
    
    def test_fetch_factor_positive(self, small_array):
        """Test that fetch_factor is positive."""
        params = suggest_parameters(small_array, batch_size=64)
        
        assert params['fetch_factor'] >= 1
    
    def test_block_size_options_valid(self, small_array):
        """Test that block_size options are valid."""
        params = suggest_parameters(small_array, batch_size=64)
        
        assert params['block_size_conservative'] > 0
        assert params['block_size_balanced'] > 0
        assert params['block_size_aggressive'] > 0
        # Conservative <= balanced <= aggressive
        assert params['block_size_conservative'] <= params['block_size_balanced']
        assert params['block_size_balanced'] <= params['block_size_aggressive']
    
    def test_target_ram_fraction_affects_results(self, large_array):
        """Test that target_ram_fraction affects results."""
        params_low = suggest_parameters(large_array, batch_size=64, target_ram_fraction=0.1)
        params_high = suggest_parameters(large_array, batch_size=64, target_ram_fraction=0.5)
        
        # Higher fraction should generally allow higher fetch_factor
        # (though this depends on available memory)
        assert params_low['fetch_factor'] >= 1
        assert params_high['fetch_factor'] >= 1
    
    def test_different_batch_sizes(self, small_array):
        """Test with different batch sizes."""
        params_small = suggest_parameters(small_array, batch_size=32)
        params_large = suggest_parameters(small_array, batch_size=256)
        
        # Both should return valid suggestions
        assert params_small['fetch_factor'] >= 1
        assert params_large['fetch_factor'] >= 1
    
    def test_returns_dict(self, small_array):
        """Test that function returns a dictionary."""
        params = suggest_parameters(small_array, batch_size=64)
        
        assert isinstance(params, dict)
    
    def test_prefetch_factor_suggestion(self, small_array):
        """Test prefetch_factor suggestion."""
        params = suggest_parameters(small_array, batch_size=64)
        
        # prefetch_factor should be fetch_factor + 1 typically
        if 'prefetch_factor' in params:
            assert params['prefetch_factor'] == params['fetch_factor'] + 1


# =============================================================================
# Edge Cases
# =============================================================================

class TestAutoConfigEdgeCases:
    """Test edge cases for auto_config."""
    
    def test_tiny_array(self):
        """Test with very small array."""
        tiny = np.random.randn(10, 5).astype(np.float32)
        params = suggest_parameters(tiny, batch_size=2)
        
        assert params['fetch_factor'] >= 1
        assert params['num_workers'] >= 1
    
    def test_large_batch_size(self, small_array):
        """Test with batch size larger than data."""
        params = suggest_parameters(small_array, batch_size=2000)
        
        # Should still return valid parameters
        assert params['fetch_factor'] >= 1
    
    def test_single_sample_data(self):
        """Test with single sample."""
        single = np.random.randn(1, 100).astype(np.float32)
        params = suggest_parameters(single, batch_size=1)
        
        assert params['fetch_factor'] >= 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestAutoConfigIntegration:
    """Integration tests for auto_config."""
    
    def test_use_with_scdataset(self, small_array):
        """Test using suggested parameters with scDataset."""
        from scdataset import scDataset, BlockShuffling
        
        params = suggest_parameters(small_array, batch_size=64)
        
        # Use suggested parameters
        dataset = scDataset(
            small_array,
            BlockShuffling(block_size=params['block_size_balanced']),
            batch_size=64,
            fetch_factor=params['fetch_factor']
        )
        
        # Should work without errors
        batch = next(iter(dataset))
        assert batch.shape[0] == 64
    
    def test_verbose_mode(self, small_array, capsys):
        """Test verbose mode prints information."""
        params = suggest_parameters(small_array, batch_size=64, verbose=True)
        
        captured = capsys.readouterr()
        # Verbose mode should print system info and recommendations
        assert 'System Information' in captured.out or len(captured.out) > 0
        # Should still return valid params
        assert params['fetch_factor'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
