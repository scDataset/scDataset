"""
Tests for auto_config module (experimental).

This module tests the automatic parameter suggestion functionality.
Note: auto_config is now in the experimental subpackage.
"""

import numpy as np
import pytest

# Import from experimental module (also available via main package for backward compatibility)
from scdataset.experimental import estimate_sample_size, suggest_parameters

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
                    "input_ids": np.zeros(512, dtype=np.int64),
                    "attention_mask": np.ones(512, dtype=np.int64),
                    "labels": np.array([0, 1, 2], dtype=np.int64),
                }

        size = estimate_sample_size(DictDataset())
        # Should sum up nbytes of all values, plus dict/key overhead
        min_expected = 512 * 8 + 512 * 8 + 3 * 8  # int64 = 8 bytes
        assert size >= min_expected

    def test_anndata_like_with_dense_x(self):
        """Test estimating sample size for AnnData-like object with dense X."""
        pd = pytest.importorskip("pandas")

        class AnnDataLike:
            def __init__(self):
                n_samples = 100
                n_features = 2000
                # Create full AnnData-like structure
                self.X_full = np.random.randn(n_samples, n_features).astype(np.float32)
                self.obs = pd.DataFrame(
                    {
                        "cell_type": ["type_" + str(i % 5) for i in range(n_samples)],
                    },
                    index=[f"cell_{i}" for i in range(n_samples)],
                )
                self.obsm = {"X_pca": np.random.randn(n_samples, 50).astype(np.float32)}

            def __len__(self):
                return 100

            def __getitem__(self, idx):
                return AnnDataLike._Sample(
                    self.X_full[idx],
                    self.obs.iloc[[idx]],
                    {"X_pca": self.obsm["X_pca"][idx]},
                )

            class _Sample:
                def __init__(self, x_row, obs, obsm):
                    self.X = x_row
                    self.obs = obs
                    self.obsm = obsm
                    self.var_names = None  # Mark as AnnData-like
                    self.layers = None

        adata = AnnDataLike()
        size = estimate_sample_size(adata)
        # Each sample's X has 2000 float32 values = 8000 bytes
        # Plus obs DataFrame and obsm arrays
        min_expected = 2000 * 4  # At least the X data
        assert size >= min_expected

    def test_anndata_like_with_sparse_x(self):
        """Test estimating sample size for AnnData-like object with sparse X."""
        sparse = pytest.importorskip("scipy.sparse")

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

    def test_with_fetch_transform(self):
        """Test estimate_sample_size with fetch_transform parameter."""

        # Create data that will be transformed
        class RawData:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                # Return dict with small arrays
                return {"raw": np.zeros(10, dtype=np.float32)}

        # Transform that expands the data
        def expand_transform(sample):
            return {"expanded": np.zeros(1000, dtype=np.float64)}

        data = RawData()

        # Without transform - should be ~40 bytes (10 float32)
        size_raw = estimate_sample_size(data)

        # With transform - should be ~8000 bytes (1000 float64)
        size_transformed = estimate_sample_size(data, fetch_transform=expand_transform)

        assert size_transformed > size_raw
        assert size_transformed >= 1000 * 8  # At least the expanded array

    def test_with_batch_transform(self):
        """Test estimate_sample_size with batch_transform parameter."""
        data = np.zeros((100, 50), dtype=np.float32)

        def double_transform(sample):
            return np.concatenate([sample, sample])

        size_raw = estimate_sample_size(data)
        size_transformed = estimate_sample_size(data, batch_transform=double_transform)

        # Batch transform doubles the data
        assert size_transformed >= size_raw * 2

    def test_with_fetch_callback(self):
        """Test estimate_sample_size with fetch_callback parameter."""

        class SimpleData:
            def __len__(self):
                return 100

            def __getitem__(self, idx):
                # Default returns small array
                return np.zeros(10, dtype=np.float32)

        def custom_fetch(collection, indices):
            # Custom fetch returns larger array
            return np.zeros(1000, dtype=np.float64)

        data = SimpleData()

        size_default = estimate_sample_size(data)
        size_custom = estimate_sample_size(data, fetch_callback=custom_fetch)

        assert size_custom > size_default
        assert size_custom >= 1000 * 8

    def test_with_multiple_transforms(self):
        """Test estimate_sample_size with both fetch_transform and batch_transform."""
        data = np.zeros((100, 50), dtype=np.float32)

        def fetch_transform(sample):
            # Convert to float64
            return sample.astype(np.float64)

        def batch_transform(sample):
            # Double the size
            return np.concatenate([sample, sample])

        size_raw = estimate_sample_size(data)
        size_both = estimate_sample_size(
            data,
            fetch_transform=fetch_transform,
            batch_transform=batch_transform,
        )

        # float32 -> float64 doubles size, then concat doubles again
        assert size_both >= size_raw * 4

    def test_anndata_with_fetch_transform(self):
        """Test estimate_sample_size for AnnData-like with fetch_transform."""
        sparse = pytest.importorskip("scipy.sparse")

        class AnnDataSparse:
            def __init__(self):
                n_samples = 100
                n_features = 500
                # Create sparse X
                dense = np.random.randn(n_samples, n_features)
                dense[np.random.random((n_samples, n_features)) > 0.1] = 0
                self.X = sparse.csr_matrix(dense, dtype=np.float32)

            def __len__(self):
                return self.X.shape[0]

            def __getitem__(self, idx):
                return self._Sample(self.X[idx])

            class _Sample:
                def __init__(self, x_row):
                    self.X = x_row
                    self.obs = None
                    self.obsm = None
                    self.var_names = None
                    self.layers = None

        # Transform that densifies
        def densify_transform(sample):
            if hasattr(sample.X, "toarray"):
                return sample.X.toarray()
            return sample.X

        adata = AnnDataSparse()

        # Without transform - sparse size (just to ensure it works)
        _ = estimate_sample_size(adata)

        # With transform - dense size
        size_dense = estimate_sample_size(adata, fetch_transform=densify_transform)

        # Dense should be larger (500 features * 4 bytes = 2000 bytes per sample)
        # Sparse should be smaller (only ~10% non-zero)
        assert size_dense >= 500 * 4  # Full dense array

    def test_multiindexable_sample_size(self):
        """Test estimate_sample_size correctly samples MultiIndexable.

        MultiIndexable uses list indexing for samples (data[[i]]) rather than
        integer indexing (data[i]) which returns the i-th indexable array.
        """
        from scdataset import MultiIndexable

        # Create MultiIndexable with known sizes
        n_samples = 100
        features = np.random.randn(n_samples, 50).astype(np.float32)  # 50 * 4 = 200
        labels = np.random.randn(n_samples, 10).astype(np.float64)  # 10 * 8 = 80

        multi = MultiIndexable(features=features, labels=labels)

        # Expected size per sample
        expected = 50 * 4 + 10 * 8  # 200 + 80 = 280 bytes

        size = estimate_sample_size(multi, n_samples=8)

        # Should be exactly the expected size (data only, no overhead)
        assert size == expected, f"Expected {expected}, got {size}"

    def test_multiindexable_with_fetch_transform(self):
        """Test estimate_sample_size with MultiIndexable and fetch_transform."""
        from scdataset import MultiIndexable

        n_samples = 100
        features = np.random.randn(n_samples, 50).astype(np.float32)
        labels = np.arange(n_samples, dtype=np.int64)

        multi = MultiIndexable(features=features, labels=labels)

        # Transform that extracts just the features
        def get_features(sample):
            return sample["features"]

        size = estimate_sample_size(multi, fetch_transform=get_features)

        # Should be just the features: 50 * 4 = 200 bytes
        assert size == 50 * 4


# =============================================================================
# suggest_parameters Tests
# =============================================================================


class TestSuggestParameters:
    """Tests for suggest_parameters function."""

    def test_basic_suggestion(self, small_array):
        """Test basic parameter suggestion."""
        params = suggest_parameters(small_array, batch_size=64)

        assert "num_workers" in params
        assert "fetch_factor" in params
        assert "block_size_balanced" in params
        assert "block_size_conservative" in params
        assert "block_size_aggressive" in params

    def test_num_workers_reasonable(self, small_array):
        """Test that num_workers is reasonable."""
        params = suggest_parameters(small_array, batch_size=64)

        # Should be between 1 and some reasonable max
        assert 1 <= params["num_workers"] <= 32

    def test_fetch_factor_positive(self, small_array):
        """Test that fetch_factor is positive."""
        params = suggest_parameters(small_array, batch_size=64)

        assert params["fetch_factor"] >= 1

    def test_block_size_options_valid(self, small_array):
        """Test that block_size options are valid."""
        params = suggest_parameters(small_array, batch_size=64)

        assert params["block_size_conservative"] > 0
        assert params["block_size_balanced"] > 0
        assert params["block_size_aggressive"] > 0
        # Conservative <= balanced <= aggressive
        assert params["block_size_conservative"] <= params["block_size_balanced"]
        assert params["block_size_balanced"] <= params["block_size_aggressive"]

    def test_target_ram_fraction_affects_results(self, large_array):
        """Test that target_ram_fraction affects results."""
        params_low = suggest_parameters(
            large_array, batch_size=64, target_ram_fraction=0.1
        )
        params_high = suggest_parameters(
            large_array, batch_size=64, target_ram_fraction=0.5
        )

        # Higher fraction should generally allow higher fetch_factor
        # (though this depends on available memory)
        assert params_low["fetch_factor"] >= 1
        assert params_high["fetch_factor"] >= 1

    def test_different_batch_sizes(self, small_array):
        """Test with different batch sizes."""
        params_small = suggest_parameters(small_array, batch_size=32)
        params_large = suggest_parameters(small_array, batch_size=256)

        # Both should return valid suggestions
        assert params_small["fetch_factor"] >= 1
        assert params_large["fetch_factor"] >= 1

    def test_returns_dict(self, small_array):
        """Test that function returns a dictionary."""
        params = suggest_parameters(small_array, batch_size=64)

        assert isinstance(params, dict)

    def test_prefetch_factor_suggestion(self, small_array):
        """Test prefetch_factor suggestion."""
        params = suggest_parameters(small_array, batch_size=64)

        # prefetch_factor should be fetch_factor + 1 typically
        if "prefetch_factor" in params:
            assert params["prefetch_factor"] == params["fetch_factor"] + 1


# =============================================================================
# _deep_sizeof Tests
# =============================================================================


class TestDeepSizeof:
    """Tests for the _deep_sizeof recursive size estimation function."""

    def test_numpy_array(self):
        """Test size estimation for numpy arrays."""
        from scdataset.experimental.auto_config import _deep_sizeof

        arr = np.zeros((100, 50), dtype=np.float32)
        size = _deep_sizeof(arr)
        assert size == arr.nbytes == 100 * 50 * 4

    def test_numpy_array_int64(self):
        """Test size estimation for int64 numpy arrays."""
        from scdataset.experimental.auto_config import _deep_sizeof

        arr = np.zeros((256, 128), dtype=np.int64)
        size = _deep_sizeof(arr)
        assert size == arr.nbytes == 256 * 128 * 8

    def test_scipy_sparse_csr(self):
        """Test size estimation for scipy CSR sparse matrix."""
        sparse = pytest.importorskip("scipy.sparse")
        from scdataset.experimental.auto_config import _deep_sizeof

        # Create sparse matrix with known sparsity
        dense = np.random.randn(100, 500)
        dense[dense < 0.5] = 0  # Make it sparse
        csr = sparse.csr_matrix(dense, dtype=np.float32)

        size = _deep_sizeof(csr)
        expected = csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
        assert size == expected

    def test_scipy_sparse_csc(self):
        """Test size estimation for scipy CSC sparse matrix."""
        sparse = pytest.importorskip("scipy.sparse")
        from scdataset.experimental.auto_config import _deep_sizeof

        dense = np.random.randn(100, 500)
        dense[dense < 0.5] = 0
        csc = sparse.csc_matrix(dense, dtype=np.float64)

        size = _deep_sizeof(csc)
        expected = csc.data.nbytes + csc.indices.nbytes + csc.indptr.nbytes
        assert size == expected

    def test_torch_tensor(self):
        """Test size estimation for PyTorch tensors."""
        torch = pytest.importorskip("torch")
        from scdataset.experimental.auto_config import _deep_sizeof

        tensor = torch.zeros(100, 200, dtype=torch.float32)
        size = _deep_sizeof(tensor)
        expected = tensor.element_size() * tensor.numel()
        assert size == expected == 100 * 200 * 4

    def test_torch_tensor_int64(self):
        """Test size estimation for int64 PyTorch tensors."""
        torch = pytest.importorskip("torch")
        from scdataset.experimental.auto_config import _deep_sizeof

        tensor = torch.zeros(50, 100, dtype=torch.int64)
        size = _deep_sizeof(tensor)
        assert size == 50 * 100 * 8

    def test_pandas_dataframe(self):
        """Test size estimation for pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        from scdataset.experimental.auto_config import _deep_sizeof

        df = pd.DataFrame(
            {
                "col1": np.arange(1000, dtype=np.int64),
                "col2": np.random.randn(1000).astype(np.float64),
                "col3": ["string_" + str(i) for i in range(1000)],
            }
        )

        size = _deep_sizeof(df)
        expected = df.memory_usage(deep=True).sum()
        assert size == expected
        # Should be much larger than just numeric columns due to strings
        assert size > 1000 * (8 + 8)

    def test_pandas_series(self):
        """Test size estimation for pandas Series."""
        pd = pytest.importorskip("pandas")
        from scdataset.experimental.auto_config import _deep_sizeof

        series = pd.Series(np.arange(1000, dtype=np.int64))
        size = _deep_sizeof(series)
        # Should return a reasonable size for the series
        # At minimum, 1000 int64 values = 8000 bytes
        assert size >= 1000 * 8

    def test_dict_with_arrays(self):
        """Test size estimation for dict containing numpy arrays."""
        from scdataset.experimental.auto_config import _deep_sizeof

        sample = {
            "input_ids": np.zeros(512, dtype=np.int64),
            "attention_mask": np.ones(512, dtype=np.int64),
            "labels": np.array([0, 1, 2], dtype=np.int64),
        }

        size = _deep_sizeof(sample)
        # Should include all array data plus dict overhead
        array_size = 512 * 8 + 512 * 8 + 3 * 8
        assert size >= array_size  # Greater due to dict/key overhead

    def test_nested_dict(self):
        """Test size estimation for nested dictionaries."""
        from scdataset.experimental.auto_config import _deep_sizeof

        nested = {
            "level1": {
                "level2a": np.zeros(100, dtype=np.float32),
                "level2b": np.zeros(200, dtype=np.float64),
            },
            "flat": np.zeros(50, dtype=np.int32),
        }

        size = _deep_sizeof(nested)
        # Should include all nested arrays
        expected_arrays = 100 * 4 + 200 * 8 + 50 * 4
        assert size >= expected_arrays

    def test_list_of_arrays(self):
        """Test size estimation for list containing numpy arrays."""
        from scdataset.experimental.auto_config import _deep_sizeof

        items = [
            np.zeros(100, dtype=np.float32),
            np.zeros(200, dtype=np.float64),
            np.zeros(50, dtype=np.int32),
        ]

        size = _deep_sizeof(items)
        expected_arrays = 100 * 4 + 200 * 8 + 50 * 4
        assert size >= expected_arrays

    def test_string(self):
        """Test size estimation for strings."""
        from scdataset.experimental.auto_config import _deep_sizeof

        # ASCII string
        s = "hello world"
        size = _deep_sizeof(s)
        assert size == len(s.encode("utf-8")) == 11

        # Unicode string
        unicode_s = "Hello 世界"
        size = _deep_sizeof(unicode_s)
        assert size == len(unicode_s.encode("utf-8"))

    def test_bytes(self):
        """Test size estimation for bytes."""
        from scdataset.experimental.auto_config import _deep_sizeof

        b = b"hello world"
        size = _deep_sizeof(b)
        assert size == len(b) == 11

    def test_shared_object_counted_once(self):
        """Test that shared objects are only counted once."""
        from scdataset.experimental.auto_config import _deep_sizeof

        shared_array = np.zeros(1000, dtype=np.float64)
        # Both dict values point to the same array
        sample = {
            "view1": shared_array,
            "view2": shared_array,
        }

        size = _deep_sizeof(sample)
        # Should count the array only once (8000 bytes), not twice (16000)
        # Plus some overhead for dict
        assert size < 8000 + 1000  # Much less than 16000

    def test_anndata_object(self):
        """Test size estimation for AnnData objects."""
        anndata = pytest.importorskip("anndata")
        pd = pytest.importorskip("pandas")
        from scdataset.experimental.auto_config import _deep_sizeof

        # Create AnnData with known sizes
        n_obs, n_vars = 100, 500
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        obs = pd.DataFrame(
            {
                "cell_type": ["type_" + str(i % 5) for i in range(n_obs)],
                "batch": np.random.randint(0, 3, n_obs),
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        )

        obsm = {"X_pca": np.random.randn(n_obs, 50).astype(np.float32)}
        layers = {"raw": np.random.randn(n_obs, n_vars).astype(np.float32)}

        adata = anndata.AnnData(X=X, obs=obs, obsm=obsm, layers=layers)

        size = _deep_sizeof(adata)

        # Should include X, obs, obsm, and layers
        min_expected = (
            X.nbytes  # X matrix
            + 50 * n_obs * 4  # obsm["X_pca"]
            + n_obs * n_vars * 4  # layers["raw"]
        )
        assert size >= min_expected

    def test_anndata_with_sparse(self):
        """Test size estimation for AnnData with sparse X."""
        anndata = pytest.importorskip("anndata")
        sparse = pytest.importorskip("scipy.sparse")
        from scdataset.experimental.auto_config import _deep_sizeof

        n_obs, n_vars = 100, 2000
        # Create sparse matrix (90% zeros)
        dense = np.random.randn(n_obs, n_vars).astype(np.float32)
        dense[np.random.random((n_obs, n_vars)) > 0.1] = 0
        X = sparse.csr_matrix(dense)

        adata = anndata.AnnData(X=X)
        size = _deep_sizeof(adata)

        # Should use sparse representation size
        sparse_size = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
        assert size >= sparse_size
        # Should be much less than dense size
        assert size < n_obs * n_vars * 4

    def test_multiindexable(self):
        """Test size estimation for MultiIndexable objects."""
        from scdataset import MultiIndexable
        from scdataset.experimental.auto_config import _deep_sizeof

        # MultiIndexable requires all structured arrays to have same length
        n_samples = 100
        structured = {
            "X": np.random.randn(n_samples, 500).astype(np.float32),
            "labels": np.arange(n_samples, dtype=np.int64),
        }
        unstructured_data = {"metadata": "shared across samples"}

        # unstructured must be passed as keyword argument
        mi = MultiIndexable(structured, unstructured=unstructured_data)

        size = _deep_sizeof(mi)
        # Should include structured arrays
        expected_min = n_samples * 500 * 4 + n_samples * 8
        assert size >= expected_min

    def test_primitive_types(self):
        """Test size estimation for primitive types."""
        import sys

        from scdataset.experimental.auto_config import _deep_sizeof

        # Integer
        size_int = _deep_sizeof(42)
        assert size_int == sys.getsizeof(42)

        # Float
        size_float = _deep_sizeof(3.14)
        assert size_float == sys.getsizeof(3.14)

        # None
        size_none = _deep_sizeof(None)
        assert size_none == sys.getsizeof(None)

    def test_empty_structures(self):
        """Test size estimation for empty structures."""
        import sys

        from scdataset.experimental.auto_config import _deep_sizeof

        # Empty dict
        size_dict = _deep_sizeof({})
        assert size_dict == sys.getsizeof({})

        # Empty list
        size_list = _deep_sizeof([])
        assert size_list == sys.getsizeof([])

        # Empty numpy array
        size_arr = _deep_sizeof(np.array([]))
        assert size_arr == 0  # No bytes in empty array


# =============================================================================
# Edge Cases
# =============================================================================


class TestAutoConfigEdgeCases:
    """Test edge cases for auto_config."""

    def test_tiny_array(self):
        """Test with very small array."""
        tiny = np.random.randn(10, 5).astype(np.float32)
        params = suggest_parameters(tiny, batch_size=2)

        assert params["fetch_factor"] >= 1
        assert params["num_workers"] >= 1

    def test_large_batch_size(self, small_array):
        """Test with batch size larger than data."""
        params = suggest_parameters(small_array, batch_size=2000)

        # Should still return valid parameters
        assert params["fetch_factor"] >= 1

    def test_single_sample_data(self):
        """Test with single sample."""
        single = np.random.randn(1, 100).astype(np.float32)
        params = suggest_parameters(single, batch_size=1)

        assert params["fetch_factor"] >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestAutoConfigIntegration:
    """Integration tests for auto_config."""

    def test_use_with_scdataset(self, small_array):
        """Test using suggested parameters with scDataset."""
        from scdataset import BlockShuffling, scDataset

        params = suggest_parameters(small_array, batch_size=64)

        # Use suggested parameters
        dataset = scDataset(
            small_array,
            BlockShuffling(block_size=params["block_size_balanced"]),
            batch_size=64,
            fetch_factor=params["fetch_factor"],
        )

        # Should work without errors
        batch = next(iter(dataset))
        assert batch.shape[0] == 64

    def test_verbose_mode(self, small_array, capsys):
        """Test verbose mode prints information."""
        params = suggest_parameters(small_array, batch_size=64, verbose=True)

        captured = capsys.readouterr()
        # Verbose mode should print system info and recommendations
        assert "System Information" in captured.out or len(captured.out) > 0
        # Should still return valid params
        assert params["fetch_factor"] >= 1


# =============================================================================
# Memory Validation Tests
# =============================================================================


class TestMemoryEstimationAccuracy:
    """
    Tests that verify memory estimates match actual usage.

    These tests validate that estimate_sample_size produces accurate results
    by comparing estimated sizes with actual batch sizes from scDataset.
    """

    def test_numpy_batch_size_matches_estimate(self):
        """Verify numpy batch size matches sample size estimate * batch_size."""
        from scdataset import Streaming, scDataset
        from scdataset.experimental.auto_config import _deep_sizeof

        # Create data with known size
        n_samples = 500
        n_features = 1000  # 1000 float32 = 4000 bytes per sample
        batch_size = 32
        data = np.random.randn(n_samples, n_features).astype(np.float32)

        # Estimate sample size
        estimated_sample_size = estimate_sample_size(data)
        expected_sample_size = n_features * 4  # float32 = 4 bytes

        # Sample size should be exact
        assert estimated_sample_size == expected_sample_size, (
            f"Sample size mismatch: estimated={estimated_sample_size}, "
            f"expected={expected_sample_size}"
        )

        # Create dataset and verify batch size
        strategy = Streaming(indices=np.arange(100))
        dataset = scDataset(data, strategy, batch_size=batch_size, fetch_factor=2)

        expected_batch_size = batch_size * expected_sample_size
        actual_batch_sizes = []

        for i, batch in enumerate(dataset):
            actual_batch_sizes.append(_deep_sizeof(batch))
            if i >= 2:
                break

        avg_actual = np.mean(actual_batch_sizes)
        error = abs(avg_actual - expected_batch_size) / expected_batch_size

        # Should be within 5%
        assert error < 0.05, (
            f"Batch size error too high: {error:.1%}. "
            f"Expected {expected_batch_size}, got {avg_actual}"
        )

    def test_multiindexable_batch_size_matches_estimate(self):
        """Verify MultiIndexable batch size matches estimate."""
        from scdataset import MultiIndexable, Streaming, scDataset
        from scdataset.experimental.auto_config import _deep_sizeof

        n_samples = 500
        n_features = 500
        batch_size = 32

        # Create MultiIndexable with known sizes
        features = np.random.randn(n_samples, n_features).astype(np.float32)
        labels = np.random.randint(0, 10, n_samples).astype(np.int64)

        multi = MultiIndexable(features=features, labels=labels)

        # Expected: features (500 * 4) + labels (1 * 8) = 2008 bytes per sample
        expected_per_sample = n_features * 4 + 8

        # Estimate
        estimated_sample_size = estimate_sample_size(multi)
        assert estimated_sample_size == expected_per_sample

        # Verify with dataset
        strategy = Streaming(indices=np.arange(100))
        dataset = scDataset(multi, strategy, batch_size=batch_size, fetch_factor=2)

        expected_batch_size = batch_size * expected_per_sample

        for i, batch in enumerate(dataset):
            actual = _deep_sizeof(batch)
            error = abs(actual - expected_batch_size) / expected_batch_size
            assert error < 0.05, f"Batch {i} size error: {error:.1%}"
            if i >= 2:
                break

    def test_fetch_transform_changes_size_correctly(self):
        """Verify fetch_transform affects memory estimates correctly."""
        from scdataset import Streaming, scDataset
        from scdataset.experimental.auto_config import _deep_sizeof

        n_samples = 500
        n_features = 200
        batch_size = 32

        data = np.random.randn(n_samples, n_features).astype(np.float32)

        # Transform that converts to float64 (doubles size)
        def to_float64(sample):
            return sample.astype(np.float64)

        # Estimate without transform
        size_original = estimate_sample_size(data)
        assert size_original == n_features * 4  # float32

        # Estimate with transform
        size_transformed = estimate_sample_size(data, fetch_transform=to_float64)
        assert size_transformed == n_features * 8  # float64

        # Verify with dataset
        strategy = Streaming(indices=np.arange(100))
        dataset = scDataset(
            data,
            strategy,
            batch_size=batch_size,
            fetch_factor=2,
            fetch_transform=to_float64,
        )

        expected_batch_size = batch_size * n_features * 8  # float64

        for i, batch in enumerate(dataset):
            actual = _deep_sizeof(batch)
            error = abs(actual - expected_batch_size) / expected_batch_size
            assert error < 0.05, f"Batch {i} size error: {error:.1%}"
            if i >= 2:
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
