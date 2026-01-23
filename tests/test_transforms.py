"""
Tests for transform functions.
"""

import numpy as np
import pytest
import torch

from scdataset.multiindexable import MultiIndexable
from scdataset.transforms import (
    adata_to_mindex,
    hf_tahoe_to_tensor,
)

# =============================================================================
# Mock objects for AnnData tests
# =============================================================================


class MockObs:
    """Mock pandas DataFrame-like obs."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        class Values:
            def __init__(self, v):
                self.values = v

        return Values(self._data[key])


class MockAnnData:
    """Mock AnnData object for testing transforms."""

    def __init__(self, X, obs_data=None):
        self.X = X
        self.obs = MockObs(obs_data or {})

    def to_memory(self):
        return self


# =============================================================================
# Tests for adata_to_mindex
# =============================================================================


class TestAdataToMindex:
    """Tests for adata_to_mindex function."""

    def test_basic_dense_array(self):
        """Test with dense numpy array."""
        X = np.random.randn(10, 100)
        batch = MockAnnData(X)

        result = adata_to_mindex(batch)

        assert isinstance(result, MultiIndexable)
        assert "X" in result.names
        np.testing.assert_array_equal(result["X"], X)

    def test_with_sparse_matrix(self):
        """Test with scipy sparse matrix."""
        pytest.importorskip("scipy")
        import scipy.sparse as sp

        X_dense = np.random.randn(10, 100)
        X_sparse = sp.csr_matrix(X_dense)
        batch = MockAnnData(X_sparse)

        result = adata_to_mindex(batch)

        assert isinstance(result, MultiIndexable)
        np.testing.assert_array_almost_equal(result["X"], X_dense)

    def test_with_columns(self):
        """Test extracting obs columns."""
        X = np.random.randn(10, 100)
        obs_data = {
            "cell_type": np.array(["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"]),
            "batch": np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 1]),
        }
        batch = MockAnnData(X, obs_data)

        result = adata_to_mindex(batch, columns=["cell_type", "batch"])

        assert "X" in result.names
        assert "cell_type" in result.names
        assert "batch" in result.names
        np.testing.assert_array_equal(result["cell_type"], obs_data["cell_type"])
        np.testing.assert_array_equal(result["batch"], obs_data["batch"])

    def test_single_column(self):
        """Test extracting single obs column."""
        X = np.random.randn(5, 50)
        obs_data = {"label": np.array([0, 1, 2, 0, 1])}
        batch = MockAnnData(X, obs_data)

        result = adata_to_mindex(batch, columns=["label"])

        assert "label" in result.names
        np.testing.assert_array_equal(result["label"], obs_data["label"])

    def test_indexable_result(self):
        """Test that result is properly indexable."""
        X = np.random.randn(10, 100)
        obs_data = {"label": np.arange(10)}
        batch = MockAnnData(X, obs_data)

        result = adata_to_mindex(batch, columns=["label"])

        # Test subset indexing
        indices = [0, 2, 4]
        subset = result[indices]

        np.testing.assert_array_equal(subset["X"], X[indices])
        np.testing.assert_array_equal(subset["label"], obs_data["label"][indices])


# =============================================================================
# Tests for hf_tahoe_to_tensor
# =============================================================================


class TestHfTahoeToTensor:
    """Tests for hf_tahoe_to_tensor function."""

    def test_dict_format(self):
        """Test with dict input format."""
        batch = {
            "genes": [np.array([0, 5, 10]), np.array([1, 3])],
            "expressions": [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])],
        }

        result = hf_tahoe_to_tensor(batch, num_genes=20)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 20)
        assert result[0, 0] == 1.0
        assert result[0, 5] == 2.0
        assert result[0, 10] == 3.0
        assert result[1, 1] == 4.0
        assert result[1, 3] == 5.0

    def test_list_format(self):
        """Test with list of dicts input format."""
        batch = [
            {"genes": np.array([0, 2]), "expressions": np.array([1.0, 2.0])},
            {"genes": np.array([1, 4, 5]), "expressions": np.array([3.0, 4.0, 5.0])},
        ]

        result = hf_tahoe_to_tensor(batch, num_genes=10)

        assert result.shape == (2, 10)
        assert result[0, 0] == 1.0
        assert result[0, 2] == 2.0
        assert result[1, 1] == 3.0
        assert result[1, 4] == 4.0
        assert result[1, 5] == 5.0

    def test_empty_genes(self):
        """Test with sample having no expressed genes."""
        batch = {
            "genes": [np.array([], dtype=np.int64), np.array([1], dtype=np.int64)],
            "expressions": [
                np.array([], dtype=np.float32),
                np.array([2.0], dtype=np.float32),
            ],
        }

        result = hf_tahoe_to_tensor(batch, num_genes=10)

        assert result.shape == (2, 10)
        assert torch.sum(result[0]) == 0  # First row should be all zeros
        assert result[1, 1] == 2.0

    def test_custom_num_genes(self):
        """Test with custom num_genes parameter."""
        batch = {"genes": [np.array([0, 99])], "expressions": [np.array([1.0, 2.0])]}

        result = hf_tahoe_to_tensor(batch, num_genes=100)

        assert result.shape == (1, 100)
        assert result[0, 0] == 1.0
        assert result[0, 99] == 2.0

    def test_output_dtype(self):
        """Test output tensor dtype is float."""
        batch = {
            "genes": [np.array([0])],
            "expressions": [np.array([1], dtype=np.int64)],  # Int input
        }

        result = hf_tahoe_to_tensor(batch, num_genes=5)

        assert result.dtype == torch.float32

    def test_invalid_input_type(self):
        """Test that invalid input raises error."""
        with pytest.raises(ValueError, match="must be a dictionary or a list"):
            hf_tahoe_to_tensor("invalid", num_genes=10)

    def test_single_sample(self):
        """Test with single sample batch."""
        batch = {
            "genes": [np.array([0, 1, 2, 3, 4])],
            "expressions": [np.array([1.0, 2.0, 3.0, 4.0, 5.0])],
        }

        result = hf_tahoe_to_tensor(batch, num_genes=10)

        assert result.shape == (1, 10)
        np.testing.assert_array_almost_equal(
            result[0, :5].numpy(), [1.0, 2.0, 3.0, 4.0, 5.0]
        )

    def test_large_batch(self):
        """Test with larger batch size."""
        batch_size = 100
        num_genes = 1000

        batch = {
            "genes": [
                np.random.choice(num_genes, size=50, replace=False)
                for _ in range(batch_size)
            ],
            "expressions": [np.random.randn(50) for _ in range(batch_size)],
        }

        result = hf_tahoe_to_tensor(batch, num_genes=num_genes)

        assert result.shape == (batch_size, num_genes)
        # Check that non-zero values are in correct positions
        for i in range(min(5, batch_size)):  # Check first few samples
            for j, gene_idx in enumerate(batch["genes"][i]):
                assert result[i, gene_idx] == pytest.approx(
                    batch["expressions"][i][j], rel=1e-5
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
