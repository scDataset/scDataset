"""
Tests for benchmark utilities.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# Test fixtures
# =============================================================================


class MockAnnDataBatch:
    """Mock AnnData batch with obs attribute (like AnnLoader produces)."""

    def __init__(self, X, plates):
        self.X = X
        self.obs = {"plate": MagicMock(values=plates)}


class MockMultiIndexableBatch:
    """Mock MultiIndexable batch (like scDataset with adata_to_mindex produces)."""

    def __init__(self, X, plates):
        self._data = {"X": X, "plate": plates}

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    @property
    def shape(self):
        return self._data["X"].shape


class MockLoader:
    """Mock data loader that yields batches for a specified duration."""

    def __init__(self, batch_type, num_batches=100, batch_size=64, num_plates=14):
        self.batch_type = batch_type
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.num_plates = num_plates
        self._count = 0

    def __iter__(self):
        self._count = 0
        return self

    def __next__(self):
        if self._count >= self.num_batches:
            raise StopIteration

        self._count += 1

        # Generate random plate assignments
        plates = np.random.randint(0, self.num_plates, size=self.batch_size)
        X = np.random.randn(self.batch_size, 100)

        if self.batch_type == "anndata":
            return MockAnnDataBatch(X, plates)
        elif self.batch_type == "multiindexable":
            return MockMultiIndexableBatch(X, plates)
        else:
            return X  # Plain tensor/array


# =============================================================================
# Tests for evaluate_loader batch entropy
# =============================================================================


class TestEvaluateLoaderBatchEntropy:
    """
    Ensure evaluate_loader correctly computes batch entropy
    for both AnnData batches AND MultiIndexable batches.
    """

    def test_batch_entropy_anndata_format(self):
        """Test batch entropy calculation with AnnData format batches."""
        from benchmarks.utils import evaluate_loader

        # Create mock loader with AnnData-style batches
        loader = MockLoader("anndata", num_batches=50, batch_size=64, num_plates=14)

        # Short test duration, skip warm-up for fast testing
        result = evaluate_loader(loader, test_time_seconds=1, description="Test", warm_up_seconds=0)

        # Entropy should be > 0 when plates are mixed
        assert result["avg_batch_entropy"] > 0, (
            "Batch entropy should be > 0 for AnnData batches with mixed plates. "
            f"Got: {result['avg_batch_entropy']}"
        )
        assert result["std_batch_entropy"] >= 0

    def test_batch_entropy_multiindexable_format(self):
        """
        Ensure batch entropy works with MultiIndexable format.
        """
        from benchmarks.utils import evaluate_loader

        # Create mock loader with MultiIndexable-style batches (scDataset format)
        loader = MockLoader("multiindexable", num_batches=50, batch_size=64, num_plates=14)

        result = evaluate_loader(loader, test_time_seconds=1, description="Test", warm_up_seconds=0)

        # This test would have caught the bug!
        assert result["avg_batch_entropy"] > 0, (
            "Batch entropy should be > 0 for MultiIndexable batches "
            "(scDataset with adata_to_mindex). If this fails, check that "
            "evaluate_loader handles batch['plate'] correctly. "
            f"Got: {result['avg_batch_entropy']}"
        )
        assert result["std_batch_entropy"] >= 0

    def test_batch_entropy_no_plates(self):
        """Test that batch entropy is 0 when no plate info is available."""
        from benchmarks.utils import evaluate_loader

        # Create mock loader with plain tensor batches (no plate info)
        loader = MockLoader("tensor", num_batches=50, batch_size=64)

        result = evaluate_loader(loader, test_time_seconds=1, description="Test", warm_up_seconds=0)

        # Without plate info, entropy should be 0
        assert result["avg_batch_entropy"] == 0
        assert result["std_batch_entropy"] == 0

    def test_batch_entropy_values_reasonable(self):
        """Test that entropy values are in reasonable range."""
        from benchmarks.utils import evaluate_loader

        # With 14 plates, max entropy is log2(14) ≈ 3.81 bits
        loader = MockLoader("multiindexable", num_batches=100, batch_size=128, num_plates=14)

        result = evaluate_loader(loader, test_time_seconds=2, description="Test", warm_up_seconds=0)

        # Entropy should be between 0 and log2(num_plates)
        max_entropy = np.log2(14)
        assert 0 < result["avg_batch_entropy"] <= max_entropy, (
            f"Entropy should be in range (0, {max_entropy}], got {result['avg_batch_entropy']}"
        )


# =============================================================================
# Tests for plot utilities
# =============================================================================


class TestPlotUtils:
    """Tests for plot_utils module."""

    @pytest.fixture
    def sample_df_anndata(self):
        """Create sample DataFrame mimicking AnnData benchmark results."""
        data = {
            "mode": ["random", "random", "random", "random", "random"],
            "loader": ["AnnLoader", "scDataset", "scDataset", "scDataset", "scDataset"],
            "collection_type": ["anncollection"] * 5,
            "batch_size": [64] * 5,
            "block_size": [np.nan, 1, 4, 16, 64],
            "fetch_factor": [np.nan, 1, 1, 1, 1],
            "num_workers": [0] * 5,
            "samples_tested": [2432, 2368, 3328, 8704, 26048],
            "elapsed": [120.0] * 5,
            "samples_per_second": [20.0, 20.0, 28.0, 73.0, 217.0],
            "avg_batch_entropy": [3.6, 0.5, 1.2, 2.1, 2.8],
            "std_batch_entropy": [0.1, 0.1, 0.1, 0.1, 0.1],
            "strategy": [None, "BlockShuffling", "BlockShuffling", "BlockShuffling", "BlockShuffling"],
        }
        return pd.DataFrame(data)

    def test_import_plot_utils(self):
        """Test that plot_utils can be imported."""
        from benchmarks.plot_utils import (
            plot_batch_entropy,
            plot_block_size_by_fetch_factor,
            plot_throughput,
        )

        assert callable(plot_throughput)
        assert callable(plot_batch_entropy)
        assert callable(plot_block_size_by_fetch_factor)

    def test_plot_throughput_creates_figure(self, sample_df_anndata):
        """Test that plot_throughput creates a figure."""
        from benchmarks.plot_utils import plot_throughput

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_df_anndata.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with patch("matplotlib.pyplot.show"):
                fig = plot_throughput(temp_path, title="Test", show_plot=False)
            assert fig is not None
        finally:
            os.unlink(temp_path)

    def test_plot_batch_entropy_creates_figure(self, sample_df_anndata):
        """Test that plot_batch_entropy creates a figure."""
        from benchmarks.plot_utils import plot_batch_entropy

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            sample_df_anndata.to_csv(f.name, index=False)
            temp_path = f.name

        try:
            with patch("matplotlib.pyplot.show"):
                fig = plot_batch_entropy(temp_path, title="Test", show_plot=False)
            assert fig is not None
        finally:
            os.unlink(temp_path)


# =============================================================================
# Tests for benchmark utility functions
# =============================================================================


class TestBenchmarkUtils:
    """Tests for benchmark utils module."""

    def test_import_utils(self):
        """Test that utils can be imported."""
        from benchmarks.utils import (
            evaluate_loader,
            adata_to_mindex,
            hf_tahoe_to_tensor,
            load_config,
            save_results_to_csv,
        )

        assert callable(evaluate_loader)
        assert callable(adata_to_mindex)
        assert callable(hf_tahoe_to_tensor)
        assert callable(load_config)
        assert callable(save_results_to_csv)

    def test_hf_tahoe_to_tensor_dict_format(self):
        """Test HuggingFace transform with dict format."""
        from benchmarks.utils import hf_tahoe_to_tensor

        batch = {
            "genes": [np.array([0, 5, 10]), np.array([2, 3])],
            "expressions": [np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.5])],
        }

        result = hf_tahoe_to_tensor(batch, num_genes=20)

        assert result.shape == (2, 20)
        assert result[0, 0] == 1.0
        assert result[0, 5] == 2.0

    def test_adata_to_mindex_with_columns(self):
        """Test AnnData transform includes plate column correctly."""
        from benchmarks.utils import adata_to_mindex

        # Mock AnnData batch
        class MockAnnData:
            def __init__(self):
                self.X = np.random.randn(10, 100)
                self.obs = pd.DataFrame({"plate": ["plate1"] * 5 + ["plate2"] * 5})

            def to_memory(self):
                return self

        batch = MockAnnData()
        result = adata_to_mindex(batch, columns=["plate"])

        # Result should be a MultiIndexable with X and plate accessible via indexing
        X = result["X"]
        plates = result["plate"]
        assert X.shape == (10, 100)
        assert len(plates) == 10

    def test_load_config_default(self):
        """Test that load_config returns defaults for missing file."""
        from benchmarks.utils import load_config

        config = load_config("nonexistent_config.yaml")

        assert "batch_sizes" in config
        assert "block_sizes" in config

    def test_save_results_to_csv(self):
        """Test saving results to CSV."""
        from benchmarks.utils import save_results_to_csv

        results = [
            {"loader": "test", "samples_per_second": 100.0, "avg_batch_entropy": 2.5},
            {"loader": "test2", "samples_per_second": 200.0, "avg_batch_entropy": 3.0},
        ]

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            df = save_results_to_csv(results, temp_path)
            assert os.path.exists(temp_path)
            assert len(df) == 2
            assert "avg_batch_entropy" in df.columns
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
