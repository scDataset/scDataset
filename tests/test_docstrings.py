"""
Tests for docstrings in scdataset modules.

This module automatically discovers and runs doctest examples from the
source code docstrings, ensuring that documentation examples remain
accurate and up-to-date.

To run these tests:
    pytest tests/test_docstrings.py -v

To run all doctests directly:
    pytest --doctest-modules src/scdataset/
"""

import doctest

import pytest

# Import modules to test their docstrings
from scdataset import MultiIndexable, scDataset


class TestModuleDocstrings:
    """Test docstrings in scdataset modules using doctest."""

    def test_scdataset_module_docstrings(self):
        """Test docstrings in the main scdataset module."""
        import scdataset.scdataset as scdataset_module

        results = doctest.testmod(
            scdataset_module,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
            verbose=False,
        )

        assert (
            results.failed == 0
        ), f"Doctest failures in scdataset module: {results.failed}"

    def test_strategy_module_docstrings(self):
        """Test docstrings in the strategy module."""
        import scdataset.strategy as strategy_module

        results = doctest.testmod(
            strategy_module,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
            verbose=False,
        )

        assert (
            results.failed == 0
        ), f"Doctest failures in strategy module: {results.failed}"

    def test_multiindexable_module_docstrings(self):
        """Test docstrings in the multiindexable module."""
        import scdataset.multiindexable as multiindexable_module

        results = doctest.testmod(
            multiindexable_module,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
            verbose=False,
        )

        assert (
            results.failed == 0
        ), f"Doctest failures in multiindexable module: {results.failed}"

    def test_auto_config_module_docstrings(self):
        """Test docstrings in the auto_config module."""
        import scdataset.experimental.auto_config as auto_config_module

        results = doctest.testmod(
            auto_config_module,
            optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE,
            verbose=False,
        )

        assert (
            results.failed == 0
        ), f"Doctest failures in auto_config module: {results.failed}"


class TestDocExamples:
    """Test specific code examples from documentation."""

    def test_minimal_example(self):
        """Test the minimal example from quickstart.rst."""
        import numpy as np
        from torch.utils.data import DataLoader

        from scdataset import Streaming

        # Your existing data (numpy array, AnnData, HuggingFace Dataset, etc.)
        data = np.random.randn(1000, 100)  # 1000 samples, 100 features

        # Create scDataset with streaming strategy
        dataset = scDataset(data, Streaming(), batch_size=64)

        # Use with DataLoader (note: batch_size=None)
        loader = DataLoader(dataset, batch_size=None, num_workers=0)

        for batch in loader:
            assert batch.shape == (64, 100), f"Expected (64, 100), got {batch.shape}"
            break

    def test_block_shuffling_example(self):
        """Test block shuffling example from quickstart.rst."""
        import numpy as np

        from scdataset import BlockShuffling

        data = np.random.randn(1000, 100)

        # Shuffle in blocks for better I/O while maintaining some randomness
        strategy = BlockShuffling(block_size=8)
        dataset = scDataset(data, strategy, batch_size=64)

        # Verify it works
        batches = list(dataset)
        assert len(batches) > 0
        assert batches[0].shape[0] == 64

    def test_weighted_sampling_example(self):
        """Test weighted sampling example from quickstart.rst."""
        import numpy as np

        from scdataset import BlockWeightedSampling

        data = np.random.randn(1000, 100)

        # Sample with custom weights (e.g., higher weight for rare samples)
        weights = np.random.rand(len(data))  # Custom weights per sample
        strategy = BlockWeightedSampling(
            weights=weights,
            total_size=10000,  # Generate 10000 samples per epoch
            block_size=8,
        )
        dataset = scDataset(data, strategy, batch_size=64)

        # Verify it works
        batch = next(iter(dataset))
        assert batch.shape[0] == 64

    def test_class_balanced_example(self):
        """Test class balanced sampling example from quickstart.rst."""
        import numpy as np

        from scdataset import ClassBalancedSampling

        data = np.random.randn(1000, 100)

        # Automatically balance classes
        labels = np.random.choice(["A", "B", "C"], size=len(data))
        strategy = ClassBalancedSampling(labels, total_size=10000)
        dataset = scDataset(data, strategy, batch_size=64)

        # Verify it works
        batch = next(iter(dataset))
        assert batch.shape[0] == 64

    def test_multiindexable_basic(self):
        """Test MultiIndexable basic usage example."""
        import numpy as np

        # Create multi-output data
        genes = np.random.randn(100, 2000)
        proteins = np.random.randn(100, 100)

        multi = MultiIndexable(genes=genes, proteins=proteins)

        assert len(multi) == 100
        assert multi.count == 2
        assert multi.names == ["genes", "proteins"]

    def test_multiindexable_with_unstructured(self):
        """Test MultiIndexable with unstructured metadata example."""
        import numpy as np

        X = np.random.randn(100, 2000)
        gene_names = ["Gene_" + str(i) for i in range(2000)]

        multi = MultiIndexable(
            X=X, unstructured={"gene_names": gene_names, "dataset_name": "MyDataset"}
        )

        assert multi.unstructured["gene_names"][:3] == ["Gene_0", "Gene_1", "Gene_2"]

        # Test that unstructured is preserved through subsetting
        subset = multi[10:20]
        assert subset.unstructured["dataset_name"] == "MyDataset"

    def test_data_transforms_example(self):
        """Test data transforms example from quickstart.rst."""
        import numpy as np

        from scdataset import Streaming

        data = np.random.randn(1000, 100)

        def normalize_batch(batch):
            # Apply per-batch normalization
            return (batch - batch.mean()) / batch.std()

        def preprocess_fetch(data):
            # Apply to fetched data before batching
            return data.astype(np.float32)

        dataset = scDataset(
            data,
            Streaming(),
            batch_size=64,
            fetch_transform=preprocess_fetch,
            batch_transform=normalize_batch,
        )

        # Verify transforms are applied
        batch = next(iter(dataset))
        assert batch.dtype == np.float32
        # Normalized batch should have mean close to 0
        assert abs(batch.mean()) < 0.1

    def test_streaming_with_shuffle(self):
        """Test streaming with shuffle example."""
        import numpy as np

        from scdataset import Streaming

        data = np.random.randn(1000, 100)

        # Sequential access with buffer-level shuffling
        strategy = Streaming(shuffle=True)
        dataset = scDataset(data, strategy, batch_size=64)

        # Verify it works
        batch = next(iter(dataset))
        assert batch.shape[0] == 64


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
