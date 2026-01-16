"""
Pytest configuration and shared fixtures for scdataset tests.
"""

import numpy as np
import pytest

# =============================================================================
# Numpy Random Seed
# =============================================================================


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield


# =============================================================================
# Common Data Fixtures
# =============================================================================


@pytest.fixture
def sample_array_small():
    """Small sample array for quick tests."""
    return np.random.randn(100, 20).astype(np.float32)


@pytest.fixture
def sample_array_medium():
    """Medium sample array."""
    return np.random.randn(1000, 50).astype(np.float32)


@pytest.fixture
def sample_array_large():
    """Large sample array."""
    return np.random.randn(10000, 100).astype(np.float32)


@pytest.fixture
def binary_labels():
    """Binary labels for 1000 samples."""
    return np.random.randint(0, 2, 1000)


@pytest.fixture
def multiclass_labels():
    """Multiclass labels for 1000 samples."""
    return np.random.choice(["A", "B", "C", "D"], 1000)


@pytest.fixture
def imbalanced_labels():
    """Imbalanced multiclass labels."""
    # 70% class A, 20% class B, 10% class C
    return np.array(["A"] * 700 + ["B"] * 200 + ["C"] * 100)


# =============================================================================
# MultiIndexable Fixtures
# =============================================================================


@pytest.fixture
def multimodal_data():
    """Multi-modal data dictionary."""
    n = 500
    return {
        "genes": np.random.randn(n, 2000).astype(np.float32),
        "proteins": np.random.randn(n, 100).astype(np.float32),
        "metadata": np.random.randn(n, 10).astype(np.float32),
    }


# =============================================================================
# Strategy Fixtures
# =============================================================================


@pytest.fixture
def streaming_strategy():
    """Default Streaming strategy."""
    from scdataset import Streaming

    return Streaming()


@pytest.fixture
def block_shuffling_strategy():
    """Default BlockShuffling strategy."""
    from scdataset import BlockShuffling

    return BlockShuffling(block_size=8)


# =============================================================================
# Utility Functions
# =============================================================================


def assert_batches_cover_data(batches, n_samples, drop_last=False):
    """Assert that batches cover all samples exactly once."""
    total_samples = sum(
        len(b) if hasattr(b, "__len__") else b.shape[0] for b in batches
    )
    if drop_last:
        # May have fewer samples
        assert total_samples <= n_samples
    else:
        assert total_samples == n_samples


def assert_all_indices_unique(indices):
    """Assert all indices are unique."""
    assert len(indices) == len(set(indices))
