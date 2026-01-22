"""
Tests for data loading strategies.

These tests verify that all 6 data loading strategies work correctly
with mock data (since the actual Tahoe dataset may not be available
in all test environments).
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from scdataset import BlockShuffling, BlockWeightedSampling, Streaming
from training_experiments.data.loader import (
    ALL_STRATEGIES,
    STRATEGY_BLOCK_SHUFFLING,
    STRATEGY_BLOCK_WEIGHTED,
    STRATEGY_RANDOM_SAMPLING,
    STRATEGY_STREAMING,
    STRATEGY_STREAMING_BUFFER,
    STRATEGY_TRUE_WEIGHTED,
    TahoeDataLoader,
)


class TestStrategyConstants:
    """Tests for strategy constants."""

    def test_all_strategies_defined(self):
        """Test that all expected strategies are defined."""
        expected = [
            "streaming",
            "streaming_buffer",
            "block_shuffling",
            "random_sampling",
            "block_weighted",
            "true_weighted",
        ]

        assert set(ALL_STRATEGIES) == set(expected)

    def test_strategy_constants(self):
        """Test individual strategy constants."""
        assert STRATEGY_STREAMING == "streaming"
        assert STRATEGY_STREAMING_BUFFER == "streaming_buffer"
        assert STRATEGY_BLOCK_SHUFFLING == "block_shuffling"
        assert STRATEGY_RANDOM_SAMPLING == "random_sampling"
        assert STRATEGY_BLOCK_WEIGHTED == "block_weighted"
        assert STRATEGY_TRUE_WEIGHTED == "true_weighted"


class TestStrategyConfigurations:
    """Tests for strategy configurations."""

    def test_streaming_strategy(self):
        """Test streaming strategy configuration."""
        strategy = Streaming(indices=None, shuffle=False)

        assert not strategy._shuffle_before_yield

    def test_streaming_buffer_strategy(self):
        """Test streaming with buffer strategy configuration."""
        strategy = Streaming(indices=None, shuffle=True)

        assert strategy._shuffle_before_yield

    def test_block_shuffling_strategy(self):
        """Test block shuffling strategy configuration."""
        strategy = BlockShuffling(block_size=4, indices=None, drop_last=False)

        assert strategy.block_size == 4
        assert strategy._shuffle_before_yield

    def test_random_sampling_strategy(self):
        """Test random sampling (block_size=1) strategy configuration."""
        strategy = BlockShuffling(block_size=1, indices=None, drop_last=False)

        assert strategy.block_size == 1

    def test_block_weighted_strategy(self):
        """Test block weighted sampling strategy configuration."""
        n_samples = 100
        weights = np.ones(n_samples)

        strategy = BlockWeightedSampling(
            block_size=4, weights=weights, total_size=n_samples, replace=True
        )

        assert strategy.block_size == 4
        assert strategy.total_size == n_samples
        assert strategy.replace is True

    def test_true_weighted_strategy(self):
        """Test true weighted sampling (block_size=1) strategy configuration."""
        n_samples = 100
        weights = np.ones(n_samples)

        strategy = BlockWeightedSampling(
            block_size=1, weights=weights, total_size=n_samples, replace=True
        )

        assert strategy.block_size == 1


class TestBlockShufflingIndices:
    """Tests for block shuffling index generation."""

    def test_block_shuffling_preserves_count(self):
        """Test that block shuffling preserves sample count."""
        n_samples = 100

        strategy = BlockShuffling(block_size=4, drop_last=False)
        indices = strategy.get_indices(range(n_samples), seed=42)

        assert len(indices) == n_samples
        assert set(indices) == set(range(n_samples))

    def test_block_shuffling_different_seeds(self):
        """Test that different seeds produce different orderings."""
        n_samples = 100

        strategy = BlockShuffling(block_size=4)
        indices1 = strategy.get_indices(range(n_samples), seed=42)
        indices2 = strategy.get_indices(range(n_samples), seed=123)

        # Should be different orderings
        assert not np.array_equal(indices1, indices2)

        # But same elements
        assert set(indices1) == set(indices2)

    def test_random_sampling_block_size_1(self):
        """Test that block_size=1 effectively shuffles all samples."""
        n_samples = 100

        strategy = BlockShuffling(block_size=1)
        indices = strategy.get_indices(range(n_samples), seed=42)

        assert len(indices) == n_samples

        # Should be shuffled (very unlikely to be in order)
        assert not np.array_equal(indices, np.arange(n_samples))


class TestBlockWeightedSamplingIndices:
    """Tests for block weighted sampling index generation."""

    def test_weighted_sampling_respects_weights(self):
        """Test that weighted sampling respects weights."""
        n_samples = 100

        # Give first 10 samples 10x weight
        weights = np.ones(n_samples)
        weights[:10] = 10.0

        strategy = BlockWeightedSampling(
            block_size=4, weights=weights, total_size=n_samples, replace=True
        )

        indices = strategy.get_indices(range(n_samples), seed=42)

        # Count how often high-weight samples appear
        high_weight_count = np.sum(indices < 10)

        # With 10x weight, expect roughly 50% of samples from first 10
        # (10 * 10) / (10 * 10 + 90 * 1) = 100/190 ≈ 52%
        expected_ratio = (10 * 10) / (10 * 10 + 90 * 1)
        actual_ratio = high_weight_count / n_samples

        # Allow some variance
        assert abs(actual_ratio - expected_ratio) < 0.2

    def test_weighted_sampling_total_size(self):
        """Test that weighted sampling returns correct total size."""
        n_samples = 100
        total_size = 200

        weights = np.ones(n_samples)

        strategy = BlockWeightedSampling(
            block_size=4, weights=weights, total_size=total_size, replace=True
        )

        indices = strategy.get_indices(range(n_samples), seed=42)

        assert len(indices) == total_size


class TestDataLoaderInvalidStrategy:
    """Tests for invalid strategy handling."""

    @patch("training_experiments.data.loader.LabelEncoder")
    def test_invalid_strategy_raises(self, mock_label_encoder):
        """Test that invalid strategy raises error."""
        # Mock label encoder
        mock_encoder_instance = MagicMock()
        mock_encoder_instance.get_task_dims.return_value = {
            "cell_line": 10,
            "drug": 20,
            "moa_broad": 5,
            "moa_fine": 15,
        }
        mock_label_encoder.return_value = mock_encoder_instance

        loader = TahoeDataLoader(
            data_dir="/nonexistent/path", label_dir="/nonexistent/path"
        )

        # Mock the collections to avoid file loading
        loader.train_collection = Mock()
        loader.train_collection.n_obs = 100
        loader.test_collection = Mock()
        loader._feature_dim = 100

        with pytest.raises(ValueError, match="Unknown strategy"):
            loader.create_dataloaders(strategy_name="invalid_strategy", batch_size=32)


class TestWeightedSamplingWithWeights:
    """Integration tests for weighted sampling with computed weights."""

    def test_weights_affect_sampling(self):
        """Test that computed weights affect sampling distribution."""
        np.random.seed(42)

        n_samples = 1000

        # Create imbalanced weights
        weights = np.ones(n_samples)
        weights[:100] = 10.0  # First 100 samples get 10x weight

        strategy = BlockWeightedSampling(
            block_size=4, weights=weights, total_size=n_samples, replace=True
        )

        # Sample multiple times to verify distribution
        all_counts = []
        for seed in range(5):
            indices = strategy.get_indices(range(n_samples), seed=seed)
            high_weight_count = np.sum(indices < 100)
            all_counts.append(high_weight_count)

        avg_count = np.mean(all_counts)

        # With 10x weight for first 100, expect them to be ~50% of samples
        # 100 * 10 / (100 * 10 + 900 * 1) = 1000/1900 ≈ 52.6%
        expected = 1000 * (1000 / 1900)

        # Should be reasonably close
        assert abs(avg_count - expected) < 100  # Allow some variance


class TestLabelEncoder:
    """Tests for LabelEncoder with bundled mappings."""

    def test_label_encoder_loads_default_mappings(self):
        """Test that LabelEncoder loads bundled mapping files by default."""
        from training_experiments.data.label_encoder import LabelEncoder

        encoder = LabelEncoder()

        # Verify mappings are loaded
        assert len(encoder.cell_line_to_id) > 0
        assert len(encoder.drug_to_id) > 0
        assert len(encoder.drug_to_moa_broad) > 0
        assert len(encoder.drug_to_moa_fine) > 0

        # Verify task dimensions are computed
        assert encoder.num_cell_lines > 0
        assert encoder.num_drugs > 0
        assert encoder.num_moa_broad > 0
        assert encoder.num_moa_fine > 0

    def test_label_encoder_encode_valid_labels(self):
        """Test that LabelEncoder encodes valid labels correctly."""
        from training_experiments.data.label_encoder import LabelEncoder

        encoder = LabelEncoder()

        # Get a valid cell line and drug from the mappings
        cell_line = list(encoder.cell_line_to_id.keys())[0]
        drug = list(encoder.drug_to_id.keys())[0]

        # Encode
        encoded = encoder.encode_labels([cell_line], [drug])

        assert len(encoded) == 1
        cell_id, drug_id, moa_broad_id, moa_fine_id = encoded[0]

        # All IDs should be valid integers
        assert isinstance(cell_id, (int, np.integer))
        assert isinstance(drug_id, (int, np.integer))
        assert isinstance(moa_broad_id, (int, np.integer))
        assert isinstance(moa_fine_id, (int, np.integer))

    def test_label_encoder_task_dimensions(self):
        """Test that task dimensions are correctly computed."""
        from training_experiments.data.label_encoder import LabelEncoder

        encoder = LabelEncoder()

        task_dims = encoder.get_task_dims()

        assert "cell_line" in task_dims
        assert "drug" in task_dims
        assert "moa_broad" in task_dims
        assert "moa_fine" in task_dims

        assert task_dims["cell_line"] == encoder.num_cell_lines
        assert task_dims["drug"] == encoder.num_drugs
        assert task_dims["moa_broad"] == encoder.num_moa_broad
        assert task_dims["moa_fine"] == encoder.num_moa_fine
