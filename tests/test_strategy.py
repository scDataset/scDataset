"""
Tests for sampling strategies.

This module tests all sampling strategies in scdataset.strategy:
- SamplingStrategy (base class)
- Streaming
- BlockShuffling
- BlockWeightedSampling
- ClassBalancedSampling
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from scdataset import (
    BlockShuffling,
    BlockWeightedSampling,
    ClassBalancedSampling,
    Streaming,
)
from scdataset.strategy import SamplingStrategy

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Create sample data array."""
    return np.random.randn(1000, 50)


@pytest.fixture
def sample_labels():
    """Create sample class labels."""
    return np.random.choice(["A", "B", "C"], size=1000)


@pytest.fixture
def sample_weights():
    """Create sample weights."""
    weights = np.random.rand(1000)
    return weights / weights.sum()


@pytest.fixture
def small_data():
    """Create small sample data for detailed tests."""
    return np.arange(100).reshape(-1, 1)


# =============================================================================
# SamplingStrategy Base Class Tests
# =============================================================================


class TestSamplingStrategy:
    """Tests for the SamplingStrategy base class."""

    def test_abstract_get_indices(self):
        """Test that base class raises NotImplementedError."""
        strategy = SamplingStrategy()
        with pytest.raises(NotImplementedError):
            strategy.get_indices([1, 2, 3])

    def test_get_rng_with_seed(self):
        """Test _get_rng with seed parameter."""
        strategy = SamplingStrategy()
        rng = strategy._get_rng(seed=42)
        assert isinstance(rng, np.random.Generator)

        # Same seed should produce same sequence
        rng1 = strategy._get_rng(seed=42)
        rng2 = strategy._get_rng(seed=42)
        assert rng1.random() == rng2.random()

    def test_get_rng_with_rng_parameter(self):
        """Test _get_rng with existing rng."""
        strategy = SamplingStrategy()
        existing_rng = np.random.default_rng(123)
        returned_rng = strategy._get_rng(rng=existing_rng)
        assert returned_rng is existing_rng

    def test_get_rng_rng_overrides_seed(self):
        """Test that rng parameter takes precedence over seed."""
        strategy = SamplingStrategy()
        existing_rng = np.random.default_rng(123)
        returned_rng = strategy._get_rng(seed=999, rng=existing_rng)
        assert returned_rng is existing_rng


# =============================================================================
# Streaming Strategy Tests
# =============================================================================


class TestStreaming:
    """Tests for the Streaming sampling strategy."""

    def test_basic_streaming(self, sample_data):
        """Test basic streaming through all indices."""
        strategy = Streaming()
        indices = strategy.get_indices(sample_data)
        assert_array_equal(indices, np.arange(len(sample_data)))

    def test_streaming_with_indices(self):
        """Test streaming with custom indices."""
        custom_indices = np.array([10, 20, 30, 40])
        strategy = Streaming(indices=custom_indices)
        indices = strategy.get_indices(range(100))
        assert_array_equal(indices, custom_indices)

    def test_streaming_sorts_unsorted_indices(self):
        """Test that unsorted indices are sorted automatically."""
        unsorted = np.array([30, 10, 40, 20])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            strategy = Streaming(indices=unsorted)
            # Check warning was raised
            assert len(w) == 1
            assert "sorted" in str(w[0].message).lower()

        indices = strategy.get_indices(range(100))
        assert_array_equal(indices, np.array([10, 20, 30, 40]))

    def test_streaming_no_warning_for_sorted_indices(self):
        """Test that no warning is raised for sorted indices."""
        sorted_indices = np.array([10, 20, 30, 40])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Streaming(indices=sorted_indices)  # No warning expected
            # No warning should be raised
            sort_warnings = [x for x in w if "sorted" in str(x.message).lower()]
            assert len(sort_warnings) == 0

    def test_get_len(self, sample_data):
        """Test get_len method."""
        # Without indices
        strategy = Streaming()
        assert strategy.get_len(sample_data) == len(sample_data)

        # With indices
        strategy_subset = Streaming(indices=[1, 2, 3])
        assert strategy_subset.get_len(sample_data) == 3

    def test_streaming_shuffle_flag(self):
        """Test shuffle flag affects _shuffle_before_yield."""
        strategy_no_shuffle = Streaming(shuffle=False)
        assert strategy_no_shuffle._shuffle_before_yield is False

        strategy_shuffle = Streaming(shuffle=True)
        assert strategy_shuffle._shuffle_before_yield is True

    def test_streaming_reproducibility(self, sample_data):
        """Test that streaming produces consistent indices."""
        strategy = Streaming()
        indices1 = strategy.get_indices(sample_data)
        indices2 = strategy.get_indices(sample_data)
        assert_array_equal(indices1, indices2)


# =============================================================================
# BlockShuffling Strategy Tests
# =============================================================================


class TestBlockShuffling:
    """Tests for the BlockShuffling sampling strategy."""

    def test_basic_block_shuffling(self, sample_data):
        """Test basic block shuffling produces all indices."""
        strategy = BlockShuffling(block_size=8)
        indices = strategy.get_indices(sample_data, seed=42)

        # All original indices should be present
        assert len(indices) == len(sample_data)
        assert set(indices) == set(range(len(sample_data)))

    def test_block_shuffling_preserves_block_order(self):
        """Test that indices within blocks remain sorted."""
        data = list(range(100))
        strategy = BlockShuffling(block_size=10)
        indices = strategy.get_indices(data, seed=42)

        # Within each block of 10, indices should be sorted
        for i in range(0, len(indices), 10):
            block = indices[i : i + 10]
            assert list(block) == sorted(block)

    def test_block_shuffling_with_indices(self):
        """Test block shuffling with custom indices."""
        custom_indices = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        strategy = BlockShuffling(indices=custom_indices, block_size=5)
        indices = strategy.get_indices(range(100), seed=42)

        # All custom indices should be present
        assert set(indices) == set(custom_indices)

    def test_block_shuffling_sorts_unsorted_indices(self):
        """Test that unsorted indices are sorted automatically."""
        unsorted = np.array([50, 10, 30, 20, 40, 0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BlockShuffling(indices=unsorted, block_size=3)  # Triggers warning
            # Check warning was raised
            assert len(w) == 1
            assert "sorted" in str(w[0].message).lower()

    def test_block_shuffling_reproducibility(self, sample_data):
        """Test reproducibility with same seed."""
        strategy = BlockShuffling(block_size=8)
        indices1 = strategy.get_indices(sample_data, seed=42)
        indices2 = strategy.get_indices(sample_data, seed=42)
        assert_array_equal(indices1, indices2)

    def test_block_shuffling_different_seeds(self, sample_data):
        """Test that different seeds produce different results."""
        strategy = BlockShuffling(block_size=8)
        indices1 = strategy.get_indices(sample_data, seed=42)
        indices2 = strategy.get_indices(sample_data, seed=123)
        # Should contain same indices but in different order
        assert set(indices1) == set(indices2)
        assert not np.array_equal(indices1, indices2)

    def test_block_shuffling_get_len(self, sample_data):
        """Test get_len method."""
        strategy = BlockShuffling(block_size=8)
        assert strategy.get_len(sample_data) == len(sample_data)

        strategy_subset = BlockShuffling(indices=[1, 2, 3, 4], block_size=2)
        assert strategy_subset.get_len(sample_data) == 4

    def test_shuffle_before_yield_flag(self):
        """Test _shuffle_before_yield is properly set."""
        strategy = BlockShuffling(block_size=8)
        assert (
            strategy._shuffle_before_yield is True
        )  # BlockShuffling always has shuffling


# =============================================================================
# BlockWeightedSampling Strategy Tests
# =============================================================================


class TestBlockWeightedSampling:
    """Tests for the BlockWeightedSampling strategy."""

    def test_basic_weighted_sampling(self, sample_data, sample_weights):
        """Test basic weighted sampling."""
        strategy = BlockWeightedSampling(
            weights=sample_weights, block_size=8, total_size=500
        )
        indices = strategy.get_indices(sample_data, seed=42)
        assert len(indices) == 500

    def test_weighted_sampling_with_uniform_weights(self, sample_data):
        """Test weighted sampling with uniform weights."""
        n = len(sample_data)
        uniform_weights = np.ones(n) / n
        strategy = BlockWeightedSampling(
            weights=uniform_weights, block_size=8, total_size=200
        )
        indices = strategy.get_indices(sample_data, seed=42)
        assert len(indices) == 200

    def test_weighted_sampling_respects_weights(self, small_data):
        """Test that weighted sampling respects weight distribution."""
        n = len(small_data)
        # Very skewed weights - first half has much higher probability
        weights = np.zeros(n)
        weights[: n // 2] = 0.99 / (n // 2)
        weights[n // 2 :] = 0.01 / (n // 2)

        strategy = BlockWeightedSampling(weights=weights, block_size=4, total_size=1000)
        indices = strategy.get_indices(small_data, seed=42)

        # Most samples should come from the first half
        first_half_count = np.sum(indices < n // 2)
        assert first_half_count > len(indices) * 0.8

    def test_weighted_sampling_sorts_unsorted_indices(self):
        """Test that unsorted indices are sorted automatically."""
        n = 100
        unsorted = np.array([50, 10, 30, 20, 40, 0])
        weights = np.ones(n) / n

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            BlockWeightedSampling(
                weights=weights, indices=unsorted, block_size=3, total_size=6
            )  # Triggers warning
            # Check warning was raised
            sort_warnings = [x for x in w if "sorted" in str(x.message).lower()]
            assert len(sort_warnings) == 1

    def test_weighted_sampling_reproducibility(self, sample_data, sample_weights):
        """Test reproducibility with same seed."""
        strategy = BlockWeightedSampling(
            weights=sample_weights, block_size=8, total_size=200
        )
        indices1 = strategy.get_indices(sample_data, seed=42)
        indices2 = strategy.get_indices(sample_data, seed=42)
        assert_array_equal(indices1, indices2)

    def test_weighted_sampling_get_len(self, sample_data, sample_weights):
        """Test get_len method."""
        strategy = BlockWeightedSampling(
            weights=sample_weights, block_size=8, total_size=500
        )
        assert strategy.get_len(sample_data) == 500


# =============================================================================
# ClassBalancedSampling Strategy Tests
# =============================================================================


class TestClassBalancedSampling:
    """Tests for the ClassBalancedSampling strategy."""

    def test_basic_class_balanced_sampling(self, sample_labels):
        """Test basic class balanced sampling."""
        strategy = ClassBalancedSampling(
            labels=sample_labels, total_size=600, block_size=8
        )
        indices = strategy.get_indices(range(len(sample_labels)), seed=42)
        assert len(indices) == 600

    def test_class_balance_distribution(self):
        """Test that classes are evenly distributed."""
        n = 1000
        # Imbalanced classes: 70% A, 20% B, 10% C
        labels = np.array(["A"] * 700 + ["B"] * 200 + ["C"] * 100)

        strategy = ClassBalancedSampling(labels=labels, total_size=300, block_size=4)
        indices = strategy.get_indices(range(n), seed=42)

        # Count samples per class in result
        sampled_labels = labels[indices]
        class_counts = {
            "A": np.sum(sampled_labels == "A"),
            "B": np.sum(sampled_labels == "B"),
            "C": np.sum(sampled_labels == "C"),
        }

        # Each class should have roughly equal representation
        expected_per_class = len(indices) / 3
        for count in class_counts.values():
            assert abs(count - expected_per_class) < expected_per_class * 0.5

    def test_class_balanced_reproducibility(self, sample_labels):
        """Test reproducibility with same seed."""
        strategy = ClassBalancedSampling(
            labels=sample_labels, total_size=300, block_size=8
        )
        indices1 = strategy.get_indices(range(len(sample_labels)), seed=42)
        indices2 = strategy.get_indices(range(len(sample_labels)), seed=42)
        assert_array_equal(indices1, indices2)

    def test_class_balanced_different_seeds(self, sample_labels):
        """Test that different seeds produce different results."""
        strategy = ClassBalancedSampling(
            labels=sample_labels, total_size=300, block_size=8
        )
        indices1 = strategy.get_indices(range(len(sample_labels)), seed=42)
        indices2 = strategy.get_indices(range(len(sample_labels)), seed=123)
        # Should be different
        assert not np.array_equal(indices1, indices2)

    def test_class_balanced_get_len(self, sample_labels):
        """Test get_len method."""
        strategy = ClassBalancedSampling(
            labels=sample_labels, total_size=500, block_size=8
        )
        assert strategy.get_len(range(len(sample_labels))) == 500

    def test_class_balanced_with_indices(self, sample_labels):
        """Test class balanced sampling with subset indices."""
        subset_indices = np.arange(0, 500)
        strategy = ClassBalancedSampling(
            labels=sample_labels, indices=subset_indices, total_size=200, block_size=4
        )
        indices = strategy.get_indices(range(len(sample_labels)), seed=42)

        # All returned indices should be within the subset
        assert all(idx in subset_indices for idx in indices)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        strategy = Streaming()
        indices = strategy.get_indices([])
        assert len(indices) == 0

    def test_single_sample(self):
        """Test handling of single sample."""
        strategy = Streaming()
        indices = strategy.get_indices([1])
        assert_array_equal(indices, [0])

    def test_block_size_larger_than_data(self):
        """Test block size larger than dataset."""
        small_data = list(range(5))
        strategy = BlockShuffling(block_size=100)
        indices = strategy.get_indices(small_data, seed=42)
        assert len(indices) == 5
        assert set(indices) == set(range(5))

    def test_block_size_one(self):
        """Test block size of 1 (full shuffle)."""
        data = list(range(100))
        strategy = BlockShuffling(block_size=1)
        indices = strategy.get_indices(data, seed=42)
        assert len(indices) == 100
        assert set(indices) == set(range(100))


# =============================================================================
# Integration Tests
# =============================================================================


class TestStrategyIntegration:
    """Integration tests for strategies."""

    def test_strategy_with_numpy_array(self, sample_data):
        """Test strategies with numpy arrays."""
        for strategy_class in [Streaming, BlockShuffling]:
            if strategy_class == BlockShuffling:
                strategy = strategy_class(block_size=8)
            else:
                strategy = strategy_class()

            indices = strategy.get_indices(sample_data, seed=42)
            assert len(indices) == len(sample_data)

    def test_strategy_with_list(self):
        """Test strategies with Python lists."""
        data = list(range(100))
        strategy = Streaming()
        indices = strategy.get_indices(data)
        assert len(indices) == 100

    def test_strategy_with_range(self):
        """Test strategies with range objects."""
        strategy = Streaming()
        indices = strategy.get_indices(range(100))
        assert_array_equal(indices, np.arange(100))


# =============================================================================
# Weighted Sampling Edge Cases
# =============================================================================


class TestBlockWeightedSamplingEdgeCases:
    """Test edge cases for BlockWeightedSampling."""

    def test_weights_with_full_data_collection(self):
        """Test that full data collection weights work correctly."""
        weights = np.ones(100)
        strategy = BlockWeightedSampling(weights=weights, block_size=4, total_size=50)

        indices = strategy.get_indices(range(100), seed=42)
        assert len(indices) == 50
        assert all(0 <= idx < 100 for idx in indices)

    def test_weights_mismatch_with_data_raises_error(self):
        """Test that mismatched weights and data_collection raises error."""
        weights = np.ones(50)  # 50 weights
        strategy = BlockWeightedSampling(weights=weights, block_size=4, total_size=50)

        # Try with 100-length data_collection - should fail
        with pytest.raises(ValueError, match="weights.*length"):
            strategy.get_indices(range(100), seed=42)

    def test_weighted_sampling_without_replacement(self):
        """Test weighted sampling without replacement."""
        weights = np.ones(100)
        strategy = BlockWeightedSampling(
            weights=weights,
            block_size=4,
            total_size=50,
            replace=False,
            sampling_size=10,
        )

        indices = strategy.get_indices(range(100), seed=42)
        assert len(indices) == 50


class TestClassBalancedSamplingEdgeCases:
    """Test edge cases for ClassBalancedSampling."""

    def test_class_balanced_with_two_classes(self):
        """Test class balancing with binary classes."""
        labels = np.array([0] * 90 + [1] * 10)  # Imbalanced
        strategy = ClassBalancedSampling(labels=labels, block_size=4, total_size=100)

        indices = strategy.get_indices(range(100), seed=42)

        # Check that class 1 is sampled more than its base rate of 10%
        sampled_labels = labels[indices]
        class_1_fraction = np.mean(sampled_labels == 1)
        # Class balancing should make it closer to 50%
        assert class_1_fraction > 0.2  # Much higher than base 10%

    def test_empty_labels_raises_error(self):
        """Test that empty labels raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ClassBalancedSampling(labels=[], block_size=4)

    def test_labels_shorter_than_indices_raises_error(self):
        """Test that labels shorter than indices raises error."""
        labels = np.array([0, 1, 2])  # Only 3 labels
        indices = np.array([0, 1, 2, 3, 4])  # 5 indices - more than labels

        with pytest.raises(ValueError, match="labels length.*must be either equal"):
            ClassBalancedSampling(labels=labels, indices=indices, block_size=4)


# =============================================================================
# Dual Behavior Tests: Full vs Subset Weights/Labels
# =============================================================================


class TestDualWeightBehavior:
    """Test dual behavior: full-length vs subset-length weights."""

    def test_weighted_full_length_weights_preserves_global_importance(self):
        """When weights length = data_collection, use global weights then subset."""
        # 100 samples: indices 0-89 have weight 1.0, indices 90-99 have weight 9.0
        full_weights = np.array([1.0] * 90 + [9.0] * 10)
        subset_indices = np.array([0, 1, 2, 3, 4, 90, 91, 92, 93, 94])

        strategy = BlockWeightedSampling(
            weights=full_weights, indices=subset_indices, total_size=1000, block_size=4
        )
        sampled = strategy.get_indices(range(100), seed=42)

        # High-weight samples (90-99) should be ~9x more likely
        high_weight_count = np.sum(sampled >= 90)
        low_weight_count = np.sum(sampled < 90)
        ratio = high_weight_count / max(low_weight_count, 1)

        # Expect ratio close to 9 (global importance preserved)
        assert 6 < ratio < 12, f"Expected ratio ~9, got {ratio}"

    def test_weighted_subset_length_weights_uses_directly(self):
        """When weights length = indices length, use weights directly."""
        subset_indices = np.array([0, 1, 2, 3, 4, 90, 91, 92, 93, 94])
        # Uniform weights for subset - 50/50 sampling
        subset_weights = np.ones(10)

        strategy = BlockWeightedSampling(
            weights=subset_weights,
            indices=subset_indices,
            total_size=1000,
            block_size=4,
        )
        sampled = strategy.get_indices(range(100), seed=42)

        # Should be approximately 50/50
        high_idx_count = np.sum(sampled >= 90)
        low_idx_count = np.sum(sampled < 90)
        ratio = high_idx_count / max(low_idx_count, 1)

        # Expect ratio close to 1 (uniform)
        assert 0.8 < ratio < 1.2, f"Expected ratio ~1, got {ratio}"

    def test_weighted_mismatched_length_raises_error(self):
        """When weights length matches neither, raise error."""
        subset_indices = np.array([0, 1, 2, 3, 4])  # 5 indices
        wrong_weights = np.ones(7)  # Neither 5 (indices) nor 100 (data)

        strategy = BlockWeightedSampling(
            weights=wrong_weights, indices=subset_indices, block_size=4
        )

        with pytest.raises(ValueError, match="weights length"):
            strategy.get_indices(range(100), seed=42)


class TestDualLabelBehavior:
    """Test dual behavior for ClassBalancedSampling with full vs subset labels."""

    def test_class_balanced_full_labels_preserves_global_importance(self):
        """When labels length = data_collection, compute global class weights.

        Class balancing gives inverse frequency weights so that each CLASS
        has equal probability of being sampled. With 90% class A and 10% class B,
        class B samples get 9x higher weight, resulting in ~50/50 class ratio.

        When sampling from a SUBSET where the local distribution differs from
        global, the global weights are used and then renormalized for the subset.
        """
        # Full dataset: 90% class A (0), 10% class B (1)
        full_labels = np.array([0] * 90 + [1] * 10)
        # Subset: 5 from A, 5 from B (50/50 in subset)
        subset_indices = np.array([0, 1, 2, 3, 4, 90, 91, 92, 93, 94])

        strategy = ClassBalancedSampling(
            labels=full_labels, indices=subset_indices, total_size=1000, block_size=4
        )
        sampled = strategy.get_indices(range(100), seed=42)

        # With global weights, class B samples get 9x weight per sample.
        # In subset: 5 samples with weight W_A, 5 samples with weight 9*W_A
        # After renormalization: class A probability = 5/(5+45) = 10%
        # Class B probability = 45/50 = 90%
        class_b_count = np.sum(full_labels[sampled] == 1)
        class_a_count = np.sum(full_labels[sampled] == 0)
        ratio = class_b_count / max(class_a_count, 1)

        # Class B should be heavily oversampled due to global class imbalance
        assert 6 < ratio < 12, f"Expected ratio ~9, got {ratio}"

    def test_class_balanced_subset_labels_uses_subset_distribution(self):
        """When labels length = indices length, compute weights on subset."""
        full_labels = np.array([0] * 90 + [1] * 10)
        subset_indices = np.array([0, 1, 2, 3, 4, 90, 91, 92, 93, 94])

        # Pass only subset labels (50/50 distribution)
        subset_labels = full_labels[subset_indices]

        strategy = ClassBalancedSampling(
            labels=subset_labels, indices=subset_indices, total_size=1000, block_size=4
        )
        sampled = strategy.get_indices(range(100), seed=42)

        # With 50/50 subset distribution, both classes equally weighted
        class_b_count = np.sum(full_labels[sampled] == 1)
        class_a_count = np.sum(full_labels[sampled] == 0)
        ratio = class_b_count / max(class_a_count, 1)

        # Expect ratio close to 1 (balanced within subset)
        assert 0.8 < ratio < 1.2, f"Expected ratio ~1, got {ratio}"

    def test_class_balanced_no_indices_uses_full_labels(self):
        """Without indices, weights are computed on full labels to balance classes."""
        labels = np.array([0] * 90 + [1] * 10)

        strategy = ClassBalancedSampling(labels=labels, total_size=1000, block_size=4)
        sampled = strategy.get_indices(range(100), seed=42)

        # Class balancing should make both classes equally likely to be sampled
        # Class B samples get 9x weight, so they should appear ~equally with class A
        class_b_count = np.sum(labels[sampled] == 1)
        class_a_count = np.sum(labels[sampled] == 0)
        ratio = class_b_count / max(class_a_count, 1)

        # Expect ratio close to 1 (balanced sampling)
        assert 0.8 < ratio < 1.2, f"Expected ratio ~1 (balanced), got {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
