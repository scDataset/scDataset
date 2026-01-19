"""
Tests for weight computation utilities.

These tests verify that the weight computation matches sklearn's balanced
class weights with our custom min_count_baseline modification.
"""

import numpy as np
import pytest
from sklearn.utils.class_weight import (
    compute_class_weight as sklearn_compute_class_weight,
)

from training_experiments.utils.weights import (
    compute_balanced_weights,
    compute_class_weights,
)


class TestComputeClassWeights:
    """Tests for single-label class weight computation."""

    def test_uniform_labels(self):
        """Test weights with uniform label distribution."""
        labels = np.array(["A", "A", "B", "B", "C", "C"])
        weights = compute_class_weights(labels, min_count_baseline=0)

        # With uniform distribution, all weights should be equal
        assert np.allclose(weights, 1.0)
        assert weights.sum() == pytest.approx(len(labels))

    def test_imbalanced_labels(self):
        """Test weights with imbalanced label distribution."""
        # 6 A's, 2 B's, 2 C's
        labels = np.array(["A"] * 6 + ["B"] * 2 + ["C"] * 2)
        weights = compute_class_weights(labels, min_count_baseline=0)

        # B and C should have higher weights than A
        a_weight = weights[0]
        b_weight = weights[6]
        c_weight = weights[8]

        assert b_weight > a_weight
        assert c_weight > a_weight
        assert b_weight == pytest.approx(c_weight)
        assert weights.sum() == pytest.approx(len(labels))

    def test_matches_sklearn(self):
        """Test that weights match sklearn's balanced class weights."""
        labels = np.array(["A"] * 10 + ["B"] * 5 + ["C"] * 3)

        # Compute sklearn weights
        unique_labels = np.unique(labels)
        sklearn_class_weights = sklearn_compute_class_weight(
            "balanced", classes=unique_labels, y=labels
        )
        sklearn_weight_map = dict(zip(unique_labels, sklearn_class_weights))

        # Compute our weights (normalized)
        our_weights = compute_class_weights(labels, min_count_baseline=0)

        # The relative ratios should match
        a_ratio = our_weights[0] / our_weights[10]
        b_ratio = sklearn_weight_map["A"] / sklearn_weight_map["B"]

        assert a_ratio == pytest.approx(b_ratio, rel=1e-5)

    def test_min_count_baseline_reduces_imbalance(self):
        """Test that min_count_baseline reduces weight imbalance."""
        # Very imbalanced: 100 A's, 10 B's
        labels = np.array(["A"] * 100 + ["B"] * 10)

        # Without baseline
        weights_no_baseline = compute_class_weights(labels, min_count_baseline=0)

        # With baseline
        weights_with_baseline = compute_class_weights(labels, min_count_baseline=100)

        # Calculate ratios
        ratio_no_baseline = weights_no_baseline[100] / weights_no_baseline[0]  # B/A
        ratio_with_baseline = (
            weights_with_baseline[100] / weights_with_baseline[0]
        )  # B/A

        # With baseline, the ratio should be smaller (less extreme)
        assert ratio_with_baseline < ratio_no_baseline

        # Both should still have B weighted higher than A
        assert ratio_no_baseline > 1.0
        assert ratio_with_baseline > 1.0

    def test_empty_input(self):
        """Test with empty input."""
        labels = np.array([])
        weights = compute_class_weights(labels)

        assert len(weights) == 0

    def test_single_class(self):
        """Test with single class."""
        labels = np.array(["A", "A", "A"])
        weights = compute_class_weights(labels, min_count_baseline=0)

        assert np.allclose(weights, 1.0)


class TestComputeBalancedWeights:
    """Tests for (cell_line, drug) combination weight computation."""

    def test_uniform_combinations(self):
        """Test weights with uniform combination distribution."""
        cell_lines = np.array(["A", "A", "B", "B"])
        drugs = np.array(["X", "Y", "X", "Y"])

        weights = compute_balanced_weights(cell_lines, drugs, min_count_baseline=0)

        # With uniform distribution, all weights should be equal
        assert np.allclose(weights, 1.0)
        assert weights.sum() == pytest.approx(len(cell_lines))

    def test_imbalanced_combinations(self):
        """Test weights with imbalanced combination distribution."""
        # (A, X): 6 samples, (B, Y): 2 samples
        cell_lines = np.array(["A"] * 6 + ["B"] * 2)
        drugs = np.array(["X"] * 6 + ["Y"] * 2)

        weights = compute_balanced_weights(cell_lines, drugs, min_count_baseline=0)

        # (B, Y) should have higher weight
        ax_weight = weights[0]
        by_weight = weights[6]

        assert by_weight > ax_weight
        assert weights.sum() == pytest.approx(len(cell_lines))

    def test_min_count_baseline_effect(self):
        """Test that min_count_baseline prevents extreme reweighting."""
        # Very imbalanced: (A, X): 1000 samples, (B, Y): 10 samples
        cell_lines = np.array(["A"] * 1000 + ["B"] * 10)
        drugs = np.array(["X"] * 1000 + ["Y"] * 10)

        # Without baseline - extreme reweighting
        weights_no_baseline = compute_balanced_weights(
            cell_lines, drugs, min_count_baseline=0
        )

        # With baseline - moderate reweighting
        weights_with_baseline = compute_balanced_weights(
            cell_lines, drugs, min_count_baseline=1000
        )

        # Calculate ratios
        ratio_no_baseline = weights_no_baseline[1000] / weights_no_baseline[0]
        ratio_with_baseline = weights_with_baseline[1000] / weights_with_baseline[0]

        # Without baseline: ratio should be 1000/10 = 100
        assert ratio_no_baseline == pytest.approx(100.0, rel=0.01)

        # With baseline: ratio should be (1000+1000)/(10+1000) ≈ 2
        assert ratio_with_baseline < ratio_no_baseline
        assert ratio_with_baseline == pytest.approx(
            (1000 + 1000) / (10 + 1000), rel=0.01
        )

    def test_baseline_1000_example(self):
        """Test the specific example from the requirements."""
        # 20000 cells for combo A, 30 cells for combo B
        cell_lines = np.array(["A"] * 20000 + ["B"] * 30)
        drugs = np.array(["X"] * 20000 + ["Y"] * 30)

        # With baseline of 1000
        weights = compute_balanced_weights(cell_lines, drugs, min_count_baseline=1000)

        # Without baseline
        weights_no_baseline = compute_balanced_weights(
            cell_lines, drugs, min_count_baseline=0
        )

        # Calculate ratios
        ratio_with_baseline = weights[20000] / weights[0]  # B/A
        ratio_no_baseline = weights_no_baseline[20000] / weights_no_baseline[0]

        # Without baseline: ratio would be 20000/30 ≈ 666
        assert ratio_no_baseline == pytest.approx(20000 / 30, rel=0.01)

        # With baseline: ratio is (20000+1000)/(30+1000) ≈ 20.4
        expected_ratio = (20000 + 1000) / (30 + 1000)
        assert ratio_with_baseline == pytest.approx(expected_ratio, rel=0.01)

        # Much less extreme
        assert ratio_with_baseline < ratio_no_baseline / 10

    def test_mismatched_lengths_raises(self):
        """Test that mismatched lengths raise ValueError."""
        cell_lines = np.array(["A", "B", "C"])
        drugs = np.array(["X", "Y"])

        with pytest.raises(ValueError, match="same length"):
            compute_balanced_weights(cell_lines, drugs)

    def test_empty_input(self):
        """Test with empty input."""
        cell_lines = np.array([])
        drugs = np.array([])
        weights = compute_balanced_weights(cell_lines, drugs)

        assert len(weights) == 0

    def test_weights_sum(self):
        """Test that weights are normalized to sum to n_samples."""
        np.random.seed(42)
        n = 1000
        cell_lines = np.random.choice(["A", "B", "C"], size=n)
        drugs = np.random.choice(["X", "Y", "Z"], size=n)

        weights = compute_balanced_weights(cell_lines, drugs, min_count_baseline=100)

        assert weights.sum() == pytest.approx(n)

    def test_all_same_combination(self):
        """Test with all same combination."""
        cell_lines = np.array(["A"] * 100)
        drugs = np.array(["X"] * 100)

        weights = compute_balanced_weights(cell_lines, drugs, min_count_baseline=0)

        # All weights should be 1.0
        assert np.allclose(weights, 1.0)


class TestWeightComputation:
    """Integration tests for weight computation."""

    def test_realistic_distribution(self):
        """Test with realistic distribution similar to Tahoe dataset."""
        np.random.seed(42)

        # Simulate realistic cell/drug distribution
        # Some combinations are very common, others rare
        cell_lines = []
        drugs = []

        # Common combinations (simulate ~80% of data)
        for _ in range(8000):
            cell_lines.append(np.random.choice(["CL1", "CL2", "CL3"]))
            drugs.append(np.random.choice(["Drug1", "Drug2"]))

        # Rare combinations (simulate ~20% of data)
        for _ in range(2000):
            cell_lines.append(np.random.choice(["CL_rare1", "CL_rare2"]))
            drugs.append(np.random.choice(["Drug_rare1", "Drug_rare2"]))

        cell_lines = np.array(cell_lines)
        drugs = np.array(drugs)

        # Compute weights
        weights = compute_balanced_weights(cell_lines, drugs, min_count_baseline=1000)

        # Basic sanity checks
        assert len(weights) == 10000
        assert weights.sum() == pytest.approx(10000)
        assert weights.min() > 0
        assert weights.max() < 100  # Shouldn't be too extreme with baseline

        # Rare combinations should have higher weights on average
        common_mask = np.isin(cell_lines, ["CL1", "CL2", "CL3"])
        rare_mask = ~common_mask

        assert weights[rare_mask].mean() > weights[common_mask].mean()
