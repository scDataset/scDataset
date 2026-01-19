"""
Weight computation utilities for balanced sampling.

This module provides utilities for computing balanced sampling weights
similar to sklearn.utils.class_weight.compute_class_weight but with
modifications to handle very low count combinations.
"""

import numpy as np
from numpy.typing import ArrayLike


def compute_balanced_weights(
    cell_lines: ArrayLike, drugs: ArrayLike, min_count_baseline: int = 1000
) -> np.ndarray:
    """
    Compute balanced sampling weights for (cell_line, drug) combinations.

    This function computes weights similar to sklearn's compute_class_weight
    with 'balanced' mode, but with a twist to handle very low count combinations.

    **Problem**: Some (cell_line, drug) combinations have 20,000 cells while
    others have only 30 cells. Simple balanced sampling would oversample the
    30-cell combinations, risking dataset memorization.

    **Solution**: We add a constant baseline (min_count_baseline) to all
    combination counts before computing the balancing weights. This suppresses
    extreme reweighting for rare combinations.

    The formula for each combination is:
        adjusted_count = actual_count + min_count_baseline
        weight = total_cells / (n_combinations * adjusted_count)

    Parameters
    ----------
    cell_lines : array-like
        Array of cell line labels for each sample
    drugs : array-like
        Array of drug labels for each sample
    min_count_baseline : int, default=1000
        Minimum count baseline to add to all combination counts.
        This prevents extreme reweighting for rare combinations.
        Set to 0 for standard balanced weighting.

    Returns
    -------
    numpy.ndarray
        Weight for each sample. Weights are normalized such that they
        sum to the total number of samples.

    Examples
    --------
    >>> cell_lines = np.array(['A', 'A', 'B', 'B', 'B'])
    >>> drugs = np.array(['X', 'X', 'Y', 'Y', 'Y'])
    >>> weights = compute_balanced_weights(cell_lines, drugs, min_count_baseline=0)
    >>> weights  # (A, X) gets higher weight than (B, Y)

    >>> # With min_count_baseline, rare combinations get less extreme weights
    >>> weights = compute_balanced_weights(cell_lines, drugs, min_count_baseline=10)

    Notes
    -----
    The weight computation follows sklearn's balanced class weighting approach:

    For standard balanced weighting (min_count_baseline=0):
        weight[c] = n_samples / (n_classes * n_samples_c)

    With our modification:
        weight[c] = n_samples / (n_classes * (n_samples_c + min_count_baseline))

    This approach is inspired by sklearn.utils.class_weight.compute_class_weight
    but extended to handle tuple-based combinations and very imbalanced data.

    See Also
    --------
    sklearn.utils.class_weight.compute_class_weight : The original sklearn function
    """
    cell_lines = np.asarray(cell_lines)
    drugs = np.asarray(drugs)

    if len(cell_lines) != len(drugs):
        raise ValueError("cell_lines and drugs must have the same length")

    n_samples = len(cell_lines)

    if n_samples == 0:
        return np.array([], dtype=np.float64)

    # Create tuple keys for each (cell_line, drug) combination
    # Using structured array for efficient unique counting
    combination_keys = np.array(
        [(c, d) for c, d in zip(cell_lines, drugs)],
        dtype=[("cell_line", cell_lines.dtype), ("drug", drugs.dtype)],
    )

    # Get unique combinations and their counts
    unique_combinations, inverse_indices, counts = np.unique(
        combination_keys, return_inverse=True, return_counts=True
    )

    n_combinations = len(unique_combinations)

    # Compute adjusted counts (add baseline to prevent extreme reweighting)
    adjusted_counts = counts.astype(np.float64) + min_count_baseline

    # Compute weights using balanced formula
    # weight[c] = n_samples / (n_combinations * adjusted_count[c])
    combination_weights = n_samples / (n_combinations * adjusted_counts)

    # Map combination weights back to individual samples
    sample_weights = combination_weights[inverse_indices]

    # Normalize weights to sum to n_samples (preserves expected number of samples)
    sample_weights = sample_weights * (n_samples / sample_weights.sum())

    return sample_weights


def compute_class_weights(labels: ArrayLike, min_count_baseline: int = 0) -> np.ndarray:
    """
    Compute balanced class weights for a single label array.

    This is a simpler version for single-label classification, similar to
    sklearn's compute_class_weight with 'balanced' mode.

    Parameters
    ----------
    labels : array-like
        Class labels for each sample
    min_count_baseline : int, default=0
        Minimum count baseline to add to all class counts

    Returns
    -------
    numpy.ndarray
        Weight for each sample

    See Also
    --------
    compute_balanced_weights : For (cell_line, drug) tuple combinations
    """
    labels = np.asarray(labels)
    n_samples = len(labels)

    if n_samples == 0:
        return np.array([], dtype=np.float64)

    # Get unique classes and counts
    unique_classes, inverse_indices, counts = np.unique(
        labels, return_inverse=True, return_counts=True
    )

    n_classes = len(unique_classes)

    # Compute adjusted counts
    adjusted_counts = counts.astype(np.float64) + min_count_baseline

    # Compute weights
    class_weights = n_samples / (n_classes * adjusted_counts)

    # Map to samples
    sample_weights = class_weights[inverse_indices]

    # Normalize
    sample_weights = sample_weights * (n_samples / sample_weights.sum())

    return sample_weights
