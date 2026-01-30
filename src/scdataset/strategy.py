"""
Sampling strategies for on-disk data collections.

This module provides various sampling strategies for efficiently iterating through
large on-disk datasets. Each strategy defines how indices are generated and
ordered for data loading.

.. autosummary::
   :toctree: generated/

   SamplingStrategy
   Streaming
   BlockShuffling
   BlockWeightedSampling
   ClassBalancedSampling
"""

import warnings
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "SamplingStrategy",
    "Streaming",
    "BlockShuffling",
    "BlockWeightedSampling",
    "ClassBalancedSampling",
]


class SamplingStrategy:
    """
    Abstract base class for sampling strategies.

    This class defines the interface that all sampling strategies must implement.
    Sampling strategies determine how indices are generated from a data collection
    for training or inference.

    Attributes
    ----------
    _shuffle_before_yield : bool or None
        Whether to shuffle indices before yielding batches. Set by subclasses.
    _indices : numpy.ndarray or None
        Stored subset of indices if provided. Always sorted for optimal I/O.

    Notes
    -----
    All subclasses must implement the :meth:`get_indices` method to define
    their specific sampling behavior.

    scDataset relies on sorted indices for efficient sequential I/O access
    patterns. When indices are provided to any strategy, they are automatically
    sorted to ensure optimal performance.

    Examples
    --------
    >>> # Custom sampling strategy
    >>> class CustomStrategy(SamplingStrategy):
    ...     def get_indices(self, data_collection, seed=None, rng=None):
    ...         return np.arange(len(data_collection))
    """

    def __init__(self, indices: Optional[ArrayLike] = None):
        """
        Initialize the sampling strategy.

        Parameters
        ----------
        indices : array-like, optional
            Subset of indices to use for sampling. If provided, they will be
            automatically sorted to ensure optimal I/O performance.
        """
        self._shuffle_before_yield = None
        self._indices = self._validate_and_sort_indices(indices)

    def _validate_and_sort_indices(
        self, indices: Optional[ArrayLike]
    ) -> Optional[np.ndarray]:
        """
        Validate and sort indices for optimal I/O performance.

        scDataset relies on sorted indices for efficient sequential I/O access.
        This method ensures any provided indices are sorted and emits a warning
        if reordering was necessary.

        Parameters
        ----------
        indices : array-like, optional
            Indices to validate and sort.

        Returns
        -------
        numpy.ndarray or None
            Sorted indices array, or None if no indices provided.

        Warns
        -----
        UserWarning
            If indices were not already sorted and had to be reordered.
        """
        if indices is None:
            return None

        indices = np.asarray(indices)
        sorted_indices = np.sort(indices)
        if not np.array_equal(indices, sorted_indices):
            warnings.warn(
                "Provided indices were not sorted. They have been automatically "
                "sorted to ensure optimal I/O performance. scDataset relies on "
                "sorted indices for efficient data access patterns.",
                UserWarning,
                stacklevel=2,
            )
        return sorted_indices

    def get_indices(
        self,
        data_collection,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.intp]:
        """
        Generate indices for sampling from the data collection.

        This is an abstract method that must be implemented by all subclasses.

        Parameters
        ----------
        data_collection : object
            The data collection to sample from.
        seed : int, optional
            Random seed for reproducible sampling. Ignored if ``rng`` is provided.
        rng : numpy.random.Generator, optional
            Random number generator to use. If provided, ``seed`` is ignored.

        Returns
        -------
        numpy.ndarray
            Array of indices to sample from the data collection.

        Raises
        ------
        NotImplementedError
            Always raised as this is an abstract method.
        """
        raise NotImplementedError("Subclasses must implement get_indices method")

    def _get_rng(
        self, seed: Optional[int] = None, rng: Optional[np.random.Generator] = None
    ) -> np.random.Generator:
        """
        Get a random number generator from seed or rng parameter.

        This helper method provides a consistent way to obtain a random number
        generator across all sampling strategies.

        Parameters
        ----------
        seed : int, optional
            Random seed to create a new generator. Ignored if ``rng`` is provided.
        rng : numpy.random.Generator, optional
            Existing random number generator to use. Takes precedence over ``seed``.

        Returns
        -------
        numpy.random.Generator
            Random number generator instance.

        Examples
        --------
        >>> strategy = SamplingStrategy()
        >>> rng = strategy._get_rng(seed=42)
        >>> isinstance(rng, np.random.Generator)
        True
        """
        if rng is not None:
            return rng
        return np.random.default_rng(seed)


class Streaming(SamplingStrategy):
    """
    Sequential streaming sampling strategy with optional buffer-level shuffling.

    This strategy provides indices in sequential order, with optional shuffling
    at the buffer level (defined by fetch_factor in scDataset). When shuffle=True,
    batches within each fetch buffer are shuffled, similar to Ray Dataset or
    WebDataset behavior, while maintaining overall sequential order across buffers.

    Parameters
    ----------
    indices : array-like, optional
        Subset of indices to use for sampling. If None, uses all indices
        from 0 to len(data_collection)-1.
    shuffle : bool, default=False
        Whether to shuffle batches within each fetch buffer. When True,
        enables buffer-level shuffling that maintains sequential order
        between buffers but randomizes the order of batches within each
        buffer (defined by fetch_factor * batch_size).

    Attributes
    ----------
    _shuffle_before_yield : bool
        Controlled by the shuffle parameter. True if buffer-level shuffling
        is enabled, False otherwise.
    _indices : numpy.ndarray or None
        Stored subset of indices if provided.
    shuffle : bool
        Whether buffer-level shuffling is enabled.

    Examples
    --------
    >>> # Stream through entire dataset without shuffling
    >>> strategy = Streaming()
    >>> indices = strategy.get_indices(range(100))
    >>> len(indices)
    100

    >>> # Stream through subset of indices
    >>> subset_strategy = Streaming(indices=[10, 20, 30])
    >>> indices = subset_strategy.get_indices(range(100))
    >>> indices.tolist()
    [10, 20, 30]

    >>> # Stream with buffer-level shuffling (like Ray Dataset/WebDataset)
    >>> shuffle_strategy = Streaming(shuffle=True)
    >>> # Batches within each fetch buffer will be shuffled,
    >>> # but buffers themselves maintain sequential order

    See Also
    --------
    BlockShuffling : For shuffled block-based sampling
    BlockWeightedSampling : For weighted sampling with shuffling

    Notes
    -----
    When shuffle=True, this strategy provides behavior similar to:

    - Ray Dataset's local shuffling within windows
    - WebDataset's shuffle buffer functionality

    The key difference from BlockShuffling is that Streaming maintains
    the overall sequential order of fetch buffers, only shuffling within
    each buffer, while BlockShuffling shuffles the order of blocks themselves.
    """

    def __init__(self, indices: Optional[ArrayLike] = None, shuffle: bool = False):
        """
        Initialize streaming strategy.

        Parameters
        ----------
        indices : array-like, optional
            Subset of indices to stream through. If None, streams through
            all available indices. Indices will be automatically sorted
            to ensure optimal I/O performance.
        shuffle : bool, default=False
            Whether to enable buffer-level shuffling. When True, batches
            within each fetch buffer are shuffled while maintaining
            sequential order between buffers.
        """
        super().__init__(indices=indices)
        self.shuffle = shuffle
        self._shuffle_before_yield = shuffle

    def get_len(self, data_collection) -> int:
        """
        Get the effective length of the data collection for this strategy.

        Parameters
        ----------
        data_collection : object
            The data collection to get length from. Must support ``len()``.

        Returns
        -------
        int
            Number of samples that will be yielded by this strategy.

        Examples
        --------
        >>> strategy = Streaming()
        >>> strategy.get_len(range(100))
        100

        >>> subset_strategy = Streaming(indices=[1, 3, 5])
        >>> subset_strategy.get_len(range(100))
        3
        """
        if self._indices is None:
            return len(data_collection)
        return len(self._indices)

    def get_indices(
        self,
        data_collection,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.intp]:
        """
        Get indices for streaming sampling.

        Returns indices in sequential order. If shuffle=True was set during
        initialization, the _shuffle_before_yield attribute will cause
        buffer-level shuffling during iteration.

        Parameters
        ----------
        data_collection : object
            The data collection to sample from. Must support ``len()``.
        seed : int, optional
            Random seed. Only used if shuffle=True for buffer-level shuffling
            during iteration, not for index generation which remains sequential.
        rng : numpy.random.Generator, optional
            Random number generator. Only used if shuffle=True for buffer-level
            shuffling during iteration.

        Returns
        -------
        numpy.ndarray
            Array of indices in sequential order.

        Examples
        --------
        >>> strategy = Streaming()
        >>> indices = strategy.get_indices(range(5))
        >>> indices.tolist()
        [0, 1, 2, 3, 4]

        >>> subset_strategy = Streaming(indices=[2, 4, 6])
        >>> indices = subset_strategy.get_indices(range(10))
        >>> indices.tolist()
        [2, 4, 6]

        >>> # With shuffle=True, indices are still sequential
        >>> shuffle_strategy = Streaming(shuffle=True)
        >>> indices = shuffle_strategy.get_indices(range(5))
        >>> indices.tolist()
        [0, 1, 2, 3, 4]
        """
        if self._indices is None:
            return np.arange(len(data_collection))
        return self._indices


class BlockShuffling(SamplingStrategy):
    """
    Block-based shuffling sampling strategy.

    This strategy divides the data into blocks of fixed size and shuffles
    the order of blocks while maintaining the original order within each block.
    This provides a balance between randomization and maintaining some locality
    of data access patterns.

    Parameters
    ----------
    block_size : int, default=8
        Size of each block for shuffling. Larger blocks maintain more locality
        but provide less randomization.
    indices : array-like, optional
        Subset of indices to use for sampling. If None, uses all indices
        from 0 to len(data_collection)-1.
    drop_last : bool, default=False
        Whether to drop the last incomplete block if the total number of
        indices is not divisible by block_size.

    Attributes
    ----------
    _shuffle_before_yield : bool
        Always True for block shuffling strategy.
    _indices : numpy.ndarray or None
        Stored subset of indices if provided.
    block_size : int
        Size of blocks for shuffling.
    drop_last : bool
        Whether to drop incomplete blocks.

    Notes
    -----
    When ``drop_last=False`` and there's a remainder block smaller than
    ``block_size``, it's inserted at a random position among the shuffled
    complete blocks.

    Examples
    --------
    >>> # Basic block shuffling
    >>> strategy = BlockShuffling(block_size=3)
    >>> np.random.seed(42)  # For reproducible example
    >>> indices = strategy.get_indices(range(10), seed=42)
    >>> len(indices)
    10

    >>> # Drop incomplete blocks
    >>> strategy = BlockShuffling(block_size=3, drop_last=True)
    >>> indices = strategy.get_indices(range(10), seed=42)
    >>> len(indices)  # 10 // 3 * 3 = 9
    9

    See Also
    --------
    Streaming : For sequential sampling without shuffling
    BlockWeightedSampling : For weighted block-based sampling
    """

    def __init__(
        self,
        block_size: int = 8,
        indices: Optional[ArrayLike] = None,
        drop_last: bool = False,
    ):
        """
        Initialize block shuffling strategy.

        Parameters
        ----------
        block_size : int, default=8
            Size of blocks for shuffling. Must be positive.
        indices : array-like, optional
            Subset of indices to sample from. Indices will be automatically
            sorted to ensure optimal I/O performance.
        drop_last : bool, default=False
            Whether to drop the last incomplete block.

        Raises
        ------
        ValueError
            If block_size is not positive.
        """
        super().__init__(indices=indices)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self._shuffle_before_yield = True
        self.block_size = block_size
        self.drop_last = drop_last

    def get_len(self, data_collection) -> int:
        """
        Get the effective length of the data collection for this strategy.

        Takes into account the drop_last setting when calculating the
        effective length.

        Parameters
        ----------
        data_collection : object
            The data collection to get length from. Must support ``len()``.

        Returns
        -------
        int
            Number of samples that will be yielded by this strategy.

        Examples
        --------
        >>> strategy = BlockShuffling(block_size=3, drop_last=False)
        >>> strategy.get_len(range(10))
        10

        >>> strategy = BlockShuffling(block_size=3, drop_last=True)
        >>> strategy.get_len(range(10))  # 10 - (10 % 3) = 9
        9
        """
        if self._indices is None:
            length = len(data_collection)
        else:
            length = len(self._indices)
        if self.drop_last:
            length -= length % self.block_size
        return length

    def get_indices(
        self,
        data_collection,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.intp]:
        """
        Generate indices with block-based shuffling.

        Divides indices into blocks and shuffles the order of complete blocks.
        Incomplete blocks are either dropped or inserted at random positions
        depending on the ``drop_last`` setting.

        Parameters
        ----------
        data_collection : object
            The data collection to sample from. Must support ``len()``.
        seed : int, optional
            Random seed for reproducible shuffling. Ignored if ``rng`` is provided.
        rng : numpy.random.Generator, optional
            Random number generator to use for shuffling. If provided, ``seed`` is ignored.

        Returns
        -------
        numpy.ndarray
            Array of indices with blocks shuffled.

        Notes
        -----
        When ``drop_last=True`` and there are remainder indices that don't form
        a complete block, they are randomly removed from the dataset.

        When ``drop_last=False``, remainder indices are inserted at a random
        position among the shuffled complete blocks.

        Examples
        --------
        >>> strategy = BlockShuffling(block_size=2, drop_last=False)
        >>> indices = strategy.get_indices(range(5), seed=42)
        >>> len(indices)
        5

        >>> strategy = BlockShuffling(block_size=2, drop_last=True)
        >>> indices = strategy.get_indices(range(5), seed=42)
        >>> len(indices)  # Drops the last incomplete block
        4

        Raises
        ------
        ValueError
            If the random number generator cannot sample the required number
            of indices for removal when drop_last=True.
        """
        if self._indices is None:
            indices = np.arange(len(data_collection))
        else:
            indices = self._indices
        rng_obj = self._get_rng(seed, rng)
        n = len(indices)
        n_blocks = n // self.block_size
        n_complete = n_blocks * self.block_size
        remainder = n - n_complete

        if self.drop_last and remainder > 0:
            remove_indices = rng_obj.choice(indices, size=remainder, replace=False)
            mask = ~np.isin(indices, remove_indices)
            complete_part = indices[mask]
        else:
            complete_part = indices[:n_complete]

        # Reshape complete part into blocks
        blocks = complete_part.reshape(n_blocks, self.block_size)
        blocks = rng_obj.permutation(blocks, axis=0)

        if self.drop_last or remainder == 0:
            return blocks.reshape(-1)
        else:
            # Insert remainder block at a random block boundary
            insert_pos = rng_obj.integers(0, n_blocks + 1)
            before = blocks[:insert_pos].reshape(-1)
            after = blocks[insert_pos:].reshape(-1)
            return np.concatenate([before, indices[n_complete:], after])


class BlockWeightedSampling(SamplingStrategy):
    """
    Weighted sampling with block-based shuffling.

    This strategy performs weighted sampling from the data collection and then
    applies block-based shuffling to the sampled indices. It supports both
    sampling with and without replacement, and can generate a different total
    number of samples than the original data collection size.

    Parameters
    ----------
    block_size : int, default=8
        Size of blocks for shuffling after weighted sampling.
    indices : array-like, optional
        Subset of indices to sample from. If None, uses all indices.
    weights : array-like, optional
        Sampling weights for each element in the data collection.
        Must be non-negative and sum to a positive value.
        If None, uses uniform sampling.
    total_size : int, optional
        Total number of samples to draw. If None, uses the length of
        indices or data_collection.
    replace : bool, default=True
        Whether to sample with replacement.
    sampling_size : int, optional
        Size of each sampling round when ``replace=False``.
        Required when ``replace=False``.

    Attributes
    ----------
    _shuffle_before_yield : bool
        Always True for weighted sampling strategy.
    _indices : numpy.ndarray or None
        Stored subset of indices if provided.
    block_size : int
        Size of blocks for shuffling.
    weights : numpy.ndarray or None
        Normalized sampling weights.
    total_size : int or None
        Total number of samples to generate.
    replace : bool
        Whether sampling is with replacement.
    sampling_size : int or None
        Size of each sampling round for replacement=False.

    Raises
    ------
    ValueError
        If weights are negative, sum to zero, or don't match data collection length.
        If sampling_size is not provided when replace=False.

    Warns
    -----
    UserWarning
        If sampling_size is provided when replace=True (it will be ignored).

    Examples
    --------
    >>> # Uniform weighted sampling
    >>> strategy = BlockWeightedSampling(block_size=2, total_size=6)
    >>> indices = strategy.get_indices(range(4), seed=42)
    >>> len(indices)
    6

    >>> # Custom weights favoring certain indices
    >>> weights = [0.1, 0.1, 0.4, 0.4]  # Favor indices 2 and 3
    >>> strategy = BlockWeightedSampling(weights=weights, total_size=8)
    >>> indices = strategy.get_indices(range(4), seed=42)
    >>> len(indices)
    8

    >>> # Sampling without replacement
    >>> strategy = BlockWeightedSampling(
    ...     total_size=10, replace=False, sampling_size=5
    ... )

    See Also
    --------
    BlockShuffling : For unweighted block-based shuffling
    ClassBalancedSampling : For automatic class-balanced sampling
    """

    def __init__(
        self,
        block_size: int = 8,
        indices: Optional[ArrayLike] = None,
        weights: Optional[ArrayLike] = None,
        total_size: Optional[int] = None,
        replace: bool = True,
        sampling_size: Optional[int] = None,
    ):
        """
        Initialize weighted sampling strategy.

        Parameters
        ----------
        block_size : int, default=8
            Size of blocks for shuffling. Must be positive.
        indices : array-like, optional
            Subset of indices to sample from. Indices will be automatically
            sorted to ensure optimal I/O performance.
        weights : array-like, optional
            Sampling weights. Will be normalized automatically.
        total_size : int, optional
            Total number of samples to generate.
        replace : bool, default=True
            Whether to sample with replacement.
        sampling_size : int, optional
            Required when replace=False. Size of each sampling round.

        Raises
        ------
        ValueError
            If block_size is not positive, weights are invalid, or
            sampling_size is missing when replace=False.
        """
        super().__init__(indices=indices)
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self._shuffle_before_yield = True
        self.block_size = block_size
        if weights is not None:
            weights = np.asarray(weights)
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
            if np.sum(weights) == 0:
                raise ValueError("weights must sum to a positive value")
            # Normalize weights
            weights = weights / np.sum(weights)
        self.weights = weights
        self.total_size = total_size
        self.replace = replace
        if not replace and sampling_size is None:
            raise ValueError("sampling_size must be provided when replace=False")

        if replace and sampling_size is not None:
            warnings.warn(
                "sampling_size is ignored when replace=True, since it will sample with replacement",
                stacklevel=2,
            )
        self.sampling_size = sampling_size

    def get_len(self, data_collection) -> int:
        """
        Get the effective length of the data collection for this strategy.

        Returns the total number of samples that will be generated,
        which may be different from the original data collection size.

        Parameters
        ----------
        data_collection : object
            The data collection to get length from. Must support ``len()``.

        Returns
        -------
        int
            Number of samples that will be yielded by this strategy.

        Examples
        --------
        >>> strategy = BlockWeightedSampling(total_size=100)
        >>> strategy.get_len(range(50))  # Returns total_size
        100

        >>> strategy = BlockWeightedSampling()  # No total_size specified
        >>> strategy.get_len(range(50))  # Returns collection length
        50
        """
        if self.total_size is not None:
            length = self.total_size
        else:
            # Use the length of indices or data_collection if total_size not specified
            if self._indices is None:
                length = len(data_collection)
            else:
                length = len(self._indices)
        return length

    def get_indices(
        self,
        data_collection,
        seed: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> NDArray[np.intp]:
        """
        Generate indices using weighted sampling followed by block shuffling.

        First performs weighted sampling (with or without replacement) to select
        indices, then applies block-based shuffling to the selected indices.

        Parameters
        ----------
        data_collection : object
            The data collection to sample from. Must support ``len()``.
        seed : int, optional
            Random seed for reproducible sampling. Ignored if ``rng`` is provided.
        rng : numpy.random.Generator, optional
            Random number generator to use. If provided, ``seed`` is ignored.

        Returns
        -------
        numpy.ndarray
            Array of indices after weighted sampling and block shuffling.

        Raises
        ------
        ValueError
            If weights don't match the data collection length.

        Notes
        -----
        When ``replace=False``, sampling is performed in rounds of size
        ``sampling_size`` until the desired ``total_size`` is reached.

        Examples
        --------
        >>> # Weighted sampling with replacement
        >>> weights = [0.25, 0.25, 0.25, 0.25]  # Uniform weights
        >>> strategy = BlockWeightedSampling(
        ...     weights=weights, total_size=8, block_size=2
        ... )
        >>> indices = strategy.get_indices(range(4), seed=42)
        >>> len(indices)
        8

        >>> # Sampling without replacement
        >>> strategy = BlockWeightedSampling(
        ...     total_size=6, replace=False, sampling_size=3, block_size=2
        ... )
        >>> indices = strategy.get_indices(range(10), seed=42)
        >>> len(indices)
        6
        """
        # Handle weights validation - must match either data_collection or indices
        working_weights = None
        if self.weights is not None:
            if self._indices is not None:
                # When both weights and indices are provided, weights should match indices length
                if len(self.weights) == len(self._indices):
                    # Weights already match indices - use directly
                    working_weights = self.weights / self.weights.sum()
                elif len(self.weights) == len(data_collection):
                    # Full weights provided - extract subset for indices
                    subset_weights = self.weights[self._indices]
                    working_weights = subset_weights / subset_weights.sum()
                else:
                    raise ValueError(
                        f"weights length ({len(self.weights)}) must match either "
                        f"data_collection length ({len(data_collection)}) or "
                        f"indices length ({len(self._indices)})"
                    )
            else:
                # No indices - weights must match data_collection
                if len(self.weights) != len(data_collection):
                    raise ValueError(
                        "weights must have the same length as data_collection"
                    )
                working_weights = self.weights / self.weights.sum()

        if self._indices is None:
            _indices = np.arange(len(data_collection))
        else:
            _indices = self._indices

        rng_obj = self._get_rng(seed, rng)
        if self.replace:
            # Sample with replacement
            if self.total_size is not None:
                size = self.total_size
            else:
                size = len(_indices)
            indices = rng_obj.choice(
                _indices, size=size, replace=True, p=working_weights
            )
        else:
            # Sample without replacement until we have total_size
            sampled = 0
            indices_list = []
            while sampled < self.total_size:
                remaining = self.total_size - sampled
                current_size = min(self.sampling_size, remaining)
                new_indices = rng_obj.choice(
                    _indices, size=current_size, replace=False, p=working_weights
                )
                indices_list.append(new_indices)
                sampled += len(new_indices)
            indices = np.concatenate(indices_list)

        indices.sort()

        n = len(indices)
        n_blocks = n // self.block_size
        n_complete = n_blocks * self.block_size
        complete_part = indices[:n_complete]
        remainder_part = indices[n_complete:]

        # Reshape complete part into blocks
        blocks = complete_part.reshape(n_blocks, self.block_size)
        rng_obj = self._get_rng(seed, rng)
        blocks = rng_obj.permutation(blocks, axis=0)

        if len(remainder_part) == 0:
            # Only return shuffled complete blocks
            return blocks.reshape(-1)
        else:
            # Insert remainder block at a random block boundary
            insert_pos = rng_obj.integers(0, n_blocks + 1)
            before = blocks[:insert_pos].reshape(-1)
            after = blocks[insert_pos:].reshape(-1)
            return np.concatenate([before, remainder_part, after])


class ClassBalancedSampling(BlockWeightedSampling):
    """
    Class-balanced sampling with automatic weight computation.

    This strategy extends :class:`BlockWeightedSampling` by automatically computing
    balanced weights from provided labels, making each class equally likely to be
    sampled regardless of the original class distribution in the dataset.

    The weights are computed as the inverse of class frequencies, ensuring that
    underrepresented classes get higher sampling probability and overrepresented
    classes get lower sampling probability.

    **Dual Behavior for Labels:**

    The strategy supports two modes based on the labels array length:

    1. **Global class balancing** (labels length = full dataset):
       Weights are computed from the full dataset's class distribution. When
       sampling from a subset (via ``indices``), samples are weighted according
       to their importance in the global distribution, not the subset.

    2. **Subset class balancing** (labels length = indices length):
       Weights are computed only from the labels of the subset indices.
       This balances classes within the subset, ignoring the global distribution.

    Parameters
    ----------
    labels : array-like
        Class labels for each sample. The length determines the balancing mode:

        - If ``len(labels) == len(data_collection)``: Global balancing mode.
          Weights computed from full dataset, then applied to subset.
        - If ``len(labels) == len(indices)``: Subset balancing mode.
          Weights computed only from the subset's labels.

    block_size : int, default=8
        Size of blocks for block shuffling after sampling.
    indices : array-like, optional
        Subset of indices to sample from. If None, uses all indices.
    total_size : int, optional
        Total number of samples to draw. If None, uses the length of indices
        or data_collection.
    replace : bool, default=True
        Whether to sample with replacement.
    sampling_size : int, optional
        Size of each sampling round when ``replace=False``. Required when
        ``replace=False``.

    Attributes
    ----------
    labels : numpy.ndarray
        Array of class labels for each sample.

    Raises
    ------
    ValueError
        If labels array is empty.

    Examples
    --------
    Global balancing - balance for full dataset distribution:

    >>> # Full dataset: 90% class 0, 10% class 1
    >>> full_labels = [0]*90 + [1]*10  # 100 samples total
    >>> subset_indices = [0, 1, 90, 91, 92, 93, 94, 95, 96, 97]  # 2 of class 0, 8 of class 1
    >>>
    >>> # Global balancing: uses full dataset weights
    >>> strategy = ClassBalancedSampling(full_labels, indices=subset_indices, total_size=20)
    >>> # Class 1 samples get ~9x higher weight (because 1/10 vs 1/90 in global dist)
    >>> # Even though subset is 80% class 1, global weights still favor class 1

    Subset balancing - balance within the subset only:

    >>> # Only provide labels for the subset indices
    >>> subset_labels = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]  # Labels for subset: 20% class 0, 80% class 1
    >>> strategy = ClassBalancedSampling(subset_labels, indices=subset_indices, total_size=20)
    >>> # Now class 0 samples get 4x higher weight (because 1/2 vs 1/8 in subset dist)
    >>> # This balances within the subset, ignoring global distribution

    See Also
    --------
    BlockWeightedSampling : For manual weight specification
    BlockShuffling : For unweighted sampling

    Notes
    -----
    The computed weights ensure that each class has equal probability of being
    sampled, not that each class appears equally often in the final sample.
    The actual class distribution in samples will depend on the random sampling
    process and may vary between different runs.

    When using global balancing with a subset that has different class proportions
    than the full dataset, the output may appear imbalanced relative to the subset.
    This is intentional - the weights reflect global importance.
    """

    def __init__(
        self,
        labels: ArrayLike,
        block_size: int = 8,
        indices: Optional[ArrayLike] = None,
        total_size: Optional[int] = None,
        replace: bool = True,
        sampling_size: Optional[int] = None,
    ):
        """
        Initialize class-balanced sampling strategy.

        Parameters
        ----------
        labels : array-like
            Class labels for samples. The length of labels determines the
            balancing mode (see class docstring for details):

            - If ``len(labels) == len(indices)``: **subset balancing** mode.
              Labels correspond to the subset samples only.
            - If ``len(labels) > len(indices)``: **global balancing** mode.
              Labels correspond to the full dataset.
        block_size : int, default=8
            Size of blocks for shuffling. Must be positive.
        indices : array-like, optional
            Subset of indices to sample from.
        total_size : int, optional
            Total number of samples to generate.
        replace : bool, default=True
            Whether to sample with replacement.
        sampling_size : int, optional
            Required when replace=False. Size of each sampling round.

        Raises
        ------
        ValueError
            If labels array is empty, block_size is not positive, or labels
            length doesn't match indices length (for subset mode) or exceed it
            (for global mode).
        """
        # Store labels and validate basic properties
        self.labels = np.asarray(labels)
        if len(self.labels) == 0:
            raise ValueError("labels cannot be empty")

        # Store indices for validation
        self._init_indices = np.asarray(indices) if indices is not None else None

        # Validate labels length vs indices length
        # - Subset mode: len(labels) == len(indices) - balance within subset
        # - Global mode: len(labels) > len(indices) - use global class frequencies
        if self._init_indices is not None and len(self.labels) < len(
            self._init_indices
        ):
            raise ValueError(
                f"labels length ({len(self.labels)}) must be either equal to "
                f"indices length ({len(self._init_indices)}) for subset balancing, "
                f"or greater than indices length for global balancing"
            )

        weights = self._compute_class_weights()

        super().__init__(
            block_size=block_size,
            indices=indices,
            weights=weights,
            total_size=total_size,
            replace=replace,
            sampling_size=sampling_size,
        )

    def _compute_class_weights(self) -> NDArray[np.floating]:
        """
        Compute balanced weights for each sample based on inverse class frequency.

        The weight computation depends on the balancing mode:

        - **Subset mode** (``len(labels) == len(indices)``): Weights are computed
          from the provided labels directly, which represent the subset's class
          distribution. This balances classes relative to the subset.

        - **Global mode** (``len(labels) > len(indices)`` or no indices): Weights
          are computed from the full label array to preserve global class importance.
          Samples from rare global classes get higher weights even if they're
          common in the subset.

        Returns
        -------
        numpy.ndarray
            Weights array. In subset mode, length equals ``len(labels)`` which
            equals ``len(indices)``. In global mode, length equals the full
            ``len(labels)``, and the parent class handles subsetting.

        Notes
        -----
        This method is called internally during ``__init__`` to compute
        weights before they are passed to the parent class.

        The weights are not normalized to sum to 1.0, as this normalization
        is handled by the parent class :class:`BlockWeightedSampling`.

        Example (Subset Mode):
            For labels ``[0, 0, 1, 2, 2, 2]`` (subset labels only):

            - Class 0: 2 samples -> weight 1/2 = 0.5 per sample
            - Class 1: 1 sample  -> weight 1/1 = 1.0 per sample
            - Class 2: 3 samples -> weight 1/3 ≈ 0.33 per sample

            Result: ``array([0.5, 0.5, 1.0, 0.333, 0.333, 0.333])``

        Example (Global Mode):
            If full dataset labels are ``[0, 0, 0, 0, 1]`` (80% class 0, 20% class 1)
            and indices select samples 3, 4 (classes 0, 1):

            - Class 0: 4 samples globally -> weight 1/4 = 0.25
            - Class 1: 1 sample globally  -> weight 1/1 = 1.0

            Subset sample 3 (class 0) gets weight 0.25,
            Subset sample 4 (class 1) gets weight 1.0.
            Class 1 samples are 4x more likely to be sampled despite
            being 50% of the subset.
        """
        unique_classes, class_counts = np.unique(self.labels, return_counts=True)

        # Compute inverse frequency weights
        class_weights = 1.0 / class_counts

        # Create sample weights by mapping class weights to each sample
        weights = np.zeros(len(self.labels))
        for cls, weight in zip(unique_classes, class_weights):
            weights[self.labels == cls] = weight

        return weights
