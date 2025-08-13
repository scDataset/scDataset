"""
Iterable PyTorch Dataset for on-disk data collections.

This module provides the main :class:`scDataset` class for creating efficient
iterable datasets from on-disk data collections with flexible sampling
strategies and customizable data transformation pipelines.

.. autosummary::
   :toctree: generated/

   scDataset
"""

from typing import Optional, List, Union, Callable
import warnings

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info
from .strategy import SamplingStrategy


class scDataset(IterableDataset):
    """
    Iterable PyTorch Dataset for on-disk data collections with flexible sampling strategies.

    This dataset implementation provides efficient iteration over large on-disk
    data collections using configurable sampling strategies. It supports various
    data transformations, custom fetch/batch callbacks, and automatic handling
    of multiprocessing workers.

    Parameters
    ----------
    data_collection : object
        The data collection to sample from (e.g., AnnCollection, HuggingFace Dataset,
        numpy array, etc.). Must support indexing (``__getitem__``) and length
        (``__len__``) operations.
    strategy : SamplingStrategy
        Strategy for sampling indices from the data collection. Determines the
        order and selection of samples.
    batch_size : int
        Number of samples per minibatch. Must be positive.
    fetch_factor : int, default=1
        Multiplier for fetch size relative to batch size. Higher values may
        improve I/O efficiency by fetching more data at once.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch if it contains fewer than
        ``batch_size`` samples.
    fetch_transform : Callable, optional
        Function to transform data after fetching but before batching.
        Applied to the entire fetch (multiple batches worth of data).
    batch_transform : Callable, optional
        Function to transform each individual batch before yielding.
    fetch_callback : Callable, optional
        Custom function to fetch data given indices. Should accept
        ``(data_collection, indices)`` and return the fetched data.
        If None, uses default indexing (``data_collection[indices]``).
    batch_callback : Callable, optional
        Custom function to extract batch data from fetched data.
        Should accept ``(fetched_data, batch_indices)`` and return the batch.
        If None, uses default indexing (``fetched_data[batch_indices]``).

    Attributes
    ----------
    collection : object
        The underlying data collection.
    strategy : SamplingStrategy
        The sampling strategy being used.
    batch_size : int
        Size of each batch.
    fetch_factor : int
        Fetch size multiplier.
    drop_last : bool
        Whether incomplete batches are dropped.
    fetch_size : int
        Total number of samples fetched at once (batch_size * fetch_factor).
    sort_before_fetch : bool
        Always True. Indices are sorted before fetching for optimal I/O.

    Raises
    ------
    ValueError
        If batch_size or fetch_factor is not positive.
    TypeError
        If data_collection doesn't support required operations or strategy
        is not a SamplingStrategy instance.

    Examples
    --------
    >>> from scdataset import scDataset
    >>> from scdataset.strategy import Streaming
    >>> import numpy as np
    
    >>> # Simple streaming dataset
    >>> data = np.random.randn(1000, 50)  # 1000 samples, 50 features
    >>> strategy = Streaming()
    >>> dataset = scDataset(data, strategy, batch_size=32)
    >>> len(dataset)  # Number of batches
    32
    
    >>> # With custom transforms
    >>> def normalize_batch(batch):
    ...     return (batch - batch.mean()) / batch.std()
    >>> dataset = scDataset(
    ...     data, strategy, batch_size=32, 
    ...     batch_transform=normalize_batch
    ... )
    
    >>> # Iterate through batches
    >>> for batch in dataset:
    ...     print(batch.shape)  # (32, 50) for most batches
    ...     break

    See Also
    --------
    scdataset.strategy.SamplingStrategy : Base class for sampling strategies
    scdataset.strategy.Streaming : Sequential sampling without shuffling
    scdataset.strategy.BlockShuffling : Block-based shuffling
    scdataset.strategy.BlockWeightedSampling : Weighted sampling with blocks
    scdataset.strategy.ClassBalancedSampling : Automatic class balancing

    Notes
    -----
    The dataset automatically handles PyTorch's multiprocessing by distributing
    fetch ranges among workers. Each worker gets a different subset of the data
    to avoid duplication.

    Data is fetched in chunks of size ``batch_size * fetch_factor`` and then
    divided into batches. This can improve I/O efficiency, especially for
    datasets where accessing non-contiguous indices is expensive.
    """
    
    def __init__(
        self, 
        data_collection, 
        strategy: SamplingStrategy,
        batch_size: int, 
        fetch_factor: int = 1, 
        drop_last: bool = False,
        fetch_transform: Optional[Callable] = None, 
        batch_transform: Optional[Callable] = None,
        fetch_callback: Optional[Callable] = None, 
        batch_callback: Optional[Callable] = None
    ):
        """
        Initialize the scDataset.
        
        Parameters
        ----------
        data_collection : object
            Data collection supporting indexing and len().
        strategy : SamplingStrategy
            Sampling strategy instance.
        batch_size : int
            Positive integer for batch size.
        fetch_factor : int, default=1
            Positive integer for fetch size multiplier.
        drop_last : bool, default=False
            Whether to drop incomplete batches.
        fetch_transform : Callable, optional
            Transform applied to fetched data.
        batch_transform : Callable, optional
            Transform applied to each batch.
        fetch_callback : Callable, optional
            Custom fetch function.
        batch_callback : Callable, optional
            Custom batch extraction function.
        """
        # Input validation
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if fetch_factor <= 0:
            raise ValueError("fetch_factor must be positive")
        if not hasattr(data_collection, '__len__') or not hasattr(data_collection, '__getitem__'):
            raise TypeError("data_collection must support indexing and len()")
        if not isinstance(strategy, SamplingStrategy):
            raise TypeError("strategy must be an instance of SamplingStrategy")
        
        self.collection = data_collection
        self.strategy = strategy
        self.batch_size = batch_size
        self.fetch_factor = fetch_factor
        self.drop_last = drop_last
        self.fetch_size = self.batch_size * self.fetch_factor
        self.sort_before_fetch = True  # Always sort before fetch as per new design
        
        # Store callback functions
        self.fetch_transform = fetch_transform
        self.batch_transform = batch_transform
        self.fetch_callback = fetch_callback
        self.batch_callback = batch_callback
        
    def __len__(self) -> int:
        """
        Return the number of batches in the dataset.
        
        Calculates the number of batches that will be yielded by the iterator
        based on the sampling strategy's effective length and the batch size.
        
        Returns
        -------
        int
            Number of batches in the dataset.
            
        Examples
        --------
        >>> from scdataset.strategy import Streaming
        >>> dataset = scDataset(range(100), Streaming(), batch_size=10)
        >>> len(dataset)
        10
        
        >>> # With drop_last=True
        >>> dataset = scDataset(range(105), Streaming(), batch_size=10, drop_last=True)
        >>> len(dataset)  # 105 // 10 = 10 (drops 5 samples)
        10
        
        >>> # With drop_last=False (default)
        >>> dataset = scDataset(range(105), Streaming(), batch_size=10, drop_last=False)
        >>> len(dataset)  # ceil(105 / 10) = 11
        11
        
        Notes
        -----
        When ``drop_last=True``, only complete batches are counted.
        When ``drop_last=False``, the last incomplete batch is included in the count.
        """
        # Get the total number of samples from the sampling strategy
        n = self.strategy.get_len(self.collection)
        
        if self.drop_last:
            # When dropping the last incomplete batch, calculate based on complete batches only
            return n // self.batch_size
        else:
            # When keeping the last incomplete batch, round up
            return (n + self.batch_size - 1) // self.batch_size

        
    def __iter__(self):
        """
        Yield batches of data according to the sampling strategy.
        
        Creates an iterator that yields batches of data by:
        
        1. Getting indices from the sampling strategy
        2. Dividing indices into fetch ranges
        3. Distributing fetch ranges among workers (if multiprocessing)
        4. Fetching data in chunks and applying fetch transforms
        5. Dividing fetched data into batches and applying batch transforms
        6. Yielding transformed batches
        
        Yields
        ------
        object
            Batches of data after applying all transforms. The exact type
            depends on the data collection and any applied transforms.
            
        Examples
        --------
        >>> from scdataset.strategy import Streaming
        >>> import numpy as np
        >>> data = np.random.randn(100, 10)
        >>> dataset = scDataset(data, Streaming(), batch_size=5)
        >>> for i, batch in enumerate(dataset):
        ...     print(f"Batch {i}: shape {batch.shape}")
        ...     if i >= 2:  # Just show first few batches
        ...         break
        Batch 0: shape (5, 10)
        Batch 1: shape (5, 10)
        Batch 2: shape (5, 10)
        
        Notes
        -----        
        The fetch-then-batch approach can improve I/O efficiency by:
        
        - Sorting indices before fetching for better disk access patterns
        - Fetching multiple batches worth of data at once
        - Reducing the number of data access operations
        
        Shuffling behavior is controlled by the sampling strategy's
        ``_shuffle_before_yield`` attribute.
        """
        worker_info = get_worker_info()
        
        # Generate seed for sampling strategy
        if worker_info is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(worker_info.seed - worker_info.id)

        # Get indices from sampling strategy
        indices = self.strategy.get_indices(self.collection, rng=rng)
        
        # Calculate fetch ranges
        n = len(indices)
        fetch_size = self.fetch_size
        num_fetches = (n + fetch_size - 1) // fetch_size
        fetch_ranges = [(i * fetch_size, min((i + 1) * fetch_size, n)) for i in range(num_fetches)]
        
        # Handle multiprocessing by distributing fetch ranges among workers
        if worker_info is not None:
            per_worker = num_fetches // worker_info.num_workers
            remainder = num_fetches % worker_info.num_workers
            if worker_info.id < remainder:
                start = worker_info.id * (per_worker + 1)
                end = start + per_worker + 1
            else:
                start = worker_info.id * per_worker + remainder
                end = start + per_worker
            fetch_ranges = fetch_ranges[start:end]
        
        # Process each fetch range
        for fetch_start, fetch_end in fetch_ranges:
            fetch_indices = indices[fetch_start:fetch_end]
            if self.sort_before_fetch:
                fetch_indices = np.sort(fetch_indices)

            # Use custom fetch callback if provided, otherwise use default indexing
            if self.fetch_callback is not None:
                data = self.fetch_callback(self.collection, fetch_indices)
            else:
                data = self.collection[fetch_indices]

            # Call fetch transform if provided
            if self.fetch_transform is not None:
                data = self.fetch_transform(data)

            if self.strategy._shuffle_before_yield:
                shuffle_indices = rng.permutation(len(fetch_indices))
            else:
                shuffle_indices = np.arange(len(fetch_indices))
            
            # Yield batches
            batch_start = 0
            while batch_start < len(fetch_indices):
                batch_end = min(batch_start + self.batch_size, len(fetch_indices))
                
                # Handle drop_last
                if self.drop_last and (batch_end - batch_start) < self.batch_size:
                    break
                
                # Get batch indices
                batch_indices = shuffle_indices[batch_start:batch_end]
                
                # Use custom batch callback if provided, otherwise use default indexing
                if self.batch_callback is not None:
                    batch_data = self.batch_callback(data, batch_indices)
                else:
                    batch_data = data[batch_indices]
                
                # Call batch transform if provided
                if self.batch_transform is not None:
                    batch_data = self.batch_transform(batch_data)
                    
                yield batch_data
                batch_start = batch_end