"""
Iterable PyTorch Dataset for on-disk data collections.

This module provides the main :class:`scDataset` class for creating efficient
iterable datasets from on-disk data collections with flexible sampling
strategies and customizable data transformation pipelines.

.. autosummary::
   :toctree: generated/

   scDataset
"""

from typing import Callable, Optional

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from .strategy import SamplingStrategy


class scDataset(IterableDataset):
    """
    Iterable PyTorch Dataset for on-disk data collections with flexible sampling strategies.

    This dataset implementation provides efficient iteration over large on-disk
    data collections using configurable sampling strategies. It supports various
    data transformations, custom fetch/batch callbacks, and automatic handling
    of multiprocessing workers and distributed training (DDP).

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
    fetch_factor : int, default=16
        Multiplier for fetch size relative to batch size. Higher values may
        improve I/O efficiency by fetching more data at once.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch if it contains fewer than
        ``batch_size`` samples.
    fetch_callback : Callable, optional
        Custom function to fetch data given indices. Should accept
        ``(data_collection, indices)`` and return the fetched data.
        If None, uses default indexing (``data_collection[indices]``).
    fetch_transform : Callable, optional
        Function to transform data after fetching but before batching.
        Applied to the entire fetch (multiple batches worth of data).
    batch_callback : Callable, optional
        Custom function to extract batch data from fetched data.
        Should accept ``(fetched_data, batch_indices)`` and return the batch.
        If None, uses default indexing (``fetched_data[batch_indices]``).
    batch_transform : Callable, optional
        Function to transform each individual batch before yielding.
    rank : int, optional
        Process rank for distributed training (DDP). If None, auto-detects
        from torch.distributed if initialized. Defaults to 0 for non-distributed.
    world_size : int, optional
        Total number of processes for distributed training (DDP). If None,
        auto-detects from torch.distributed if initialized. Defaults to 1.
    seed : int or None, optional
        Base seed for reproducible shuffling. Combined with auto-incrementing
        epoch counter to produce different shuffling each epoch while ensuring
        reproducibility across runs with the same seed.
        If None, a random seed is generated and shared across all DDP ranks
        to ensure consistent shuffling while providing variety between runs.

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
    rank : int
        Process rank for distributed training.
    world_size : int
        Total number of distributed processes.

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
    >>> for batch in dataset:  # doctest: +ELLIPSIS
    ...     print(batch.shape)
    ...     break
    (32, 50)

    >>> # Distributed Data Parallel (DDP) usage
    >>> # In DDP training script:
    >>> # import torch.distributed as dist
    >>> # dist.init_process_group(...)
    >>> # dataset = scDataset(data, strategy, batch_size=32)  # Auto-detects DDP
    >>> # Or manually specify:
    >>> # dataset = scDataset(data, strategy, batch_size=32, rank=0, world_size=4)

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

    **DDP Support**: When using Distributed Data Parallel, fetches are distributed
    across ranks in round-robin fashion for better load balancing. Each rank
    processes every ``world_size``-th fetch, ensuring no data duplication.
    Combined with PyTorch DataLoader's ``num_workers``, this provides two levels
    of parallelism: across DDP ranks and across DataLoader workers within each rank.
    """

    def __init__(
        self,
        data_collection,
        strategy: SamplingStrategy,
        batch_size: int,
        fetch_factor: int = 16,
        drop_last: bool = False,
        fetch_callback: Optional[Callable] = None,
        fetch_transform: Optional[Callable] = None,
        batch_callback: Optional[Callable] = None,
        batch_transform: Optional[Callable] = None,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        seed: Optional[int] = None,
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
        fetch_factor : int, default=16
            Positive integer for fetch size multiplier.
        drop_last : bool, default=False
            Whether to drop incomplete batches.
        fetch_callback : Callable, optional
            Custom fetch function.
        fetch_transform : Callable, optional
            Transform applied to fetched data.
        batch_callback : Callable, optional
            Custom batch extraction function.
        batch_transform : Callable, optional
            Transform applied to each batch.
        rank : int, optional
            Process rank for DDP. Auto-detected if None.
        world_size : int, optional
            Number of DDP processes. Auto-detected if None.
        seed : int or None, optional
            Base seed for reproducible shuffling. If None, generates a random
            seed shared across DDP ranks for consistent shuffling. Combined
            with auto-incrementing epoch counter for different shuffling each epoch.
        """
        # Input validation
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if fetch_factor <= 0:
            raise ValueError("fetch_factor must be positive")
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
        self.fetch_callback = fetch_callback
        self.fetch_transform = fetch_transform
        self.batch_callback = batch_callback
        self.batch_transform = batch_transform

        # DDP support with auto-detection
        self.rank, self.world_size = self._detect_ddp(rank, world_size)

        # Epoch counter for reproducible shuffling - auto-increments each iteration
        self._epoch = 0
        # Base seed for deterministic shuffling sequences
        # If None, generate random seed shared across DDP ranks
        self._base_seed = self._init_seed(seed)

    def _init_seed(self, seed: Optional[int]) -> int:
        """
        Initialize or generate the base seed for shuffling.

        If a seed is provided, use it directly. If None, generate a random
        seed and broadcast it from rank 0 to all other ranks to ensure
        consistent shuffling across DDP processes.

        Parameters
        ----------
        seed : int or None
            Explicit seed, or None to generate a random shared seed.

        Returns
        -------
        int
            The seed to use for shuffling.
        """
        if seed is not None:
            return seed

        # Generate random seed - will be shared across ranks
        import torch

        # Single process: just generate random seed
        if self.world_size == 1:
            return int(torch.randint(0, 2**31, (1,)).item())

        # Multi-process DDP: broadcast seed from rank 0
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                f"world_size={self.world_size} but torch.distributed is not initialized. "
                "Please call torch.distributed.init_process_group() before creating the dataset."
            )

        # Rank 0 generates seed and broadcasts to all ranks
        # Use device-appropriate tensor for the backend (NCCL needs CUDA tensors)
        backend = dist.get_backend()
        if backend == "nccl" and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.rank}")
            seed_tensor = torch.zeros(1, dtype=torch.int64, device=device)
            if self.rank == 0:
                seed_tensor[0] = torch.randint(0, 2**31, (1,), device=device).item()
        else:
            # gloo and other backends work with CPU tensors
            seed_tensor = torch.zeros(1, dtype=torch.int64)
            if self.rank == 0:
                seed_tensor[0] = torch.randint(0, 2**31, (1,)).item()

        dist.broadcast(seed_tensor, src=0)

        return int(seed_tensor.item())

    def _detect_ddp(self, rank: Optional[int], world_size: Optional[int]) -> tuple:
        """
        Detect or validate DDP settings.

        Auto-detects from torch.distributed if available and initialized,
        otherwise defaults to single-process settings.

        Parameters
        ----------
        rank : int or None
            Explicit rank, or None for auto-detection.
        world_size : int or None
            Explicit world_size, or None for auto-detection.

        Returns
        -------
        tuple
            (rank, world_size) tuple.
        """
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                detected_rank = dist.get_rank()
                detected_world_size = dist.get_world_size()
            else:
                detected_rank = 0
                detected_world_size = 1
        except ImportError:
            detected_rank = 0
            detected_world_size = 1

        final_rank = rank if rank is not None else detected_rank
        final_world_size = world_size if world_size is not None else detected_world_size

        return final_rank, final_world_size

    def __len__(self) -> int:
        """
        Return the number of batches in the dataset for this rank.

        Calculates the number of batches that will be yielded by the iterator
        based on the sampling strategy's effective length, batch size, and
        the number of DDP ranks (if using distributed training).

        Returns
        -------
        int
            Number of batches in the dataset for this rank.

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

        When using DDP (``world_size > 1``), the returned length is the number
        of batches this specific rank will process, which is approximately
        ``total_batches / world_size``.
        """
        # Get the total number of samples from the sampling strategy
        n = self.strategy.get_len(self.collection)

        # Calculate total fetches and per-rank fetches
        fetch_size = self.fetch_size
        num_fetches = (n + fetch_size - 1) // fetch_size

        # Round-robin distribution: this rank gets fetches at positions
        # rank, rank + world_size, rank + 2*world_size, ...
        rank_fetch_ids = list(range(self.rank, num_fetches, self.world_size))

        # Check if this rank gets any fetches
        if len(rank_fetch_ids) == 0:
            return 0

        # Calculate exact number of samples for this rank
        per_rank_samples = 0
        for fetch_id in rank_fetch_ids:
            fetch_start = fetch_id * fetch_size
            fetch_end = min((fetch_id + 1) * fetch_size, n)
            per_rank_samples += fetch_end - fetch_start

        # Calculate batches from samples, accounting for drop_last
        if self.drop_last:
            # Each fetch may have leftover samples that don't form a complete batch
            # We need to count batches per fetch, not total samples
            num_batches = 0
            for fetch_id in rank_fetch_ids:
                fetch_start = fetch_id * fetch_size
                fetch_end = min((fetch_id + 1) * fetch_size, n)
                fetch_samples = fetch_end - fetch_start
                num_batches += fetch_samples // self.batch_size
            return num_batches
        else:
            # Each fetch yields ceil(fetch_samples / batch_size) batches
            num_batches = 0
            for fetch_id in rank_fetch_ids:
                fetch_start = fetch_id * fetch_size
                fetch_end = min((fetch_id + 1) * fetch_size, n)
                fetch_samples = fetch_end - fetch_start
                num_batches += (fetch_samples + self.batch_size - 1) // self.batch_size
            return num_batches

    def __iter__(self):
        """
        Yield batches of data according to the sampling strategy.

        Creates an iterator that yields batches of data by:

        1. Getting indices from the sampling strategy (same across all DDP ranks)
        2. Dividing indices into fetch ranges
        3. Distributing fetch ranges among DDP ranks (round-robin)
        4. Further distributing among DataLoader workers (if multiprocessing)
        5. Fetching data in chunks and applying fetch transforms
        6. Dividing fetched data into batches and applying batch transforms
        7. Yielding transformed batches

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

        **DDP Distribution**: When using multiple ranks (``world_size > 1``),
        fetches are distributed in round-robin fashion. Rank 0 gets fetches
        0, world_size, 2*world_size, etc. Rank 1 gets fetches 1, world_size+1,
        2*world_size+1, etc. This ensures even load distribution and that
        all data is processed exactly once across all ranks.

        **Auto-incrementing epoch**: The epoch counter automatically increments
        each time the dataset is iterated. This ensures different shuffling
        each epoch without requiring manual ``set_epoch()`` calls.
        """
        worker_info = get_worker_info()

        # Generate seed for sampling strategy - combine base_seed with epoch
        # All ranks use the same seed for consistent global ordering
        # epoch * 1000 provides sufficient separation between epochs
        current_seed = self._base_seed + self._epoch * 1000

        # Auto-increment epoch for next iteration (different shuffling each epoch)
        self._epoch += 1

        if worker_info is None:
            rng = np.random.default_rng(current_seed)
        else:
            # All workers use the same seed for consistent global ordering
            # (they partition work, not randomness)
            rng = np.random.default_rng(current_seed)

        # Get indices from sampling strategy - same ordering across all ranks
        indices = self.strategy.get_indices(self.collection, seed=current_seed)

        # Calculate fetch ranges
        n = len(indices)
        fetch_size = self.fetch_size
        num_fetches = (n + fetch_size - 1) // fetch_size

        # DDP: Distribute fetches among ranks in round-robin fashion
        # This rank gets fetches: rank, rank + world_size, rank + 2*world_size, ...
        rank_fetch_ids = list(range(self.rank, num_fetches, self.world_size))

        # Build fetch ranges for this rank only
        fetch_ranges = [
            (i * fetch_size, min((i + 1) * fetch_size, n)) for i in rank_fetch_ids
        ]

        # Handle DataLoader multiprocessing by distributing fetch ranges among workers
        if worker_info is not None and len(fetch_ranges) > 0:
            num_rank_fetches = len(fetch_ranges)
            per_worker = num_rank_fetches // worker_info.num_workers
            remainder = num_rank_fetches % worker_info.num_workers
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
