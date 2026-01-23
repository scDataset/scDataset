"""
Automatic configuration utilities for scDataset.

This module provides functions to automatically suggest optimal parameters
for scDataset based on system resources and data characteristics.

.. autosummary::
   :toctree: generated/

   suggest_parameters
   estimate_sample_size
"""

import os
import sys
import warnings
from typing import Any, Callable, Dict, Optional, Set

import numpy as np


def _deep_sizeof(obj: Any, seen: Optional[Set[int]] = None) -> int:
    """
    Recursively estimate memory size of an object in bytes.

    Handles numpy arrays, scipy sparse matrices, torch tensors,
    pandas DataFrames/Series, dicts, lists, tuples, strings,
    AnnData objects, and nested structures.

    Parameters
    ----------
    obj : Any
        The object to estimate size for.
    seen : set, optional
        Set of object IDs already counted (to avoid double-counting).

    Returns
    -------
    int
        Estimated size in bytes.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0  # Avoid counting same object twice
    seen.add(obj_id)

    # Pandas DataFrame - check before numpy arrays since DataFrames have nbytes too
    if hasattr(obj, "memory_usage") and hasattr(obj, "columns"):
        try:
            return int(obj.memory_usage(deep=True).sum())
        except (TypeError, AttributeError):
            # Fall through to other handlers if memory_usage fails
            pass  # noqa: B110

    # Pandas Series - check before numpy arrays since Series have nbytes too
    if (
        hasattr(obj, "memory_usage")
        and hasattr(obj, "index")
        and not hasattr(obj, "columns")
    ):
        try:
            return int(obj.memory_usage(deep=True))
        except (TypeError, AttributeError):
            # Fall through to other handlers if memory_usage fails
            pass  # noqa: B110

    # NumPy arrays - check dtype attribute to distinguish from other objects with nbytes
    if hasattr(obj, "nbytes") and hasattr(obj, "dtype") and hasattr(obj, "shape"):
        return int(obj.nbytes)

    # PyTorch tensors
    if hasattr(obj, "element_size") and hasattr(obj, "numel"):
        return int(obj.element_size() * obj.numel())

    # Scipy sparse matrices (CSR, CSC, COO, etc.)
    if hasattr(obj, "data") and hasattr(obj, "indices") and hasattr(obj, "format"):
        size = obj.data.nbytes + obj.indices.nbytes
        if hasattr(obj, "indptr"):
            size += obj.indptr.nbytes
        return int(size)

    # AnnCollection objects - count obs and internal structures
    # Check for AnnCollection by looking for adatas attribute and n_obs
    if hasattr(obj, "adatas") and hasattr(obj, "n_obs") and hasattr(obj, "n_vars"):
        size = 0
        # obs DataFrame (concatenated cell metadata) - this gets copied per worker
        if hasattr(obj, "obs") and obj.obs is not None:
            size += _deep_sizeof(obj.obs, seen)
        # var DataFrame (shared gene metadata)
        if hasattr(obj, "var") and obj.var is not None:
            size += _deep_sizeof(obj.var, seen)
        # obs_names and var_names (index objects)
        if hasattr(obj, "obs_names") and obj.obs_names is not None:
            size += _deep_sizeof(obj.obs_names, seen)
        if hasattr(obj, "var_names") and obj.var_names is not None:
            size += _deep_sizeof(obj.var_names, seen)
        # Internal adatas list
        for adata in obj.adatas:
            size += _deep_sizeof(adata, seen)
        return int(size)

    # AnnData objects - special handling for accurate estimation
    if hasattr(obj, "X") and hasattr(obj, "obs") and hasattr(obj, "var_names"):
        size = 0
        # X matrix (main data) - for backed mode, this doesn't load data
        if obj.X is not None:
            size += _deep_sizeof(obj.X, seen)
        # obs DataFrame (cell metadata for this sample)
        if hasattr(obj, "obs") and obj.obs is not None:
            size += _deep_sizeof(obj.obs, seen)
        # var DataFrame (gene metadata)
        if hasattr(obj, "var") and obj.var is not None:
            size += _deep_sizeof(obj.var, seen)
        # obsm matrices (e.g., embeddings)
        if hasattr(obj, "obsm") and obj.obsm is not None:
            for key in obj.obsm.keys():
                size += _deep_sizeof(obj.obsm[key], seen)
        # layers (alternative matrices like raw counts)
        if hasattr(obj, "layers") and obj.layers is not None:
            for key in obj.layers.keys():
                size += _deep_sizeof(obj.layers[key], seen)
        return int(size)

    # MultiIndexable from scdataset - has _indexables list and unstructured property
    if hasattr(obj, "_indexables") and hasattr(obj, "unstructured"):
        size = 0
        for indexable in obj._indexables:
            size += _deep_sizeof(indexable, seen)
        # unstructured is typically shared, count once
        if obj.unstructured:
            size += _deep_sizeof(obj.unstructured, seen)
        return int(size)

    # Dictionaries - recursive
    if isinstance(obj, dict):
        size = sys.getsizeof(obj)  # Dict overhead
        for k, v in obj.items():
            size += _deep_sizeof(k, seen)
            size += _deep_sizeof(v, seen)
        return int(size)

    # Lists and tuples - recursive
    if isinstance(obj, (list, tuple)):
        size = sys.getsizeof(obj)  # Container overhead
        for item in obj:
            size += _deep_sizeof(item, seen)
        return int(size)

    # Strings - UTF-8 encoded size is more accurate than sys.getsizeof
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))

    # Bytes
    if isinstance(obj, bytes):
        return len(obj)

    # Default: sys.getsizeof for primitive types and unknown objects
    return sys.getsizeof(obj)


def estimate_sample_size(
    data_collection,
    n_samples: int = 16,
    fetch_callback: Optional[Callable] = None,
    fetch_transform: Optional[Callable] = None,
    batch_callback: Optional[Callable] = None,
    batch_transform: Optional[Callable] = None,
) -> int:
    """
    Estimate the memory size of a single sample from the data collection.

    This function samples a few elements from the data collection and estimates
    the average memory size per sample in bytes using recursive deep size
    estimation. If transforms/callbacks are provided, they are applied to
    simulate the actual memory usage during training.

    Parameters
    ----------
    data_collection : object
        Data collection to estimate sample size from. Must support indexing.
    n_samples : int, default=16
        Number of samples to average over for estimation.
    fetch_callback : Callable, optional
        Custom fetch function. If provided, called as
        ``fetch_callback(data_collection, [index])`` to get the sample.
        This should match the ``fetch_callback`` parameter used with scDataset.
    fetch_transform : Callable, optional
        Transform to apply after fetching data. This should match the
        ``fetch_transform`` parameter used with scDataset. Applied to the
        fetched sample before size estimation.
    batch_callback : Callable, optional
        Custom batch extraction function. If provided, called as
        ``batch_callback(fetched_data, [0])`` to extract a single sample.
        This should match the ``batch_callback`` parameter used with scDataset.
    batch_transform : Callable, optional
        Transform to apply to batches. This should match the
        ``batch_transform`` parameter used with scDataset. Applied after
        fetch_transform.

    Returns
    -------
    int
        Estimated size per sample in bytes.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 2000)  # 1000 samples, 2000 features
    >>> size = estimate_sample_size(data)
    >>> print(f"Estimated sample size: {size} bytes")
    Estimated sample size: 16000 bytes

    For AnnData with fetch transform (not runnable without data):

    .. code-block:: python

        from scdataset import adata_to_mindex
        size = estimate_sample_size(adata_collection, fetch_transform=adata_to_mindex)

    Notes
    -----
    The estimation uses recursive deep size estimation that correctly handles:

    - NumPy arrays (using ``nbytes``)
    - Scipy sparse matrices (CSR, CSC, COO - counting data, indices, indptr)
    - PyTorch tensors (using ``element_size * numel``)
    - Pandas DataFrames/Series (using ``memory_usage(deep=True)``)
    - AnnData objects (counting X, obs, obsm, and layers)
    - Dictionaries and lists (recursively counting all elements)
    - Strings (UTF-8 encoded byte length)

    Shared objects (same ``id()``) are only counted once to avoid
    double-counting.

    When using backed AnnData (``backed='r'``), it's important to provide
    the ``fetch_transform`` parameter to get accurate memory estimates,
    as backed data remains on disk until transformed.
    """
    n_samples = min(n_samples, len(data_collection))
    sizes = []

    # Check if data_collection is a MultiIndexable (uses list indexing for samples)
    is_multiindexable = hasattr(data_collection, "_indexables") and hasattr(
        data_collection, "unstructured"
    )

    for i in range(n_samples):
        # Step 1: Fetch sample (using callback or default indexing)
        if fetch_callback is not None:
            sample = fetch_callback(data_collection, [i])
        elif is_multiindexable:
            # MultiIndexable uses list indexing for samples, not integer indexing
            sample = data_collection[[i]]
        else:
            sample = data_collection[i]

        # Step 2: Apply fetch transform if provided
        if fetch_transform is not None:
            sample = fetch_transform(sample)

        # Step 3: Extract single item via batch_callback if provided
        # (This simulates what happens when extracting a batch of 1)
        if batch_callback is not None:
            sample = batch_callback(sample, [0])

        # Step 4: Apply batch transform if provided
        if batch_transform is not None:
            sample = batch_transform(sample)

        sizes.append(_deep_sizeof(sample))

    return int(np.mean(sizes)) if sizes else 0


def suggest_parameters(
    data_collection,
    batch_size: int,
    target_ram_fraction: float = 0.20,
    max_workers: int = 16,
    min_workers: int = 1,
    verbose: bool = True,
    fetch_callback: Optional[Callable] = None,
    fetch_transform: Optional[Callable] = None,
    batch_callback: Optional[Callable] = None,
    batch_transform: Optional[Callable] = None,
) -> Dict[str, Any]:
    r"""
    Suggest optimal parameters for scDataset based on system resources.

    This function analyzes the data collection and available system resources
    to suggest optimal values for ``num_workers``, ``fetch_factor``, and
    ``block_size`` parameters.

    Parameters
    ----------
    data_collection : object
        The data collection to be used with scDataset.
    batch_size : int
        The batch size you plan to use.
    target_ram_fraction : float, default=0.20
        Maximum fraction of available RAM to use for data loading.
        Default is 20% which leaves room for model and other processes.
    max_workers : int, default=16
        Maximum number of workers to suggest. More than 16 workers
        typically has diminishing returns.
    min_workers : int, default=1
        Minimum number of workers to suggest.
    verbose : bool, default=True
        If True, print detailed suggestions and explanations.
    fetch_callback : Callable, optional
        Custom fetch function. Pass the same function you will use
        with scDataset for accurate memory estimation.
    fetch_transform : Callable, optional
        Transform to apply after fetching data. Pass the same function
        you will use with scDataset for accurate memory estimation.
    batch_callback : Callable, optional
        Custom batch extraction function. Pass the same function you will use
        with scDataset for accurate memory estimation.
    batch_transform : Callable, optional
        Transform to apply to batches. Pass the same function
        you will use with scDataset for accurate memory estimation.

    Returns
    -------
    dict
        Dictionary containing suggested parameters:

        - ``num_workers``: Suggested number of DataLoader workers
        - ``fetch_factor``: Suggested fetch factor for scDataset
        - ``block_size_conservative``: Block size for more randomness (fetch_factor // 2)
        - ``block_size_balanced``: Block size balancing randomness and throughput
        - ``block_size_aggressive``: Block size for maximum throughput (fetch_factor * 2)
        - ``prefetch_factor``: Suggested prefetch_factor for DataLoader
        - ``estimated_memory_per_fetch_mb``: Estimated memory per fetch operation in MB
        - ``system_info``: Dictionary with system information used for calculation

    Examples
    --------
    >>> import numpy as np
    >>> from scdataset import scDataset, BlockShuffling
    >>> from scdataset.experimental import suggest_parameters
    >>> from torch.utils.data import DataLoader
    >>>
    >>> data = np.random.randn(10000, 200)
    >>> params = suggest_parameters(data, batch_size=64, verbose=False)
    >>>
    >>> # Use suggested parameters
    >>> strategy = BlockShuffling(block_size=params['block_size_balanced'])
    >>> dataset = scDataset(
    ...     data, strategy,
    ...     batch_size=64,
    ...     fetch_factor=params['fetch_factor']
    ... )
    >>> loader = DataLoader(
    ...     dataset, batch_size=None,
    ...     num_workers=min(params['num_workers'], 2),  # Limit for example
    ...     prefetch_factor=params['prefetch_factor']
    ... )

    Notes
    -----
    **Worker selection logic:**

    The number of workers is set to ``min(available_cores // 2, max_workers)``.
    Using half the cores leaves resources for the main process and model training.

    **Fetch factor selection logic:**

    The fetch factor is chosen such that the total data loaded by all workers
    does not exceed ``target_ram_fraction`` of available RAM. The calculation
    accounts for prefetching (prefetch_factor = fetch_factor + 1), which
    effectively doubles memory usage since both the current and prefetched
    data are in memory simultaneously:

    .. math::

        2 \\times batch\\_size \\times fetch\\_factor \\times num\\_workers \\times sample\\_size < target\\_ram\\_fraction \\times RAM

    The factor of 2 accounts for the prefetch buffer in the DataLoader.

    **Block size recommendations:**

    - ``block_size_conservative`` (fetch_factor // 2): More randomness, slightly
      lower throughput. Good for training where randomization is important.
    - ``block_size_balanced`` (fetch_factor): Balanced randomness and throughput.
    - ``block_size_aggressive`` (fetch_factor * 2): Higher throughput, less
      randomness.

    Block sizes smaller than ``fetch_factor // 2`` or larger than ``fetch_factor * 2``
    have diminishing returns.

    Raises
    ------
    ImportError
        If psutil is not installed (optional dependency).

    Warns
    -----
    UserWarning
        If psutil is not available, uses conservative defaults.
    """
    result = {}
    system_info = {}

    # Try to get system information
    try:
        import psutil

        available_ram = psutil.virtual_memory().available
        total_ram = psutil.virtual_memory().total
        cpu_count = os.cpu_count() or 4
        system_info["available_ram_gb"] = available_ram / (1024**3)
        system_info["total_ram_gb"] = total_ram / (1024**3)
        system_info["cpu_count"] = cpu_count
        has_psutil = True
    except ImportError:
        warnings.warn(
            "psutil not installed. Using conservative defaults. "
            "Install psutil for better parameter suggestions: pip install psutil",
            stacklevel=2,
        )
        # Conservative defaults
        available_ram = 8 * 1024**3  # Assume 8GB available
        total_ram = 16 * 1024**3  # Assume 16GB total
        cpu_count = 4
        system_info["available_ram_gb"] = "unknown (psutil not installed)"
        system_info["total_ram_gb"] = "unknown (psutil not installed)"
        system_info["cpu_count"] = cpu_count
        has_psutil = False

    # Calculate num_workers
    num_workers = min(max(cpu_count // 2, min_workers), max_workers)
    result["num_workers"] = num_workers

    # Estimate sample size (applying transforms/callbacks for accurate estimation)
    sample_size = estimate_sample_size(
        data_collection,
        fetch_transform=fetch_transform,
        batch_transform=batch_transform,
        fetch_callback=fetch_callback,
        batch_callback=batch_callback,
    )
    system_info["estimated_sample_size_bytes"] = sample_size

    # Calculate maximum fetch_factor based on RAM constraint
    # Formula: 2 * batch_size * fetch_factor * num_workers * sample_size < target_ram_fraction * available_ram
    # The factor of 2 accounts for prefetch_factor = fetch_factor + 1 (prefetch buffer doubles memory)
    target_ram = target_ram_fraction * available_ram

    if sample_size > 0 and batch_size > 0 and num_workers > 0:
        # Account for prefetch doubling memory (factor of 2)
        max_fetch_factor = int(
            target_ram / (2 * batch_size * num_workers * sample_size)
        )
        # Clamp to reasonable range
        fetch_factor = max(1, min(max_fetch_factor, 256))
    else:
        fetch_factor = 8  # Default fallback

    result["fetch_factor"] = fetch_factor

    # Calculate block sizes
    result["block_size_conservative"] = max(1, fetch_factor // 2)
    result["block_size_balanced"] = max(1, fetch_factor)
    result["block_size_aggressive"] = max(1, fetch_factor * 2)

    # Prefetch factor should be fetch_factor + 1 for optimal performance
    result["prefetch_factor"] = fetch_factor + 1

    # Calculate estimated memory usage (includes prefetch buffer - hence * 2)
    memory_per_fetch = batch_size * fetch_factor * sample_size
    memory_total = memory_per_fetch * num_workers * 2  # * 2 for prefetch buffer
    result["estimated_memory_per_fetch_mb"] = memory_per_fetch / (1024**2)
    result["estimated_total_memory_mb"] = memory_total / (1024**2)

    result["system_info"] = system_info

    if verbose:
        print("=" * 60)
        print("scDataset Parameter Suggestions")
        print("=" * 60)
        print()
        print("System Information:")
        if has_psutil:
            print(f"  Available RAM: {system_info['available_ram_gb']:.1f} GB")
            print(f"  Total RAM: {system_info['total_ram_gb']:.1f} GB")
        else:
            print("  RAM info: Not available (install psutil)")
        print(f"  CPU cores: {system_info['cpu_count']}")
        print(
            f"  Estimated sample size: {sample_size:,} bytes ({sample_size/1024:.1f} KB)"
        )
        print()
        print("Suggested Parameters:")
        print(f"  num_workers: {num_workers}")
        print(f"  fetch_factor: {fetch_factor}")
        print(f"  prefetch_factor: {result['prefetch_factor']}")
        print()
        print("Block Size Options (choose based on your needs):")
        print(f"  block_size_conservative: {result['block_size_conservative']}")
        print("    └─ More randomness, good for training")
        print(f"  block_size_balanced: {result['block_size_balanced']}")
        print("    └─ Balanced randomness and throughput (recommended)")
        print(f"  block_size_aggressive: {result['block_size_aggressive']}")
        print("    └─ Maximum throughput, less randomness")
        print()
        print("Memory Estimates (includes prefetch buffer):")
        print(f"  Per fetch: {result['estimated_memory_per_fetch_mb']:.1f} MB")
        print(
            f"  Total (all workers + prefetch): {result['estimated_total_memory_mb']:.1f} MB"
        )
        print(f"  Target RAM usage: {target_ram_fraction*100:.0f}% of available")
        print()
        print("Tips:")
        print("  • block_size = fetch_factor is optimal (recommended)")
        print("  • block_size < fetch_factor/2: diminishing returns on randomness")
        print("  • block_size > fetch_factor*2: diminishing returns on throughput")
        print("  • Increase fetch_factor if I/O is the bottleneck")
        print("  • Decrease num_workers if memory is constrained")
        print("=" * 60)

    return result
