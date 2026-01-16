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
from typing import Dict, Any

import numpy as np


def estimate_sample_size(data_collection, n_samples: int = 16) -> int:
    """
    Estimate the memory size of a single sample from the data collection.
    
    This function samples a few elements from the data collection and estimates
    the average memory size per sample in bytes.
    
    Parameters
    ----------
    data_collection : object
        Data collection to estimate sample size from. Must support indexing.
    n_samples : int, default=16
        Number of samples to average over for estimation.
        
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
    
    Notes
    -----
    The estimation uses ``sys.getsizeof`` for basic Python objects and
    ``nbytes`` for numpy arrays. For complex objects like AnnData,
    the estimate may be approximate.
    """
    n_samples = min(n_samples, len(data_collection))
    sizes = []
    
    for i in range(n_samples):
        sample = data_collection[i]
        
        # Handle different sample types
        if hasattr(sample, 'nbytes'):
            # NumPy array
            sizes.append(sample.nbytes)
        elif hasattr(sample, 'X'):
            # AnnData-like object
            X = sample.X
            if hasattr(X, 'nbytes'):
                sizes.append(X.nbytes)
            elif hasattr(X, 'data'):  # Sparse matrix
                sizes.append(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes)
            else:
                sizes.append(sys.getsizeof(X))
        elif isinstance(sample, dict):
            # Dictionary (e.g., HuggingFace format)
            total = 0
            for v in sample.values():
                if hasattr(v, 'nbytes'):
                    total += v.nbytes
                else:
                    total += sys.getsizeof(v)
            sizes.append(total)
        else:
            sizes.append(sys.getsizeof(sample))
    
    return int(np.mean(sizes)) if sizes else 0


def suggest_parameters(
    data_collection,
    batch_size: int,
    target_ram_fraction: float = 0.25,
    max_workers: int = 16,
    min_workers: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
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
    >>> from scdataset.auto_config import suggest_parameters
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
    - ``block_size_aggressive`` (fetch_factor * 2): Maximum throughput, less
      randomness. Good for validation/inference or when data is already shuffled.
    
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
        system_info['available_ram_gb'] = available_ram / (1024**3)
        system_info['total_ram_gb'] = total_ram / (1024**3)
        system_info['cpu_count'] = cpu_count
        has_psutil = True
    except ImportError:
        warnings.warn(
            "psutil not installed. Using conservative defaults. "
            "Install psutil for better parameter suggestions: pip install psutil"
        )
        # Conservative defaults
        available_ram = 8 * 1024**3  # Assume 8GB available
        total_ram = 16 * 1024**3     # Assume 16GB total
        cpu_count = 4
        system_info['available_ram_gb'] = 'unknown (psutil not installed)'
        system_info['total_ram_gb'] = 'unknown (psutil not installed)'
        system_info['cpu_count'] = cpu_count
        has_psutil = False
    
    # Calculate num_workers
    num_workers = min(max(cpu_count // 2, min_workers), max_workers)
    result['num_workers'] = num_workers
    
    # Estimate sample size
    sample_size = estimate_sample_size(data_collection)
    system_info['estimated_sample_size_bytes'] = sample_size
    
    # Calculate maximum fetch_factor based on RAM constraint
    # Formula: 2 * batch_size * fetch_factor * num_workers * sample_size < target_ram_fraction * available_ram
    # The factor of 2 accounts for prefetch_factor = fetch_factor + 1 (prefetch buffer doubles memory)
    target_ram = target_ram_fraction * available_ram
    
    if sample_size > 0 and batch_size > 0 and num_workers > 0:
        # Account for prefetch doubling memory (factor of 2)
        max_fetch_factor = int(target_ram / (2 * batch_size * num_workers * sample_size))
        # Clamp to reasonable range
        fetch_factor = max(1, min(max_fetch_factor, 256))
    else:
        fetch_factor = 8  # Default fallback
    
    result['fetch_factor'] = fetch_factor
    
    # Calculate block sizes
    result['block_size_conservative'] = max(1, fetch_factor // 2)
    result['block_size_balanced'] = max(1, fetch_factor)
    result['block_size_aggressive'] = max(1, fetch_factor * 2)
    
    # Prefetch factor should be fetch_factor + 1 for optimal performance
    result['prefetch_factor'] = fetch_factor + 1
    
    # Calculate estimated memory usage (includes prefetch buffer - hence * 2)
    memory_per_fetch = batch_size * fetch_factor * sample_size
    memory_total = memory_per_fetch * num_workers * 2  # * 2 for prefetch buffer
    result['estimated_memory_per_fetch_mb'] = memory_per_fetch / (1024**2)
    result['estimated_total_memory_mb'] = memory_total / (1024**2)
    
    result['system_info'] = system_info
    
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
        print(f"  Estimated sample size: {sample_size:,} bytes ({sample_size/1024:.1f} KB)")
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
        print(f"  Total (all workers + prefetch): {result['estimated_total_memory_mb']:.1f} MB")
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
