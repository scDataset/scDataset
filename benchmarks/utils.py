"""
Utility functions for scDataset benchmarks.

This module provides utility functions for transforming and loading data
from various sources when benchmarking scDataset. These transforms can also
serve as reference implementations for users working with similar data formats.

Transforms
----------
fetch_transform_hf : Transform for HuggingFace sparse gene expression data
fetch_transform_adata : Transform for AnnData/AnnCollection with MultiIndexable output
fetch_callback_bionemo : Fetch callback for BioNeMo SingleCellMemMapDataset

Utilities
---------
load_config : Load benchmark configuration from YAML
evaluate_loader : Evaluate data loader performance
save_results_to_csv : Save benchmark results to CSV
"""

import gc
import os

# Import MultiIndexable for fetch_transform_adata
import sys
import time
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from tqdm.auto import tqdm

# Add the src folder to path for development imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    from scdataset import MultiIndexable
except ImportError:
    from scdataset.multiindexable import MultiIndexable


def fetch_transform_hf(batch, num_genes: int = 62713):
    """
    Transform HuggingFace sparse gene expression data to dense tensors.

    This transform converts sparse gene expression data stored in HuggingFace
    format (with separate 'genes' and 'expressions' arrays) into dense PyTorch
    tensors suitable for model training.

    Parameters
    ----------
    batch : dict or list
        Batch of data from HuggingFace dataset. Can be:
        - dict with 'genes' and 'expressions' keys (list of arrays)
        - list of dicts, each with 'genes' and 'expressions' keys
    num_genes : int, default=62713
        Total number of genes (dimension of output tensor).
        Default is the Tahoe-100M gene count.

    Returns
    -------
    torch.Tensor
        Dense tensor of shape (batch_size, num_genes) with gene expression values.

    Examples
    --------
    >>> # With scDataset
    >>> from scdataset import scDataset, BlockShuffling
    >>> dataset = scDataset(
    ...     hf_dataset,
    ...     BlockShuffling(),
    ...     batch_size=64,
    ...     fetch_transform=fetch_transform_hf
    ... )

    Notes
    -----
    This transform is specifically designed for datasets like Tahoe-100M that
    store sparse gene expression data in HuggingFace Datasets format, where
    each sample has variable-length arrays of gene indices and their expression
    values.

    The transform efficiently converts the sparse representation to dense
    tensors using numpy operations before converting to PyTorch, which is
    faster than building sparse PyTorch tensors directly.
    """
    if isinstance(batch, dict):
        # Extract numpy arrays from batch
        batch_genes = batch["genes"]  # List of numpy arrays
        batch_expr = batch["expressions"]  # List of numpy arrays
    elif isinstance(batch, list):
        # Extract numpy arrays from batch
        batch_genes = [item["genes"] for item in batch]
        batch_expr = [item["expressions"] for item in batch]
    else:
        raise ValueError("Batch must be a dictionary or a list of dictionaries.")

    batch_size = len(batch_genes)

    # Generate batch indices using numpy
    lengths = [len(arr) for arr in batch_genes]
    batch_indices_np = np.concatenate(
        [np.full(l, i, dtype=np.int64) for i, l in enumerate(lengths)]
    )

    # Concatenate all genes and expressions in numpy first
    gene_indices_np = np.concatenate(batch_genes)
    values_np = np.concatenate(batch_expr)

    # Single conversion to tensors
    batch_indices = torch.from_numpy(batch_indices_np)
    gene_indices = torch.from_numpy(gene_indices_np)
    values = torch.from_numpy(values_np).float()

    # Create combined indices tensor
    indices = torch.stack([batch_indices, gene_indices], dim=0)

    # Create dense tensor in one assignment
    dense_tensor = torch.zeros(batch_size, num_genes, dtype=values.dtype)
    dense_tensor[indices[0], indices[1]] = values

    return dense_tensor


def fetch_transform_adata(batch, columns: Optional[List[str]] = None):
    """
    Transform AnnData/AnnCollection batch to MultiIndexable with optional obs columns.

    This transform converts a batch from an AnnCollection (or backed AnnData)
    into a MultiIndexable object containing the expression matrix and optionally
    selected observation columns. The MultiIndexable can then be indexed in
    subsequent batch operations.

    Parameters
    ----------
    batch : AnnData-like
        Batch from AnnCollection or backed AnnData. Must have:
        - `.to_memory()` method (for AnnCollection)
        - `.X` attribute (expression matrix)
        - `.obs` attribute (observation metadata)
    columns : list of str, optional
        List of observation column names to include in the output.
        If None, only the X matrix is included.

    Returns
    -------
    MultiIndexable
        A MultiIndexable object with:
        - 'X': Dense expression matrix as numpy array
        - Additional keys for each column in `columns` (as numpy arrays)

    Examples
    --------
    >>> # Basic usage - just X matrix
    >>> from scdataset import scDataset, BlockShuffling
    >>> from functools import partial
    >>>
    >>> dataset = scDataset(
    ...     ann_collection,
    ...     BlockShuffling(),
    ...     batch_size=64,
    ...     fetch_transform=fetch_transform_adata
    ... )

    >>> # With observation columns
    >>> fetch_fn = partial(fetch_transform_adata, columns=['cell_type', 'batch'])
    >>> dataset = scDataset(
    ...     ann_collection,
    ...     BlockShuffling(),
    ...     batch_size=64,
    ...     fetch_transform=fetch_fn
    ... )
    >>> for batch in dataset:
    ...     X = batch['X']
    ...     cell_types = batch['cell_type']
    ...     break

    Notes
    -----
    This transform calls `.to_memory()` to materialize the AnnData object,
    which is necessary when working with backed or lazy AnnCollection objects.

    Sparse matrices are automatically converted to dense numpy arrays for
    compatibility with standard indexing operations.

    The returned MultiIndexable preserves any unstructured metadata from
    the original batch if you set it explicitly.
    """
    # Import scipy.sparse locally to avoid hard dependency
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None

    # Materialize the AnnData batch in memory
    batch = batch.to_memory()

    X = batch.X
    # Densify if X is a sparse matrix
    if sp is not None and sp.issparse(X):
        X = X.toarray()

    obs = batch.obs

    # Create dict with X and all obs columns as numpy arrays
    data_dict = {"X": X}

    if columns is not None:
        for col in columns:
            data_dict[col] = obs[col].values

    multi = MultiIndexable(data_dict)

    return multi


def fetch_callback_bionemo(
    data_collection, idx: Union[int, slice, Sequence[int], np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """
    Fetch callback for BioNeMo SingleCellMemMapDataset.

    This callback provides custom indexing logic for BioNeMo's
    SingleCellMemMapDataset, which returns sparse matrices that need
    to be collated and densified for use with scDataset.

    Parameters
    ----------
    data_collection : SingleCellMemMapDataset
        The BioNeMo dataset to fetch from.
    idx : int, slice, sequence, or tensor
        Indices to fetch. Can be:
        - int: Single index
        - slice: Slice object
        - list/ndarray/tensor: Batch of indices

    Returns
    -------
    torch.Tensor
        Dense tensor of shape (batch_size, num_genes) with expression values.

    Examples
    --------
    >>> from scdataset import scDataset, BlockShuffling
    >>> from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
    >>>
    >>> bionemo_data = SingleCellMemMapDataset(data_path='/path/to/data')
    >>> dataset = scDataset(
    ...     bionemo_data,
    ...     BlockShuffling(),
    ...     batch_size=64,
    ...     fetch_callback=fetch_callback_bionemo
    ... )

    Notes
    -----
    This callback requires the bionemo-scdl package to be installed.
    The collate function handles the sparse matrix format used by BioNeMo.
    """
    # Import bionemo collate function locally
    from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch

    if isinstance(idx, int):
        # Single index
        return collate_sparse_matrix_batch(
            [data_collection.__getitem__(idx)]
        ).to_dense()
    elif isinstance(idx, slice):
        # Slice: convert to a list of indices
        indices = list(range(*idx.indices(len(data_collection))))
        batch_tensors = [data_collection.__getitem__(i) for i in indices]
        return collate_sparse_matrix_batch(batch_tensors).to_dense()
    elif isinstance(idx, (list, np.ndarray, torch.Tensor)):
        # Batch indexing
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        batch_tensors = [data_collection.__getitem__(int(i)) for i in idx]
        return collate_sparse_matrix_batch(batch_tensors).to_dense()
    else:
        raise TypeError(f"Unsupported index type: {type(idx)}")


def load_config(config_path: str) -> dict:
    """
    Load benchmark configuration from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Configuration dictionary with benchmark parameters.
        If loading fails, returns default configuration.

    Examples
    --------
    >>> config = load_config('experiments/my_config.yaml')
    >>> batch_sizes = config.get('batch_sizes', [64, 128])

    Notes
    -----
    Default configuration includes:
    - results_path: '/path-to-data/data_loader_performance.csv'
    - batch_sizes: [16, 32, 64, 128, 256]
    - block_sizes: [1, 2, 4, 8, 16, 32, 64, 128]
    - fetch_factors: [1, 2, 4, 8, 16]
    - num_workers_options: [0, 1, 2, 4, 8, 16]
    - collection_type: 'anncollection'
    - test_modes: 'all'
    """
    try:
        with open(config_path) as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading config file: {e}")
        print("Using default configuration...")
        return {
            "results_path": "/path-to-data/data_loader_performance.csv",
            "batch_sizes": [16, 32, 64, 128, 256],
            "block_sizes": [1, 2, 4, 8, 16, 32, 64, 128],
            "fetch_factors": [1, 2, 4, 8, 16],
            "num_workers_options": [0, 1, 2, 4, 8, 16],
            "collection_type": "anncollection",
            "test_modes": "all",
        }


def evaluate_loader(
    loader, test_time_seconds: int = 120, description: str = "Testing loader"
) -> dict:
    """
    Evaluate the performance of a data loader for a specified duration.

    This function benchmarks a data loader by iterating through it for a
    specified amount of time (after a warm-up period) and measuring throughput
    metrics.

    Parameters
    ----------
    loader : iterable
        Data loader to evaluate. Should be a PyTorch DataLoader or similar
        iterable that yields batches.
    test_time_seconds : int, default=120
        Duration of the test in seconds (after warm-up).
    description : str, default="Testing loader"
        Description shown in the progress bar.

    Returns
    -------
    dict
        Dictionary containing:
        - samples_tested: Total number of samples processed
        - elapsed: Total elapsed time in seconds
        - avg_time_per_sample: Average time per sample in seconds
        - samples_per_second: Throughput in samples per second
        - avg_batch_entropy: Average entropy of plate distribution (if available)
        - std_batch_entropy: Std dev of entropy (if available)

    Examples
    --------
    >>> from torch.utils.data import DataLoader
    >>> loader = DataLoader(dataset, batch_size=64)
    >>> results = evaluate_loader(loader, test_time_seconds=60)
    >>> print(f"Throughput: {results['samples_per_second']:.1f} samples/sec")

    Notes
    -----
    The function includes a 30-second warm-up period before measuring
    performance to allow for JIT compilation and cache warming.

    Entropy calculation is only performed if batches have an `.obs['plate']`
    attribute, which is specific to AnnData-based datasets.
    """
    gc.collect()

    total_samples = 0
    batch_plates = []

    pbar = tqdm(desc=f"{description} (for {test_time_seconds}s)")

    # Initialize warm-up timer
    warm_up_seconds = 30
    warm_up_start = time.perf_counter()
    warm_up_end = warm_up_start + warm_up_seconds
    is_warming_up = True

    for i, batch in enumerate(loader):
        # Handle different batch structures
        if hasattr(batch, "X"):
            # AnnCollection batch
            batch_size = batch.X.shape[0]
            if not is_warming_up:
                # Collect plate info for entropy calculation
                batch_plates.append(batch.obs["plate"].values)
        else:
            batch_size = batch.shape[0] if hasattr(batch, "shape") else len(batch)

        current_time = time.perf_counter()

        if is_warming_up:
            # We're in warm-up period
            if current_time >= warm_up_end:
                # Warm-up complete, start the actual timing
                is_warming_up = False
                total_samples = 0
                start_time = time.perf_counter()
                end_time = start_time + test_time_seconds
                pbar.set_description(
                    f"{description} (warming up complete, testing for {test_time_seconds}s)"
                )
            else:
                pbar.set_description(
                    f"{description} (warming up: {current_time - warm_up_start:.1f}/{warm_up_seconds}s)"
                )
                pbar.update(1)
                continue

        # Now we're past the warm-up period
        total_samples += batch_size

        elapsed = current_time - start_time
        pbar.set_postfix(samples=total_samples, elapsed=f"{elapsed:.2f}s")
        pbar.update(1)

        if current_time >= end_time:
            break

    pbar.close()

    # Calculate the load time metrics
    elapsed = time.perf_counter() - start_time
    avg_time_per_sample = elapsed / total_samples if total_samples > 0 else 0
    samples_per_second = total_samples / elapsed if elapsed > 0 else 0

    # Calculate entropy measures (if plate data is available)
    avg_batch_entropy = 0
    std_batch_entropy = 0

    if batch_plates:
        batch_entropies = []
        # Calculate entropy for each batch
        for plates in batch_plates:
            if len(plates) > 1:
                unique_plates, counts = np.unique(plates, return_counts=True)
                probabilities = counts / len(plates)
                batch_entropy = stats.entropy(probabilities, base=2)
                batch_entropies.append(batch_entropy)

        # Calculate average and standard deviation of entropy across all batches
        if batch_entropies:
            avg_batch_entropy = np.mean(batch_entropies)
            std_batch_entropy = np.std(batch_entropies)

    return {
        "samples_tested": total_samples,
        "elapsed": elapsed,
        "avg_time_per_sample": avg_time_per_sample,
        "samples_per_second": samples_per_second,
        "avg_batch_entropy": avg_batch_entropy,
        "std_batch_entropy": std_batch_entropy,
    }


def save_results_to_csv(results: list, filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Save benchmark results to a CSV file.

    Parameters
    ----------
    results : list of dict
        List of result dictionaries from evaluate_loader.
    filepath : str, optional
        Path to save the CSV file. If None, only returns DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the results.

    Examples
    --------
    >>> results = []
    >>> for config in configs:
    ...     result = evaluate_loader(loader)
    ...     result['config'] = config
    ...     results.append(result)
    >>> df = save_results_to_csv(results, 'benchmark_results.csv')
    """
    df = pd.DataFrame(results)

    # Save to CSV
    if filepath is not None:
        df.to_csv(filepath, index=False)
        print(f"Updated results saved to {filepath}")

    return df
