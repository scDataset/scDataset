"""
Utility functions for scDataset benchmarks.

This module provides utility functions for transforming and loading data
from various sources when benchmarking scDataset.

Transforms are imported from scdataset.transforms for consistency.

Utilities
---------
load_config : Load benchmark configuration from YAML
evaluate_loader : Evaluate data loader performance
save_results_to_csv : Save benchmark results to CSV
"""

import gc
import os
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

# Import transforms from scdataset - these are the canonical implementations
try:
    from scdataset.transforms import (
        adata_to_mindex,
        hf_tahoe_to_tensor,
        bionemo_to_tensor,
    )
    from scdataset import MultiIndexable
except ImportError:
    from scdataset.transforms import (
        adata_to_mindex,
        hf_tahoe_to_tensor,
        bionemo_to_tensor,
    )
    from scdataset.multiindexable import MultiIndexable


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
    loader,
    test_time_seconds: int = 120,
    description: str = "Testing loader",
    warm_up_seconds: int = 30,
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
    warm_up_seconds : int, default=30
        Duration of warm-up period in seconds. Set to 0 to skip warm-up.

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
    The function includes a warm-up period before measuring
    performance to allow for JIT compilation and cache warming.
    Set warm_up_seconds=0 to skip warm-up.

    Entropy calculation is performed if batches have plate information,
    either via `.obs['plate']` (AnnData) or `batch['plate']` (scDataset).
    """
    gc.collect()

    total_samples = 0
    batch_plates = []

    pbar = tqdm(desc=f"{description} (for {test_time_seconds}s)")

    # Initialize timing - handle both warm-up and no warm-up cases
    warm_up_start = time.perf_counter()
    warm_up_end = warm_up_start + warm_up_seconds
    is_warming_up = warm_up_seconds > 0

    # Initialize start_time early to handle edge cases where data exhausts during warm-up
    start_time = warm_up_start
    end_time = start_time + test_time_seconds

    if not is_warming_up:
        # No warm-up, start timing immediately
        start_time = time.perf_counter()
        end_time = start_time + test_time_seconds

    for i, batch in enumerate(loader):
        # Handle different batch structures
        plates_data = None

        if hasattr(batch, "X"):
            # AnnCollection batch (from AnnLoader)
            batch_size = batch.X.shape[0]
            # Check for plate column safely - batch.obs may be AnnCollectionObs
            try:
                if hasattr(batch, "obs") and hasattr(batch.obs, "columns") and "plate" in batch.obs.columns:
                    plates_data = batch.obs["plate"].values
            except (KeyError, AttributeError, TypeError):
                plates_data = None
        elif hasattr(batch, "__getitem__"):
            # Dict-like batch (dict or MultiIndexable from scDataset with adata_to_mindex)
            try:
                X = batch["X"]
                batch_size = X.shape[0] if hasattr(X, "shape") else len(X)
                try:
                    plates_data = batch["plate"]
                    if hasattr(plates_data, "values"):
                        plates_data = plates_data.values
                except (KeyError, IndexError, TypeError):
                    plates_data = None
            except (KeyError, IndexError, TypeError):
                # Not a dict-like with X, try shape attribute
                batch_size = batch.shape[0] if hasattr(batch, "shape") else len(batch)
        else:
            batch_size = batch.shape[0] if hasattr(batch, "shape") else len(batch)

        if not is_warming_up and plates_data is not None:
            # Collect plate info for entropy calculation
            batch_plates.append(plates_data)

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
