# Define dataset paths
HUGGINGFACE_DATASET_PATH = "tahoebio/Tahoe-100m"
H5AD_FILES_PATH = "/path-to-data/2025-02-25/original_h5ad"
BIONEMO_DATA_PATH = "/path-to-data/2025-02-25/scdl_out"
import os
os.environ["HF_HOME"] = "/path-to-data/.cache/huggingface"

import time
import pandas as pd
import numpy as np
import anndata as ad
import gc
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
from anndata.experimental import AnnLoader, AnnCollection
from tqdm.auto import tqdm
from scdataset import scDataset
from scipy import stats
from datasets import load_dataset
from typing import Union, Sequence
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch

def fetch_transform_hf(batch, num_genes=62713):
    if isinstance(batch, dict):
        # Extract numpy arrays from batch
        batch_genes = batch['genes']  # List of numpy arrays
        batch_expr = batch['expressions']  # List of numpy arrays
    elif isinstance(batch, list):
        # Extract numpy arrays from batch
        batch_genes = [item['genes'] for item in batch]
        batch_expr = [item['expressions'] for item in batch]
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

def fetch_transform_adata(batch):
    return batch.to_adata()

def fetch_callback_bionemo(self, idx: Union[int, slice, Sequence[int], np.ndarray, torch.Tensor]) -> torch.Tensor:
    """Fetch callback for bionemo dataset when used with scDataset."""
    if isinstance(idx, int):
        # Single index
        return collate_sparse_matrix_batch([self.__getitem__(idx)]).to_dense()
    elif isinstance(idx, slice):
        # Slice: convert to a list of indices
        indices = list(range(*idx.indices(len(self))))
        batch_tensors = [self.__getitem__(i) for i in indices]
        return collate_sparse_matrix_batch(batch_tensors).to_dense()
    elif isinstance(idx, (list, np.ndarray, torch.Tensor)):
        # Batch indexing
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        batch_tensors = [self.__getitem__(int(i)) for i in idx]
        return collate_sparse_matrix_batch(batch_tensors).to_dense()
    else:
        raise TypeError(f"Unsupported index type: {type(idx)}")

def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
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
            "test_modes": "all",  # Options: "all", "random", "stream"
        }

def evaluate_loader(loader, test_time_seconds=120, description="Testing loader"):
    """Evaluate the performance of a data loader for a specified duration."""
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
                batch_plates.append(batch.obs['plate'].values)
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
                pbar.set_description(f"{description} (warming up complete, testing for {test_time_seconds}s)")
            else:
                pbar.set_description(f"{description} (warming up: {current_time - warm_up_start:.1f}/{warm_up_seconds}s)")
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

def save_results_to_csv(results, filepath=None):
    """Save or update results to CSV file."""
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    if filepath is not None:
        df.to_csv(filepath, index=False)
        print(f"Updated results saved to {filepath}")
    
    return df

def run_evaluations(config_path):
    # Load configuration
    config = load_config(config_path)
    
    # Get collection type from config
    collection_type = config.get("collection_type", "anncollection")
    print(f"Using collection type: {collection_type}")
    
    # Get test mode configuration
    test_modes = config.get("test_modes", "all").lower()
    print(f"Testing modes: {test_modes}")
    
    # Load the appropriate data source based on collection_type
    if collection_type == "anncollection":
        # Load the anndata files
        print("Loading AnnData files...")
        adata_paths = config.get(
            "adata_paths",
            [f'{H5AD_FILES_PATH}/plate{i}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad' for i in range(1, 15)]
        )
        
        # Read AnnData files in read-only mode
        adatas = [ad.read_h5ad(path, backed='r') for path in adata_paths]
        
        for adata in adatas:
            adata.obs = adata.obs[['plate']]
        
        # Create AnnCollection
        data_collection = AnnCollection(adatas)
        print(f"Created AnnCollection with {data_collection.n_obs} observations and {data_collection.n_vars} variables.")
    
    elif collection_type == "huggingface":
        print("Loading Hugging Face dataset...")
        ds = load_dataset(HUGGINGFACE_DATASET_PATH, split="train")
        data_collection = ds.with_format("numpy", columns=["genes", "expressions"])
        print(f"Loaded Hugging Face dataset with {len(data_collection)} samples.")
    
    elif collection_type == "bionemo":
        print("Loading Bionemo SingleCellMemMapDataset...")
        bionemo_data_path = config.get("bionemo_data_path", BIONEMO_DATA_PATH)
        data_collection = SingleCellMemMapDataset(
            data_path=bionemo_data_path,
        )
        print(f"Loaded Bionemo dataset with {len(data_collection)} samples.")
    
    else:
        raise ValueError(f"Unsupported collection type: {collection_type}")
    
    # Get parameters from config
    batch_sizes = config.get("batch_sizes", [16, 32, 64, 128, 256])
    block_sizes = config.get("block_sizes", [1, 2, 4, 8, 16, 32, 64, 128])
    fetch_factors = config.get("fetch_factors", [1, 2, 4, 8, 16])
    num_workers_options = config.get("num_workers_options", [0, 1, 2, 4, 8, 16])
    results_path = config.get("results_path", "/path-to-data/data_loader_performance.csv")
    
    # Set up results storage
    results = []
    
    # Only run AnnLoader tests with AnnCollection
    if collection_type == "anncollection":
        # Test AnnLoader in stream mode
        if test_modes in ["all", "stream"]:
            print("\nTesting AnnLoader in stream mode (shuffle=False, drop_last=False)...")
            for batch_size in batch_sizes:
                loader = AnnLoader(data_collection, batch_size=batch_size, shuffle=False, drop_last=False)
                
                desc = f"AnnLoader (stream) - batch_size={batch_size}"
                result = evaluate_loader(loader, description=desc)
                
                results.append({
                    "mode": "stream",
                    "loader": "AnnLoader",
                    "collection_type": collection_type,
                    "batch_size": batch_size,
                    "block_size": None,
                    "fetch_factor": None,
                    "num_workers": None,
                    "prefetch_factor": None,
                    **result
                })
                
                # Save results after each experiment
                save_results_to_csv(results, results_path)
                
                del loader
                gc.collect()
            
        # Test AnnLoader in random sampling mode
        if test_modes in ["all", "random"]:
            print("\nTesting AnnLoader in random sampling mode (shuffle=True, drop_last=True)...")
            for batch_size in batch_sizes:
                loader = AnnLoader(data_collection, batch_size=batch_size, shuffle=True, drop_last=True)
                
                desc = f"AnnLoader (random) - batch_size={batch_size}"
                result = evaluate_loader(loader, description=desc)
                
                results.append({
                    "mode": "random",
                    "loader": "AnnLoader",
                    "collection_type": collection_type,
                    "batch_size": batch_size,
                    "block_size": None,
                    "fetch_factor": None,
                    "num_workers": None,
                    "prefetch_factor": None,
                    **result
                })
                
                # Save results after each experiment
                save_results_to_csv(results, results_path)
                
                del loader
                gc.collect()
    
    # For Hugging Face datasets, use PyTorch DataLoader directly
    elif collection_type == "huggingface":
        # Test PyTorch DataLoader in stream mode (shuffle=False)
        if test_modes in ["all", "stream"]:
            print("\nTesting PyTorch DataLoader in stream mode (shuffle=False)...")
            for batch_size in batch_sizes:
                for num_workers in num_workers_options:
                    loader = DataLoader(
                        data_collection,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=fetch_transform_hf,
                    )
                    
                    desc = f"HuggingFace (stream) - batch_size={batch_size}, w={num_workers}"
                    result = evaluate_loader(loader, description=desc)
                    
                    results.append({
                        "mode": "stream",
                        "loader": "HuggingFace",
                        "collection_type": collection_type,
                        "batch_size": batch_size,
                        "block_size": None,
                        "fetch_factor": None,
                        "num_workers": num_workers,
                        "prefetch_factor": None,
                        **result
                    })
                    
                    # Save results after each experiment
                    save_results_to_csv(results, results_path)
                    
                    del loader
                    gc.collect()
        
        # Test PyTorch DataLoader in random sampling mode (shuffle=True)
        if test_modes in ["all", "random"]:
            print("\nTesting PyTorch DataLoader in random sampling mode (shuffle=True)...")
            for batch_size in batch_sizes:
                for num_workers in num_workers_options:
                    loader = DataLoader(
                        data_collection,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=fetch_transform_hf,
                    )
                    
                    desc = f"PyTorch DataLoader (random) - batch_size={batch_size}, w={num_workers}"
                    result = evaluate_loader(loader, description=desc)
                    
                    results.append({
                        "mode": "random",
                        "loader": "HuggingFace",
                        "collection_type": collection_type,
                        "batch_size": batch_size,
                        "block_size": None,
                        "fetch_factor": None,
                        "num_workers": num_workers,
                        "prefetch_factor": None,
                        **result
                    })
                    
                    # Save results after each experiment
                    save_results_to_csv(results, results_path)
                    
                    del loader
                    gc.collect()
    
    # For BioNeMo dataset, use PyTorch DataLoader with collate_sparse_matrix_batch
    elif collection_type == "bionemo":
        # Test PyTorch DataLoader in stream mode (shuffle=False)
        if test_modes in ["all", "stream"]:
            print("\nTesting PyTorch DataLoader with BioNeMo in stream mode (shuffle=False)...")
            for batch_size in batch_sizes:
                for num_workers in num_workers_options:
                    loader = DataLoader(
                        data_collection,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=collate_sparse_matrix_batch
                    )
                    
                    desc = f"PyTorch DataLoader with BioNeMo (stream) - batch_size={batch_size}, w={num_workers}"
                    result = evaluate_loader(loader, description=desc)
                    
                    results.append({
                        "mode": "stream",
                        "loader": "BioNeMo",
                        "collection_type": collection_type,
                        "batch_size": batch_size,
                        "block_size": None,
                        "fetch_factor": None,
                        "num_workers": num_workers,
                        "prefetch_factor": None,
                        **result
                    })
                    
                    # Save results after each experiment
                    save_results_to_csv(results, results_path)
                    
                    del loader
                    gc.collect()
        
        # Test PyTorch DataLoader in random sampling mode (shuffle=True)
        if test_modes in ["all", "random"]:
            print("\nTesting PyTorch DataLoader with BioNeMo in random sampling mode (shuffle=True)...")
            for batch_size in batch_sizes:
                for num_workers in num_workers_options:
                    loader = DataLoader(
                        data_collection,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=collate_sparse_matrix_batch
                    )
                    
                    desc = f"PyTorch DataLoader with BioNeMo (random) - batch_size={batch_size}, w={num_workers}"
                    result = evaluate_loader(loader, description=desc)
                    
                    results.append({
                        "mode": "random",
                        "loader": "BioNeMo",
                        "collection_type": collection_type,
                        "batch_size": batch_size,
                        "block_size": None,
                        "fetch_factor": None,
                        "num_workers": num_workers,
                        "prefetch_factor": None,
                        **result
                    })
                    
                    # Save results after each experiment
                    save_results_to_csv(results, results_path)
                    
                    del loader
                    gc.collect()
    
    # Test scDataset in train mode (random)
    if test_modes in ["all", "random"]:
        print("\nTesting scDataset in train mode (random)...")
        for batch_size in batch_sizes:
            for block_size in block_sizes:
                for fetch_factor in fetch_factors:
                    for num_workers in num_workers_options:
                        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None
                        
                        # Choose appropriate fetch transform based on collection type
                        if collection_type == "huggingface":
                            fetch_transform = fetch_transform_hf
                        elif collection_type == "anncollection":
                            fetch_transform = fetch_transform_adata
                        elif collection_type == "bionemo":
                            fetch_transform = None
                        
                        extra_params = {}
                        if collection_type == "bionemo":
                            extra_params["fetch_callback"] = fetch_callback_bionemo
                        
                        dataset = scDataset(
                            data_collection=data_collection,
                            batch_size=batch_size,
                            block_size=block_size,
                            fetch_factor=fetch_factor,
                            fetch_transform=fetch_transform,
                            **extra_params
                        )
                        
                        # Set to train mode for random access
                        dataset.train_mode()
                        
                        loader = DataLoader(
                            dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor,
                        )
                        
                        desc = f"scDataset (random) - batch={batch_size}, block={block_size}, ff={fetch_factor}, w={num_workers}"
                        result = evaluate_loader(loader, description=desc)
                        
                        results.append({
                            "mode": "random",
                            "loader": "scDataset",
                            "collection_type": collection_type,
                            "batch_size": batch_size,
                            "block_size": block_size,
                            "fetch_factor": fetch_factor,
                            "num_workers": num_workers,
                            "prefetch_factor": prefetch_factor,
                            **result
                        })
                        
                        # Save results after each experiment
                        save_results_to_csv(results, results_path)
                        
                        del dataset, loader
                        gc.collect()
    
    # Test scDataset in eval mode (stream)
    if test_modes in ["all", "stream"]:
        print("\nTesting scDataset in eval mode (stream)...")
        for batch_size in batch_sizes:
            for block_size in block_sizes:
                for fetch_factor in fetch_factors:
                    for num_workers in num_workers_options:
                        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None
                        
                        dataset = scDataset(
                            data_collection=data_collection,
                            batch_size=batch_size,
                            block_size=block_size,
                            fetch_factor=fetch_factor,
                            fetch_transform=fetch_transform_hf if collection_type == "huggingface" else fetch_transform_adata,
                        )
                        
                        # Set to eval mode for streaming
                        dataset.eval_mode()
                        
                        loader = DataLoader(
                            dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor,
                        )
                        
                        desc = f"scDataset (stream) - batch={batch_size}, block={block_size}, ff={fetch_factor}, w={num_workers}"
                        result = evaluate_loader(loader, description=desc)
                        
                        results.append({
                            "mode": "stream",
                            "loader": "scDataset",  # Fixed typo in loader name
                            "collection_type": collection_type,
                            "batch_size": batch_size,
                            "block_size": block_size,
                            "fetch_factor": fetch_factor,
                            "num_workers": num_workers,
                            "prefetch_factor": prefetch_factor,
                            **result
                        })
                        
                        # Save results after each experiment
                        save_results_to_csv(results, results_path)
                        
                        del dataset, loader
                        gc.collect()
    
    # Load final results for display and visualization
    df = pd.read_csv(results_path)
    
    # Display and save results
    print("\n===== PERFORMANCE RESULTS =====")
    
    # Print results
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print(df[["mode", "loader", "collection_type", "batch_size", "block_size", 
                     "fetch_factor", "num_workers", "samples_per_second", "elapsed"]])
    
    return df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate AnnData loaders performance.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to configuration YAML file")
    args = parser.parse_args()
    
    print(f"Starting data loader performance evaluation using config: {args.config}")
    result_df = run_evaluations(args.config)