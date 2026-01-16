# Define dataset paths
HUGGINGFACE_DATASET_PATH = "tahoebio/Tahoe-100m"
H5AD_FILES_PATH = "/path-to-data/2025-02-25/original_h5ad"
BIONEMO_DATA_PATH = "/path-to-data/2025-02-25/scdl_out"
import os

os.environ["HF_HOME"] = "/path-to-data/.cache/huggingface"

import argparse
import gc
from functools import partial

import anndata as ad
import pandas as pd
from anndata.experimental import AnnCollection, AnnLoader
from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch
from datasets import load_dataset
from torch.utils.data import DataLoader

# Import utilities from utils module
from utils import (
    evaluate_loader,
    fetch_callback_bionemo,
    fetch_transform_adata,
    fetch_transform_hf,
    load_config,
    save_results_to_csv,
)

from src.scdataset.scdataset import scDataset
from src.scdataset.strategy import BlockShuffling, BlockWeightedSampling, Streaming


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
            [
                f"{H5AD_FILES_PATH}/plate{i}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
                for i in range(1, 15)
            ],
        )

        # Read AnnData files in read-only mode
        adatas = [ad.read_h5ad(path, backed="r") for path in adata_paths]

        for adata in adatas:
            adata.obs = adata.obs[["plate"]]

        # Create AnnCollection
        data_collection = AnnCollection(adatas)
        print(
            f"Created AnnCollection with {data_collection.n_obs} observations and {data_collection.n_vars} variables."
        )

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
    results_path = config.get(
        "results_path", "/path-to-data/data_loader_performance.csv"
    )

    # Set up results storage
    results = []

    # Only run AnnLoader tests with AnnCollection
    if collection_type == "anncollection":
        # Test AnnLoader in stream mode
        if test_modes in ["all", "stream"]:
            print(
                "\nTesting AnnLoader in stream mode (shuffle=False, drop_last=False)..."
            )
            for batch_size in batch_sizes:
                loader = AnnLoader(
                    data_collection,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                )

                desc = f"AnnLoader (stream) - batch_size={batch_size}"
                result = evaluate_loader(loader, description=desc)

                results.append(
                    {
                        "mode": "stream",
                        "loader": "AnnLoader",
                        "collection_type": collection_type,
                        "batch_size": batch_size,
                        "block_size": None,
                        "fetch_factor": None,
                        "num_workers": None,
                        "prefetch_factor": None,
                        **result,
                    }
                )

                # Save results after each experiment
                save_results_to_csv(results, results_path)

                del loader
                gc.collect()

        # Test AnnLoader in random sampling mode
        if test_modes in ["all", "random"]:
            print(
                "\nTesting AnnLoader in random sampling mode (shuffle=True, drop_last=True)..."
            )
            for batch_size in batch_sizes:
                loader = AnnLoader(
                    data_collection, batch_size=batch_size, shuffle=True, drop_last=True
                )

                desc = f"AnnLoader (random) - batch_size={batch_size}"
                result = evaluate_loader(loader, description=desc)

                results.append(
                    {
                        "mode": "random",
                        "loader": "AnnLoader",
                        "collection_type": collection_type,
                        "batch_size": batch_size,
                        "block_size": None,
                        "fetch_factor": None,
                        "num_workers": None,
                        "prefetch_factor": None,
                        **result,
                    }
                )

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

                    results.append(
                        {
                            "mode": "stream",
                            "loader": "HuggingFace",
                            "collection_type": collection_type,
                            "batch_size": batch_size,
                            "block_size": None,
                            "fetch_factor": None,
                            "num_workers": num_workers,
                            "prefetch_factor": None,
                            **result,
                        }
                    )

                    # Save results after each experiment
                    save_results_to_csv(results, results_path)

                    del loader
                    gc.collect()

        # Test PyTorch DataLoader in random sampling mode (shuffle=True)
        if test_modes in ["all", "random"]:
            print(
                "\nTesting PyTorch DataLoader in random sampling mode (shuffle=True)..."
            )
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

                    results.append(
                        {
                            "mode": "random",
                            "loader": "HuggingFace",
                            "collection_type": collection_type,
                            "batch_size": batch_size,
                            "block_size": None,
                            "fetch_factor": None,
                            "num_workers": num_workers,
                            "prefetch_factor": None,
                            **result,
                        }
                    )

                    # Save results after each experiment
                    save_results_to_csv(results, results_path)

                    del loader
                    gc.collect()

    # For BioNeMo dataset, use PyTorch DataLoader with collate_sparse_matrix_batch
    elif collection_type == "bionemo":
        # Test PyTorch DataLoader in stream mode (shuffle=False)
        if test_modes in ["all", "stream"]:
            print(
                "\nTesting PyTorch DataLoader with BioNeMo in stream mode (shuffle=False)..."
            )
            for batch_size in batch_sizes:
                for num_workers in num_workers_options:
                    loader = DataLoader(
                        data_collection,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=collate_sparse_matrix_batch,
                    )

                    desc = f"PyTorch DataLoader with BioNeMo (stream) - batch_size={batch_size}, w={num_workers}"
                    result = evaluate_loader(loader, description=desc)

                    results.append(
                        {
                            "mode": "stream",
                            "loader": "BioNeMo",
                            "collection_type": collection_type,
                            "batch_size": batch_size,
                            "block_size": None,
                            "fetch_factor": None,
                            "num_workers": num_workers,
                            "prefetch_factor": None,
                            **result,
                        }
                    )

                    # Save results after each experiment
                    save_results_to_csv(results, results_path)

                    del loader
                    gc.collect()

        # Test PyTorch DataLoader in random sampling mode (shuffle=True)
        if test_modes in ["all", "random"]:
            print(
                "\nTesting PyTorch DataLoader with BioNeMo in random sampling mode (shuffle=True)..."
            )
            for batch_size in batch_sizes:
                for num_workers in num_workers_options:
                    loader = DataLoader(
                        data_collection,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        collate_fn=collate_sparse_matrix_batch,
                    )

                    desc = f"PyTorch DataLoader with BioNeMo (random) - batch_size={batch_size}, w={num_workers}"
                    result = evaluate_loader(loader, description=desc)

                    results.append(
                        {
                            "mode": "random",
                            "loader": "BioNeMo",
                            "collection_type": collection_type,
                            "batch_size": batch_size,
                            "block_size": None,
                            "fetch_factor": None,
                            "num_workers": num_workers,
                            "prefetch_factor": None,
                            **result,
                        }
                    )

                    # Save results after each experiment
                    save_results_to_csv(results, results_path)

                    del loader
                    gc.collect()

    # Test scDataset in random mode using BlockShuffling strategy
    if test_modes in ["all", "random"]:
        print("\nTesting scDataset with BlockShuffling strategy (random mode)...")
        for batch_size in batch_sizes:
            for block_size in block_sizes:
                for fetch_factor in fetch_factors:
                    for num_workers in num_workers_options:
                        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None

                        # Choose appropriate fetch transform based on collection type
                        if collection_type == "huggingface":
                            fetch_transform = fetch_transform_hf
                        elif collection_type == "anncollection":
                            # Use new fetch_transform_adata with plate column
                            fetch_transform = partial(
                                fetch_transform_adata, columns=["plate"]
                            )
                        elif collection_type == "bionemo":
                            fetch_transform = None

                        extra_params = {}
                        if collection_type == "bionemo":
                            extra_params["fetch_callback"] = fetch_callback_bionemo

                        # Create BlockShuffling strategy
                        strategy = BlockShuffling(block_size=block_size)

                        dataset = scDataset(
                            data_collection=data_collection,
                            strategy=strategy,
                            batch_size=batch_size,
                            fetch_factor=fetch_factor,
                            fetch_transform=fetch_transform,
                            **extra_params,
                        )

                        loader = DataLoader(
                            dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor,
                        )

                        desc = f"scDataset (BlockShuffling) - batch={batch_size}, block={block_size}, ff={fetch_factor}, w={num_workers}"
                        result = evaluate_loader(loader, description=desc)

                        results.append(
                            {
                                "mode": "random",
                                "loader": "scDataset",
                                "strategy": "BlockShuffling",
                                "collection_type": collection_type,
                                "batch_size": batch_size,
                                "block_size": block_size,
                                "fetch_factor": fetch_factor,
                                "num_workers": num_workers,
                                "prefetch_factor": prefetch_factor,
                                **result,
                            }
                        )

                        # Save results after each experiment
                        save_results_to_csv(results, results_path)

                        del dataset, loader
                        gc.collect()

    # Test scDataset with BlockWeightedSampling strategy (new)
    if test_modes in ["all", "weighted"]:
        print("\nTesting scDataset with BlockWeightedSampling strategy...")
        for batch_size in batch_sizes:
            for block_size in block_sizes:
                for fetch_factor in fetch_factors:
                    for num_workers in num_workers_options:
                        prefetch_factor = fetch_factor + 1 if num_workers > 0 else None

                        # Choose appropriate fetch transform based on collection type
                        if collection_type == "huggingface":
                            fetch_transform = fetch_transform_hf
                        elif collection_type == "anncollection":
                            fetch_transform = partial(
                                fetch_transform_adata, columns=["plate"]
                            )
                        elif collection_type == "bionemo":
                            fetch_transform = None

                        extra_params = {}
                        if collection_type == "bionemo":
                            extra_params["fetch_callback"] = fetch_callback_bionemo

                        # Create BlockWeightedSampling strategy (uniform weights)
                        strategy = BlockWeightedSampling(block_size=block_size)

                        dataset = scDataset(
                            data_collection=data_collection,
                            strategy=strategy,
                            batch_size=batch_size,
                            fetch_factor=fetch_factor,
                            fetch_transform=fetch_transform,
                            **extra_params,
                        )

                        loader = DataLoader(
                            dataset,
                            batch_size=None,
                            num_workers=num_workers,
                            prefetch_factor=prefetch_factor,
                        )

                        desc = f"scDataset (BlockWeightedSampling) - batch={batch_size}, block={block_size}, ff={fetch_factor}, w={num_workers}"
                        result = evaluate_loader(loader, description=desc)

                        results.append(
                            {
                                "mode": "weighted",
                                "loader": "scDataset",
                                "strategy": "BlockWeightedSampling",
                                "collection_type": collection_type,
                                "batch_size": batch_size,
                                "block_size": block_size,
                                "fetch_factor": fetch_factor,
                                "num_workers": num_workers,
                                "prefetch_factor": prefetch_factor,
                                **result,
                            }
                        )

                        # Save results after each experiment
                        save_results_to_csv(results, results_path)

                        del dataset, loader
                        gc.collect()

    # Test scDataset in streaming mode using Streaming strategy
    if test_modes in ["all", "stream"]:
        print("\nTesting scDataset with Streaming strategy (stream mode)...")
        for batch_size in batch_sizes:
            for fetch_factor in fetch_factors:
                for num_workers in num_workers_options:
                    prefetch_factor = fetch_factor + 1 if num_workers > 0 else None

                    # Choose appropriate fetch transform based on collection type
                    if collection_type == "huggingface":
                        fetch_transform = fetch_transform_hf
                    elif collection_type == "anncollection":
                        fetch_transform = partial(
                            fetch_transform_adata, columns=["plate"]
                        )
                    elif collection_type == "bionemo":
                        fetch_transform = None

                    extra_params = {}
                    if collection_type == "bionemo":
                        extra_params["fetch_callback"] = fetch_callback_bionemo

                    # Create Streaming strategy
                    strategy = Streaming()

                    dataset = scDataset(
                        data_collection=data_collection,
                        strategy=strategy,
                        batch_size=batch_size,
                        fetch_factor=fetch_factor,
                        fetch_transform=fetch_transform,
                        **extra_params,
                    )

                    loader = DataLoader(
                        dataset,
                        batch_size=None,
                        num_workers=num_workers,
                        prefetch_factor=prefetch_factor,
                    )

                    desc = f"scDataset (Streaming) - batch={batch_size}, ff={fetch_factor}, w={num_workers}"
                    result = evaluate_loader(loader, description=desc)

                    results.append(
                        {
                            "mode": "stream",
                            "loader": "scDataset",
                            "strategy": "Streaming",
                            "collection_type": collection_type,
                            "batch_size": batch_size,
                            "block_size": None,  # Not applicable for streaming
                            "fetch_factor": fetch_factor,
                            "num_workers": num_workers,
                            "prefetch_factor": prefetch_factor,
                            **result,
                        }
                    )

                    # Save results after each experiment
                    save_results_to_csv(results, results_path)

                    del dataset, loader
                    gc.collect()

    # Load final results for display and visualization
    df = pd.read_csv(results_path)

    # Display and save results
    print("\n===== PERFORMANCE RESULTS =====")

    # Print results
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 150)
    print(
        df[
            [
                "mode",
                "loader",
                "collection_type",
                "batch_size",
                "block_size",
                "fetch_factor",
                "num_workers",
                "samples_per_second",
                "elapsed",
            ]
        ]
    )

    return df


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate AnnData loaders performance."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    print(f"Starting data loader performance evaluation using config: {args.config}")
    result_df = run_evaluations(args.config)
