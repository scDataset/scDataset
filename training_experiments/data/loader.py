"""
Tahoe-100M dataset loader with scDataset support.

This module provides data loading utilities for the Tahoe-100M dataset,
supporting all 6 data loading strategies:
1. Streaming
2. Streaming with Buffer
3. Block Shuffling (block_size=4)
4. Random Sampling (block_size=1)
5. Block Weighted Sampling (block_size=4)
6. True Weighted Sampling (block_size=1)
"""

import os
from typing import Callable, Dict, Optional, Tuple

import anndata as ad
import numpy as np
import torch
from anndata.experimental import AnnCollection
from scipy import sparse
from torch.utils.data import DataLoader

from scdataset import BlockShuffling, BlockWeightedSampling, Streaming, scDataset
from scdataset.multiindexable import MultiIndexable
from training_experiments.data.label_encoder import LabelEncoder
from training_experiments.utils.weights import compute_balanced_weights

# Strategy name constants
STRATEGY_STREAMING = "streaming"
STRATEGY_STREAMING_BUFFER = "streaming_buffer"
STRATEGY_BLOCK_SHUFFLING = "block_shuffling"
STRATEGY_RANDOM_SAMPLING = "random_sampling"
STRATEGY_BLOCK_WEIGHTED = "block_weighted"
STRATEGY_TRUE_WEIGHTED = "true_weighted"

ALL_STRATEGIES = [
    STRATEGY_STREAMING,
    STRATEGY_STREAMING_BUFFER,
    STRATEGY_BLOCK_SHUFFLING,
    STRATEGY_RANDOM_SAMPLING,
    STRATEGY_BLOCK_WEIGHTED,
    STRATEGY_TRUE_WEIGHTED,
]


class TahoeDataLoader:
    """
    Data loader for Tahoe-100M dataset using h5ad files and AnnCollection.

    This loader creates train/test splits using plates 1-13 for training and
    plate 14 for testing. It supports all 6 data loading strategies.

    Parameters
    ----------
    data_dir : str
        Directory containing h5ad files
    label_dir : str, optional
        Directory containing label mapping pickle files.
        If None, uses the default mappings directory within training_experiments.
    """

    def __init__(
        self,
        data_dir: str = "/home/kidara/raid/volume/vevo-data/2025-02-25/original_h5ad",
        label_dir: str = None,
    ):
        """
        Initialize the Tahoe data loader.

        Parameters
        ----------
        data_dir : str
            Directory containing h5ad files
        label_dir : str, optional
            Directory containing label mapping pickle files.
            If None, uses the default mappings directory.
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.label_encoder = LabelEncoder(label_dir)

        # Collections for train and test
        self.train_collection: Optional[AnnCollection] = None
        self.test_collection: Optional[AnnCollection] = None
        self._feature_dim: Optional[int] = None

    def create_collections(self, verbose: bool = True) -> None:
        """
        Create AnnCollection objects for train and test sets.

        Training uses plates 1-13, testing uses plate 14.

        Parameters
        ----------
        verbose : bool
            Whether to print progress information
        """
        if verbose:
            print("Creating AnnCollection for training set (plates 1-13)...")

        # Load training plates (1-13)
        train_adatas = []
        for i in range(1, 14):
            path = f"{self.data_dir}/plate{i}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"
            if os.path.exists(path):
                adata = ad.read_h5ad(path, backed="r")
                # Keep only necessary columns
                adata.obs = adata.obs[["cell_line", "drug"]]
                train_adatas.append(adata)
            elif verbose:
                print(f"Warning: {path} not found")

        if not train_adatas:
            raise FileNotFoundError(f"No training files found in {self.data_dir}")

        self.train_collection = AnnCollection(train_adatas)
        self._feature_dim = self.train_collection.n_vars

        if verbose:
            print(
                f"Training set: {self.train_collection.n_obs:,} cells, "
                f"{self._feature_dim:,} genes"
            )

        # Load test plate (14)
        if verbose:
            print("Creating AnnCollection for test set (plate 14)...")

        test_path = f"{self.data_dir}/plate14_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad"

        if os.path.exists(test_path):
            test_adata = ad.read_h5ad(test_path, backed="r")
            test_adata.obs = test_adata.obs[["cell_line", "drug"]]
            self.test_collection = AnnCollection([test_adata])
            if verbose:
                print(f"Test set: {self.test_collection.n_obs:,} cells")
        else:
            raise FileNotFoundError(f"Test file not found: {test_path}")

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension (number of genes)."""
        if self._feature_dim is None:
            raise RuntimeError("Call create_collections() first")
        return self._feature_dim

    @property
    def task_dims(self) -> Dict[str, int]:
        """Get number of classes for each task."""
        return self.label_encoder.get_task_dims()

    def _create_batch_transform(self) -> Callable:
        """
        Create batch transform function with label encoder closure.

        Returns
        -------
        callable
            Batch transform function
        """
        label_encoder = self.label_encoder

        def batch_transform(batch) -> MultiIndexable:
            """
            Batch transform that extracts features and labels.

            Returns a MultiIndexable with X, cell_line, drug, moa_broad, moa_fine.
            """
            # Extract gene expression matrix
            X = batch.X
            if sparse.issparse(X):
                X = X.toarray()
            X = X.astype(np.float32)
            features = torch.from_numpy(X)

            # Get observation data
            obs = batch.obs
            cell_lines = obs["cell_line"].tolist()
            drugs = obs["drug"].tolist()

            # Get all task labels for the batch
            batch_labels = label_encoder.encode_labels(cell_lines, drugs)

            # Unpack to separate lists
            cell_line_ids, drug_ids, moa_broad_ids, moa_fine_ids = zip(*batch_labels)

            return MultiIndexable(
                X=features,
                cell_line=torch.tensor(cell_line_ids, dtype=torch.long),
                drug=torch.tensor(drug_ids, dtype=torch.long),
                moa_broad=torch.tensor(moa_broad_ids, dtype=torch.long),
                moa_fine=torch.tensor(moa_fine_ids, dtype=torch.long),
            )

        return batch_transform

    def _compute_weights(
        self, min_count_baseline: int = 1000, verbose: bool = True
    ) -> np.ndarray:
        """
        Compute balanced weights for (cell_line, drug) combinations.

        Parameters
        ----------
        min_count_baseline : int
            Minimum count baseline to add to all combination counts
            before computing weights. This prevents extreme reweighting
            for rare combinations.
        verbose : bool
            Whether to print weight statistics

        Returns
        -------
        numpy.ndarray
            Weight for each cell in the training collection
        """
        if self.train_collection is None:
            raise RuntimeError("Call create_collections() first")

        # Get observation data
        obs = self.train_collection.obs
        cell_lines = obs["cell_line"].values
        drugs = obs["drug"].values

        # Compute balanced weights
        weights = compute_balanced_weights(
            cell_lines=cell_lines, drugs=drugs, min_count_baseline=min_count_baseline
        )

        if verbose:
            print(
                f"Weight statistics: min={weights.min():.6f}, "
                f"max={weights.max():.6f}, mean={weights.mean():.6f}"
            )

        return weights

    def create_dataloaders(
        self,
        strategy_name: str,
        batch_size: int = 64,
        fetch_factor: int = 16,
        num_workers: int = 12,
        min_count_baseline: int = 1000,
        block_size: int | None = None,
        verbose: bool = True,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train and test sets.

        Parameters
        ----------
        strategy_name : str
            One of: 'streaming', 'streaming_buffer', 'block_shuffling',
            'random_sampling', 'block_weighted', 'true_weighted'
        batch_size : int
            Batch size for training
        fetch_factor : int
            Fetch factor for scDataset
        num_workers : int
            Number of workers for DataLoader
        min_count_baseline : int
            Minimum count baseline for weight computation (only used for
            weighted strategies)
        block_size : int, optional
            Block size for block-based strategies. If None, uses defaults:
            - block_shuffling: 4
            - random_sampling: 1
            - block_weighted: 4
            - true_weighted: 1
        verbose : bool
            Whether to print progress information

        Returns
        -------
        tuple
            (train_loader, test_loader)
        """
        if self.train_collection is None or self.test_collection is None:
            self.create_collections(verbose=verbose)

        if strategy_name not in ALL_STRATEGIES:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Must be one of: {ALL_STRATEGIES}"
            )

        # Create batch transform
        batch_transform = self._create_batch_transform()

        # Create fetch transform (just convert to AnnData)
        def fetch_transform(batch):
            return batch.to_adata()

        # Determine train workers based on strategy
        # Streaming strategies don't support multiprocessing
        if strategy_name in (STRATEGY_STREAMING, STRATEGY_STREAMING_BUFFER):
            train_workers = 1
        else:
            train_workers = num_workers

        # Create training strategy
        if strategy_name == STRATEGY_STREAMING:
            strategy = Streaming(indices=None, shuffle=False)

        elif strategy_name == STRATEGY_STREAMING_BUFFER:
            strategy = Streaming(indices=None, shuffle=True)

        elif strategy_name == STRATEGY_BLOCK_SHUFFLING:
            # Default block_size=4 for block shuffling
            effective_block_size = block_size if block_size is not None else 4
            strategy = BlockShuffling(
                block_size=effective_block_size, indices=None, drop_last=False
            )

        elif strategy_name == STRATEGY_RANDOM_SAMPLING:
            # block_size=1 mimics true random shuffling
            effective_block_size = block_size if block_size is not None else 1
            strategy = BlockShuffling(
                block_size=effective_block_size, indices=None, drop_last=False
            )

        elif strategy_name == STRATEGY_BLOCK_WEIGHTED:
            # Default block_size=4 for block weighted sampling
            effective_block_size = block_size if block_size is not None else 4
            weights = self._compute_weights(min_count_baseline, verbose)
            strategy = BlockWeightedSampling(
                block_size=effective_block_size,
                weights=weights,
                total_size=len(self.train_collection),
                replace=True,
            )

        elif strategy_name == STRATEGY_TRUE_WEIGHTED:
            # block_size=1 for true weighted sampling
            effective_block_size = block_size if block_size is not None else 1
            weights = self._compute_weights(min_count_baseline, verbose)
            strategy = BlockWeightedSampling(
                block_size=effective_block_size,
                weights=weights,
                total_size=len(self.train_collection),
                replace=True,
            )

        # Create train dataset
        train_dataset = scDataset(
            data_collection=self.train_collection,
            strategy=strategy,
            batch_size=batch_size,
            fetch_factor=fetch_factor,
            fetch_transform=fetch_transform,
            batch_transform=batch_transform,
            drop_last=False,
        )

        # Create test dataset (always use streaming without shuffle)
        test_strategy = Streaming(indices=None, shuffle=False)
        test_dataset = scDataset(
            data_collection=self.test_collection,
            strategy=test_strategy,
            batch_size=batch_size,
            fetch_factor=fetch_factor,
            fetch_transform=fetch_transform,
            batch_transform=batch_transform,
            drop_last=False,
        )

        # Create PyTorch DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=None,  # scDataset handles batching
            num_workers=train_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=None,  # scDataset handles batching
            num_workers=train_workers,
        )

        if verbose:
            print(f"Created dataloaders with strategy: {strategy_name}")
            print(f"  Train workers: {train_workers}")
            print(f"  Batch size: {batch_size}")
            print(f"  Fetch factor: {fetch_factor}")

        return train_loader, test_loader


def create_dataloaders(
    strategy_name: str,
    batch_size: int = 64,
    fetch_factor: int = 16,
    num_workers: int = 12,
    data_dir: str = "/home/kidara/raid/volume/vevo-data/2025-02-25/original_h5ad",
    label_dir: str = None,
    min_count_baseline: int = 1000,
    block_size: int | None = None,
    verbose: bool = True,
) -> Tuple[DataLoader, DataLoader, LabelEncoder, int]:
    """
    Convenience function to create dataloaders with a single call.

    Parameters
    ----------
    strategy_name : str
        One of: 'streaming', 'streaming_buffer', 'block_shuffling',
        'random_sampling', 'block_weighted', 'true_weighted'
    batch_size : int
        Batch size for training
    fetch_factor : int
        Fetch factor for scDataset
    num_workers : int
        Number of workers for DataLoader
    data_dir : str
        Directory containing h5ad files
    label_dir : str, optional
        Directory containing label mapping pickle files.
        If None, uses the default mappings directory.
    min_count_baseline : int
        Minimum count baseline for weight computation
    block_size : int, optional
        Block size for block-based strategies. If None, uses defaults.
    verbose : bool
        Whether to print progress information

    Returns
    -------
    tuple
        (train_loader, test_loader, label_encoder, feature_dim)
    """
    loader = TahoeDataLoader(data_dir=data_dir, label_dir=label_dir)
    loader.create_collections(verbose=verbose)

    train_loader, test_loader = loader.create_dataloaders(
        strategy_name=strategy_name,
        batch_size=batch_size,
        fetch_factor=fetch_factor,
        num_workers=num_workers,
        min_count_baseline=min_count_baseline,
        block_size=block_size,
        verbose=verbose,
    )

    return train_loader, test_loader, loader.label_encoder, loader.feature_dim
