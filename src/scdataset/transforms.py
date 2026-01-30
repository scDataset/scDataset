"""
Transform functions for scDataset.

This module provides utility transform functions for common data formats
like AnnData/AnnCollection, HuggingFace datasets, and BioNeMo datasets.
These transforms can be used as ``fetch_callback``, ``fetch_transform``,
``batch_callback``, or ``batch_transform`` arguments in scDataset.

.. autosummary::
   :toctree: generated/

   adata_to_mindex
   hf_tahoe_to_tensor
   bionemo_to_tensor
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

__all__ = [
    "adata_to_mindex",
    "hf_tahoe_to_tensor",
    "bionemo_to_tensor",
]


def adata_to_mindex(batch, columns: Optional[List[str]] = None):
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

        - ``.to_memory()`` method (for AnnCollection/backed AnnData)
        - ``.X`` attribute (expression matrix)
        - ``.obs`` attribute (observation metadata)

    columns : list of str, optional
        List of observation column names to include in the output.
        If None, only the X matrix is included.

    Returns
    -------
    MultiIndexable
        A MultiIndexable object with:

        - ``'X'``: Dense expression matrix as numpy array
        - Additional keys for each column in ``columns`` (as numpy arrays)

    Examples
    --------
    >>> # Basic usage - just X matrix
    >>> from scdataset import scDataset, BlockShuffling
    >>> from scdataset.transforms import adata_to_mindex
    >>>
    >>> dataset = scDataset(
    ...     ann_collection,
    ...     BlockShuffling(),
    ...     batch_size=64,
    ...     fetch_transform=adata_to_mindex
    ... )

    >>> # With observation columns using functools.partial
    >>> from functools import partial
    >>> fetch_fn = partial(adata_to_mindex, columns=['cell_type', 'batch'])
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
    This transform calls ``.to_memory()`` to materialize the AnnData object,
    which is necessary when working with backed or lazy AnnCollection objects.

    Sparse matrices are automatically converted to dense numpy arrays for
    compatibility with standard indexing operations.

    See Also
    --------
    MultiIndexable : Container for synchronized multi-modal data
    """
    # Import here to avoid circular imports
    from .multiindexable import MultiIndexable

    # Import scipy.sparse locally to avoid hard dependency
    try:
        import scipy.sparse as sp
    except ImportError:
        sp = None

    # Materialize the AnnData batch in memory
    # Handle different AnnData-like types:
    # - AnnData: use to_memory() if backed
    # - AnnCollectionView: use to_adata()
    if hasattr(batch, "to_adata"):
        # AnnCollectionView from AnnCollection indexing
        batch = batch.to_adata()
    elif hasattr(batch, "to_memory"):
        # Backed AnnData
        batch = batch.to_memory()
    # else: already in-memory AnnData, use as-is

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


def hf_tahoe_to_tensor(batch, num_genes: int = 62713):
    """
    Transform HuggingFace Tahoe-100M sparse gene expression data to dense tensors.

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
    >>> from scdataset.transforms import hf_tahoe_to_tensor
    >>>
    >>> dataset = scDataset(
    ...     hf_dataset,
    ...     BlockShuffling(),
    ...     batch_size=64,
    ...     fetch_transform=hf_tahoe_to_tensor
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
        [np.full(length, i, dtype=np.int64) for i, length in enumerate(lengths)]
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


def bionemo_to_tensor(
    data_collection, idx: Union[int, slice, Sequence[int], np.ndarray, torch.Tensor]
) -> torch.Tensor:
    """
    Fetch callback for BioNeMo SingleCellMemMapDataset.

    This callback provides custom indexing logic for BioNeMo's
    SingleCellMemMapDataset, which returns sparse matrices that need
    to be collated and densified for use with scDataset.

    Use this as a ``fetch_callback`` in scDataset.

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
    >>> from scdataset.transforms import bionemo_to_tensor
    >>> from bionemo.scdl.io.single_cell_memmap_dataset import SingleCellMemMapDataset
    >>>
    >>> bionemo_data = SingleCellMemMapDataset(data_path='/path/to/data')
    >>> dataset = scDataset(
    ...     bionemo_data,
    ...     BlockShuffling(),
    ...     batch_size=64,
    ...     fetch_callback=bionemo_to_tensor
    ... )

    Notes
    -----
    This callback requires the bionemo-scdl package to be installed.
    The collate function handles the sparse matrix format used by BioNeMo.

    Raises
    ------
    ImportError
        If bionemo-scdl is not installed.
    """
    # Import bionemo collate function locally to avoid hard dependency
    try:
        from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch
    except ImportError as e:
        raise ImportError(
            "bionemo_to_tensor requires the bionemo-scdl package. "
            "Install it with: pip install bionemo-scdl"
        ) from e

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
