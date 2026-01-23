"""
scDataset: Efficient PyTorch Datasets for Single-Cell Data
==========================================================

A PyTorch-compatible dataset library designed specifically for large-scale
single-cell data analysis. Provides flexible sampling strategies and efficient
data loading patterns optimized for genomics and single-cell datasets.

Main Components
---------------

**Core Classes**

* :class:`~scdataset.scDataset` - Main iterable dataset class with configurable sampling
* :class:`~scdataset.MultiIndexable` - Container for multi-modal data with synchronized indexing
* :class:`~scdataset.strategy.SamplingStrategy` - Abstract base for sampling strategies

**Sampling Strategies**

* :class:`~scdataset.strategy.Streaming` - Sequential sampling without shuffling
* :class:`~scdataset.strategy.BlockShuffling` - Block-based shuffling for locality
* :class:`~scdataset.strategy.BlockWeightedSampling` - Weighted sampling with blocks
* :class:`~scdataset.strategy.ClassBalancedSampling` - Automatic class balancing

Quick Start
-----------

Basic usage with streaming (sequential) sampling::

    from scdataset import scDataset, Streaming
    import numpy as np

    # Create sample data
    data = np.random.randn(10000, 2000)  # 10k cells, 2k genes

    # Create dataset with streaming strategy
    dataset = scDataset(data, Streaming(), batch_size=64)

    # Use with PyTorch DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=None, num_workers=4)

    for batch in loader:
        print(f"Batch shape: {batch.shape}")
        break

Block shuffling for better randomization while maintaining some locality::

    from scdataset import BlockShuffling

    # Shuffle in blocks of 8 samples
    strategy = BlockShuffling(block_size=8)
    dataset = scDataset(data, strategy, batch_size=64)

Weighted sampling for handling imbalanced data::

    from scdataset import BlockWeightedSampling
    import numpy as np

    # Sample with custom weights (e.g., higher weight for rare samples)
    weights = np.random.rand(len(data))
    strategy = BlockWeightedSampling(weights=weights, total_size=500, block_size=32)
    dataset = scDataset(data, strategy, batch_size=64)

Class-balanced sampling for imbalanced datasets::

    from scdataset import ClassBalancedSampling

    # Assume we have cell type labels
    cell_types = np.random.choice(['T_cell', 'B_cell', 'NK_cell'], size=10000,
                                 p=[0.7, 0.2, 0.1])  # Imbalanced

    # Automatically balance classes
    strategy = ClassBalancedSampling(cell_types, total_size=5000)
    dataset = scDataset(data, strategy, batch_size=64)

Multi-modal data with synchronized indexing::

    from scdataset import MultiIndexable

    # Group multiple data modalities together
    multimodal = MultiIndexable(
        genes=gene_expression_data,    # Shape: (n_cells, n_genes)
        proteins=protein_data,         # Shape: (n_cells, n_proteins)
        metadata=cell_metadata         # Shape: (n_cells, n_features)
    )

    # Use with scDataset - all modalities indexed together
    dataset = scDataset(multimodal, Streaming(), batch_size=64)

    for batch in dataset:
        genes = batch['genes']      # Genes for this batch
        proteins = batch['proteins'] # Corresponding proteins

Key Features
------------

* **Memory Efficient**: Streams data without loading everything into memory
* **Flexible Sampling**: Multiple sampling strategies for different use cases
* **PyTorch Compatible**: Works seamlessly with PyTorch DataLoader and multiprocessing
* **Customizable**: Support for custom fetch/batch callbacks and transforms
* **Auto-configuration**: Automatic parameter suggestion based on system resources (experimental)

Performance Tips
----------------

* Use ``block_size > 1`` to read data in contiguous chunks
* Use ``fetch_factor > 1`` to fetch multiple batches at once for better I/O
* Use ``num_workers > 0`` in DataLoader for parallel data loading
* Use ``suggest_parameters()`` to get optimal settings for your system (experimental)
"""
# See Also
# --------
# * :doc:`/api/index` - Complete API reference
# * :doc:`/examples/index` - Usage examples and tutorials

# Re-export experimental auto_config functions for convenience
from .experimental.auto_config import (
    estimate_sample_size,
    suggest_parameters,
)
from .multiindexable import MultiIndexable
from .scdataset import scDataset
from .strategy import (
    BlockShuffling,
    BlockWeightedSampling,
    ClassBalancedSampling,
    SamplingStrategy,
    Streaming,
)
from .transforms import (
    adata_to_mindex,
    bionemo_to_tensor,
    hf_tahoe_to_tensor,
)

__version__ = "0.3.0"

__all__ = [
    "scDataset",
    "SamplingStrategy",
    "Streaming",
    "BlockShuffling",
    "BlockWeightedSampling",
    "ClassBalancedSampling",
    "MultiIndexable",
    "suggest_parameters",
    "estimate_sample_size",
    "adata_to_mindex",
    "hf_tahoe_to_tensor",
    "bionemo_to_tensor",
]
