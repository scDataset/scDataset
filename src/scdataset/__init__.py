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
    loader = DataLoader(dataset, num_workers=4)
    
    for batch in loader:
        print(f"Batch shape: {batch.shape}")
        break

Block shuffling for better randomization while maintaining some locality::

    from scdataset import BlockShuffling
    
    # Shuffle in blocks of 32 samples
    strategy = BlockShuffling(block_size=32)
    dataset = scDataset(data, strategy, batch_size=64)

Class-balanced sampling for imbalanced datasets::

    from scdataset import ClassBalancedSampling
    
    # Assume we have cell type labels
    cell_types = np.random.choice(['T_cell', 'B_cell', 'NK_cell'], size=10000, 
                                 p=[0.7, 0.2, 0.1])  # Imbalanced
    
    # Automatically balance classes
    strategy = ClassBalancedSampling(cell_types, total_size=5000)
    dataset = scDataset(data, strategy, batch_size=64)

Key Features
------------

* **Memory Efficient**: Streams data without loading everything into memory
* **Flexible Sampling**: Multiple sampling strategies for different use cases  
* **PyTorch Compatible**: Works seamlessly with PyTorch DataLoader and multiprocessing
* **Customizable**: Support for custom fetch/batch callbacks and transforms
* **Single-Cell Optimized**: Designed for the specific patterns of genomics data

Performance Tips
----------------

* Use ``fetch_factor > 1`` to fetch multiple batches at once for better I/O
* Set ``block_size`` in shuffling strategies based on your data access patterns
* Use ``num_workers > 0`` in DataLoader for parallel data loading

See Also
--------
* :doc:`/api/index` - Complete API reference
* :doc:`/examples/index` - Usage examples and tutorials
"""

from .scdataset import scDataset
from .strategy import (
    SamplingStrategy,
    Streaming, 
    BlockShuffling, 
    BlockWeightedSampling, 
    ClassBalancedSampling
)

__version__ = "0.1.0"

__all__ = [
    "scDataset",
    "SamplingStrategy", 
    "Streaming",
    "BlockShuffling", 
    "BlockWeightedSampling",
    "ClassBalancedSampling",
]