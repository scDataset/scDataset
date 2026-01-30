Quick Start Guide
=================

This guide will help you get started with ``scDataset`` quickly.

Basic Concepts
--------------

``scDataset`` is built around two main concepts:

1. **Data Collections**: Any object that supports default indexing (``__getitem__``) or custom indexing that can be implemented with ``fetch_callback``
2. **Sampling Strategies**: Define how data is sampled and batched

Minimal Example
---------------

The simplest way to use ``scDataset`` is as a drop-in replacement for your existing dataset:

.. code-block:: python

   from scdataset import scDataset, Streaming
   from torch.utils.data import DataLoader
   import numpy as np

   # Your existing data (numpy array, AnnData, HuggingFace Dataset, etc.)
   data = np.random.randn(1000, 100)  # 1000 samples, 100 features
   
   # Create scDataset with streaming strategy
   dataset = scDataset(data, Streaming(), batch_size=64, fetch_factor=16)
   
   # Use with DataLoader (note: batch_size=None)
   loader = DataLoader(dataset, batch_size=None, num_workers=4, prefetch_factor=17)
   
   for batch in loader:
       print(f"Batch shape: {batch.shape}")  # (64, 100)
       # Your training code here
       break

.. note::
   Always set ``batch_size=None`` in the DataLoader when using ``scDataset``, 
   as batching is handled internally by ``scDataset``.

Sampling Strategies
-------------------

``scDataset`` supports several sampling strategies:

Streaming (Sequential)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import Streaming
   
   # Sequential access without shuffling
   strategy = Streaming()
   dataset = scDataset(data, strategy, batch_size=64)
   
   # Sequential access with buffer-level shuffling (similar to Ray Data/WebDataset)
   strategy = Streaming(shuffle=True)
   dataset = scDataset(data, strategy, batch_size=64)
   # This shuffles batches within each fetch buffer while maintaining
   # sequential order between buffers

Block Shuffling
~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import BlockShuffling
   
   # Shuffle in blocks for better I/O while maintaining some randomness
   strategy = BlockShuffling(block_size=16)
   dataset = scDataset(data, strategy, batch_size=64)

Weighted Sampling
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import BlockWeightedSampling
   
   # Sample with custom weights (e.g., higher weight for rare samples)
   weights = np.random.rand(len(data))  # Custom weights per sample
   strategy = BlockWeightedSampling(
       weights=weights, 
       total_size=10000,  # Generate 10000 samples per epoch
       block_size=16
   )
   dataset = scDataset(data, strategy, batch_size=64)

Class Balanced Sampling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import ClassBalancedSampling
   
   # Automatically balance classes
   labels = np.random.choice(['A', 'B', 'C'], size=len(data))
   strategy = ClassBalancedSampling(labels, total_size=10000)
   dataset = scDataset(data, strategy, batch_size=64)

Working with Different Data Formats
------------------------------------

NumPy Arrays
~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   
   data = np.random.randn(5000, 2000)
   dataset = scDataset(data, Streaming(), batch_size=64)

AnnData Objects
~~~~~~~~~~~~~~~

.. code-block:: python

   import anndata as ad
   import scanpy as sc
   
   # Load your single-cell data
   adata = sc.datasets.pbmc3k()
   
   # Use the expression matrix
   dataset = scDataset(adata.X, Streaming(), batch_size=64)
   
   # Or create a custom fetch callback for more complex data
   def fetch_adata(collection, indices):
       return collection[indices].X.toarray()
   
   dataset = scDataset(adata, Streaming(), batch_size=64, fetch_callback=fetch_adata)

AnnCollection (Multiple Files)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For large datasets spanning multiple files, use ``AnnCollection`` with backed mode:

.. code-block:: python

   import anndata as ad
   from anndata.experimental import AnnCollection
   from scdataset import scDataset, BlockShuffling
   from scdataset.transforms import adata_to_mindex
   from torch.utils.data import DataLoader
   
   # Load multiple AnnData files in backed mode (memory-efficient)
   adatas = [
       ad.read_h5ad("plate1.h5ad", backed='r'),
       ad.read_h5ad("plate2.h5ad", backed='r'),
   ]
   collection = AnnCollection(adatas)
   
   # Create dataset with adata_to_mindex to materialize backed data
   dataset = scDataset(
       collection,
       BlockShuffling(block_size=32),
       batch_size=64,
       fetch_factor=32,
       fetch_transform=adata_to_mindex  # Calls to_adata() internally
   )
   
   loader = DataLoader(dataset, batch_size=None, num_workers=8, prefetch_factor=33)

See :doc:`examples` for a complete AnnCollection pipeline with transforms.

HuggingFace Datasets
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datasets import load_dataset
   
   dataset_hf = load_dataset("your/dataset", split="train")
   dataset = scDataset(dataset_hf, Streaming(), batch_size=64)

Performance Optimization
-------------------------

For optimal performance with large datasets, consider these guidelines:

**Automatic Configuration**

Use :func:`~scdataset.experimental.suggest_parameters` for automatic tuning:

.. code-block:: python

   from scdataset.experimental import suggest_parameters
   
   params = suggest_parameters(data_collection, batch_size=64)
   print(params)  # {'fetch_factor': 64, 'block_size': 32, 'num_workers': 8}

**Manual Tuning Guidelines**

1. **prefetch_factor**: Set to ``fetch_factor + 1`` in DataLoader to trigger the 
   next fetch while processing the current one.

2. **num_workers**: 4-12 workers typically works well for most systems. Start with 
   4 and increase if GPU utilization is low.

3. **fetch_factor**: Can be large (e.g., 64, 128, or even 256) if you have enough 
   memory. Larger values amortize I/O overhead but increase memory usage.

4. **block_size**: Tune based on ``fetch_factor`` to balance randomness vs throughput:
   
   - ``block_size = fetch_factor``: Best tradeoff between randomness and throughput
   - ``block_size = fetch_factor / 2`` or ``/ 4``: Conservative, good randomness
   - ``block_size = fetch_factor * 2``: Higher throughput, less randomness
   - Going beyond the above values has diminishing returns.

**Example Configuration**

.. code-block:: python

   dataset = scDataset(
       data,
       BlockShuffling(block_size=256),
       batch_size=64,
       fetch_factor=256,                 # Large fetch for efficiency
   )
   
   loader = DataLoader(
       dataset,
       batch_size=None,
       num_workers=8,           # 4-12 workers typically optimal
       prefetch_factor=257,      # fetch_factor + 1
       pin_memory=True,         # For GPU training
   )

Data Transforms
---------------

You can apply transforms at different stages:

.. code-block:: python
   
   def preprocess_fetch(data):
       # Apply to fetched data before batching
       return data.astype(np.float32)

   def normalize_batch(batch):
       # Apply per-batch normalization
       return (batch - batch.mean()) / batch.std()
   
   dataset = scDataset(
       data,
       Streaming(),
       batch_size=64,
       fetch_transform=preprocess_fetch,
       batch_transform=normalize_batch
   )

Next Steps
----------

* See :doc:`examples` for more detailed use cases
* Check the :doc:`scdataset` for complete API reference
* Read about advanced features in the full examples
