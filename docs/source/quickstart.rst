Quick Start Guide
================

This guide will help you get started with ``scDataset`` quickly.

Basic Concepts
--------------

``scDataset`` is built around two main concepts:

1. **Data Collections**: Any object that supports indexing (``__getitem__``) and length (``__len__``)
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
   dataset = scDataset(data, Streaming(), batch_size=32)
   
   # Use with DataLoader (note: batch_size=None)
   loader = DataLoader(dataset, batch_size=None, num_workers=2)
   
   for batch in loader:
       print(f"Batch shape: {batch.shape}")  # (32, 100)
       # Your training code here
       break

.. note::
   Always set ``batch_size=None`` in the DataLoader when using ``scDataset``, 
   as batching is handled internally by the dataset.

Sampling Strategies
-------------------

``scDataset`` supports several sampling strategies:

Streaming (Sequential)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import Streaming
   
   # Sequential access without shuffling
   strategy = Streaming()
   dataset = scDataset(data, strategy, batch_size=32)

Block Shuffling
~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import BlockShuffling
   
   # Shuffle in blocks for better I/O while maintaining some randomness
   strategy = BlockShuffling(block_size=64)
   dataset = scDataset(data, strategy, batch_size=32)

Weighted Sampling
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import BlockWeightedSampling
   
   # Sample with custom weights
   weights = np.random.rand(len(data))  # Custom weights per sample
   strategy = BlockWeightedSampling(weights=weights, total_size=500)
   dataset = scDataset(data, strategy, batch_size=32)

Class Balanced Sampling
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import ClassBalancedSampling
   
   # Automatically balance classes
   labels = np.random.choice(['A', 'B', 'C'], size=len(data))
   strategy = ClassBalancedSampling(labels, total_size=600)
   dataset = scDataset(data, strategy, batch_size=32)

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

HuggingFace Datasets
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from datasets import load_dataset
   
   dataset_hf = load_dataset("your/dataset", split="train")
   dataset = scDataset(dataset_hf, Streaming(), batch_size=32)

Performance Optimization
-------------------------

For large datasets, you can optimize performance using these parameters:

.. code-block:: python

   dataset = scDataset(
       data,
       BlockShuffling(block_size=4),  # Larger blocks for better I/O
       batch_size=64,
       fetch_factor=16,  # Fetch 16 batches at once
   )
   
   loader = DataLoader(
       dataset,
       num_workers=12,        # Multiple workers for parallel loading
       prefetch_factor=17,    # fetch_factor + 1
   )

Data Transforms
---------------

You can apply transforms at different stages:

.. code-block:: python

   def normalize_batch(batch):
       # Apply per-batch normalization
       return (batch - batch.mean()) / batch.std()
   
   def preprocess_fetch(data):
       # Apply to fetched data before batching
       return data.astype(np.float32)
   
   dataset = scDataset(
       data,
       Streaming(),
       batch_size=32,
       fetch_transform=preprocess_fetch,
       batch_transform=normalize_batch
   )

Next Steps
----------

* See :doc:`examples` for more detailed use cases
* Check the :doc:`api` for complete API reference
* Read about advanced features in the full examples
