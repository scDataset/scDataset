Examples
========

This section provides comprehensive examples of using ``scDataset`` in various scenarios.

Single-Cell Data Analysis
--------------------------

Working with AnnData
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import anndata as ad
   import scanpy as sc
   import numpy as np
   from scdataset import scDataset, BlockShuffling
   from torch.utils.data import DataLoader
   
   # Load single-cell data
   adata = sc.datasets.pbmc3k_processed()
   
   # Create custom fetch function for AnnData
   def fetch_anndata(adata, indices):
       # Get expression matrix and convert to dense if sparse
       data = adata[indices].X
       if hasattr(data, 'toarray'):
           data = data.toarray()
       return data.astype(np.float32)
   
   # Create dataset with block shuffling
   dataset = scDataset(
       adata,
       BlockShuffling(block_size=8),
       batch_size=64,
       fetch_callback=fetch_anndata
   )
   
   # Use with DataLoader
   loader = DataLoader(dataset, batch_size=None, num_workers=4)
   
   for batch in loader:
       print(f"Processing batch of shape: {batch.shape}")
       # Your model training code here
       break

Class-Balanced Training
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from scdataset import ClassBalancedSampling
   
   # Assume you have cell type annotations
   cell_types = adata.obs['cell_type'].values
   
   # Create balanced sampling strategy
   strategy = ClassBalancedSampling(
       cell_types, 
       total_size=10000,  # Generate 10k balanced samples per epoch
       block_size=8
   )
   
   dataset = scDataset(adata, strategy, batch_size=32, fetch_callback=fetch_anndata)

   loader = DataLoader(dataset, batch_size=None, num_workers=4)

   # Training loop with balanced batches
   for epoch in range(10):
       for batch in loader:
           # Each batch will be class-balanced
           train_step(batch)

Multi-Modal Data
~~~~~~~~~~~~~~~~

.. code-block:: python

   def fetch_multimodal(adata, indices):
       # Fetch both gene expression and protein data
       gene_data = adata[indices].X.toarray()
       protein_data = adata[indices].obsm['protein'].toarray()
       
       return {
           'genes': gene_data.astype(np.float32),
           'proteins': protein_data.astype(np.float32),
           'cell_types': adata[indices].obs['cell_type'].values
       }
   
   dataset = scDataset(
       adata,
       BlockShuffling(block_size=8),
       batch_size=32,
       fetch_callback=fetch_multimodal
   )

Large-Scale Training
--------------------

Memory-Efficient Data Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scdataset import BlockWeightedSampling
   
   # For very large datasets, use higher fetch factors
   dataset = scDataset(
       large_data_collection,
       BlockShuffling(block_size=4),
       batch_size=64,
       fetch_factor=16,  # Fetch 16 batches worth of data at once
   )
   
   # Configure DataLoader for optimal performance
   loader = DataLoader(
       dataset,
       batch_size=None,
       num_workers=12,          # Use multiple workers
       prefetch_factor=17,      # fetch_factor + 1
       pin_memory=True,        # For GPU training
   )

Subset Training and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import train_test_split
   
   # Split indices for train/validation
   indices = np.arange(len(data))
   train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
   
   # Training dataset
   train_dataset = scDataset(
       data,
       BlockShuffling(indices=train_idx, block_size=8),
       batch_size=64
   )
   
   # Validation dataset (streaming for deterministic evaluation)
   val_dataset = scDataset(
       data,
       Streaming(indices=val_idx),
       batch_size=64
   )

   # Training loader
   train_loader = DataLoader(train_dataset, batch_size=None)

   # Validation loader
   val_loader = DataLoader(val_dataset, batch_size=None)

   # Training loop
   for epoch in range(num_epochs):
       # Training
       for batch in train_loader:
           train_step(batch)
       
       # Validation
       for batch in val_loader:
           val_step(batch)

Custom Data Transformations
----------------------------

On-the-Fly Normalization
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def log_normalize(batch):
       # Apply log1p normalization per batch
       return np.log1p(batch)
   
   def standardize_genes(batch):
       # Standardize genes (features) across batch
       return (batch - batch.mean(axis=0)) / (batch.std(axis=0) + 1e-8)
   
   dataset = scDataset(
       data,
       BlockShuffling(block_size=8),
       batch_size=64,
       batch_transform=lambda x: standardize_genes(log_normalize(x))
   )

Data Augmentation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def add_noise(batch, noise_level=0.1):
       # Add Gaussian noise for data augmentation
       noise = np.random.normal(0, noise_level, batch.shape)
       return batch + noise
   
   def dropout_genes(batch, dropout_rate=0.1):
       # Randomly set some genes to zero
       mask = np.random.random(batch.shape) > dropout_rate
       return batch * mask
   
   def augment_batch(batch):
       batch = add_noise(batch)
       batch = dropout_genes(batch)
       return batch.astype(np.float32)
   
   dataset = scDataset(
       data,
       BlockShuffling(block_size=8),
       batch_size=64,
       batch_transform=augment_batch
   )

Working with HuggingFace Datasets
----------------------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from datasets import load_dataset
   from torch.utils.data import DataLoader
   
   # Load a HuggingFace dataset
   hf_dataset = load_dataset("imdb", split="train[:1000]")
   
   # Custom batch callback for HuggingFace datasets
   def extract_hf_batch(fetched_data, batch_indices):
       """Extract a batch from HuggingFace dataset fetched data."""
       batch = {}
       for key, values in fetched_data.items():
           batch[key] = [values[i] for i in batch_indices]
       return batch
   
   # Create dataset with custom batch callback
   dataset = scDataset(
       hf_dataset, 
       Streaming(), 
       batch_size=64,
       batch_callback=extract_hf_batch
   )
   
   for batch in DataLoader(dataset, batch_size=None):
       # batch will be a dictionary with dataset features
       print("Batch keys:", batch.keys())
       print("Batch size:", len(batch['text']))
       break

Custom Processing for HuggingFace Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def extract_hf_batch(fetched_data, batch_indices):
       """Extract a batch from HuggingFace dataset fetched data."""
       batch = {}
       for key, values in fetched_data.items():
           batch[key] = [values[i] for i in batch_indices]
       return batch

   def process_hf_batch(batch_dict):
       """Process HuggingFace batch into numpy arrays."""
       # Extract and process specific features
       features = np.array(batch_dict['expression'])
       labels = np.array(batch_dict['cell_type_id'])
       
       return {
           'features': features.astype(np.float32),
           'labels': labels.astype(np.int64)
       }
   
   dataset = scDataset(
       hf_dataset,
       BlockShuffling(block_size=64),
       batch_size=32,
       batch_callback=extract_hf_batch,
       batch_transform=process_hf_batch
   )

Integration with PyTorch Lightning
-----------------------------------

.. code-block:: python

   import pytorch_lightning as pl
   from torch.utils.data import DataLoader
   
   class SingleCellDataModule(pl.LightningDataModule):
       def __init__(self, data_path, batch_size=64, num_workers=4):
           super().__init__()
           self.data_path = data_path
           self.batch_size = batch_size
           self.num_workers = num_workers
           
       def setup(self, stage=None):
           # Load your data
           self.data = load_data(self.data_path)
           
           # Split indices
           indices = np.arange(len(self.data))
           train_idx, val_idx = train_test_split(indices, test_size=0.2)
           
           # Create datasets
           self.train_dataset = scDataset(
               self.data,
               BlockShuffling(block_size=8),
               batch_size=self.batch_size
           )
           
           self.val_dataset = scDataset(
               self.data,
               Streaming(indices=val_idx),
               batch_size=self.batch_size
           )
       
       def train_dataloader(self):
           return DataLoader(
               self.train_dataset,
               num_workers=self.num_workers,
               prefetch_factor=2
           )
       
       def val_dataloader(self):
           return DataLoader(
               self.val_dataset,
               num_workers=self.num_workers,
               prefetch_factor=2
           )

Advanced Sampling Strategies
-----------------------------

Custom Weighted Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create weights based on cell type frequency (inverse frequency weighting)
   cell_types = adata.obs['cell_type']
   type_counts = cell_types.value_counts()
   weights = 1.0 / type_counts[cell_types].values
   weights = weights / weights.sum()  # Normalize
   
   strategy = BlockWeightedSampling(
       weights=weights,
       total_size=5000,
       block_size=8,
       replace=True
   )
   
   dataset = scDataset(adata, strategy, batch_size=32)

Temporal Sampling for Time-Series Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom strategy for time-series single-cell data
   def create_temporal_indices(timepoints, window_size=5):
       indices = []
       for i in range(len(timepoints) - window_size + 1):
           indices.extend(range(i, i + window_size))
       return np.array(indices)
   
   temporal_indices = create_temporal_indices(adata.obs['timepoint'])
   
   dataset = scDataset(
       adata,
       Streaming(indices=temporal_indices),
       batch_size=32
   )

Performance Benchmarking
------------------------

.. code-block:: python

   import time
   from contextlib import contextmanager
   
   @contextmanager
   def timer():
       start = time.time()
       yield
       end = time.time()
       print(f"Time taken: {end - start:.2f} seconds")
   
   # Compare different configurations
   configs = [
       {'block_size': 32, 'fetch_factor': 1},
       {'block_size': 64, 'fetch_factor': 2},
       {'block_size': 128, 'fetch_factor': 4},
   ]
   
   for config in configs:
       dataset = scDataset(
           large_data,
           BlockShuffling(block_size=config['block_size']),
           batch_size=64,
           fetch_factor=config['fetch_factor']
       )
       
       loader = DataLoader(dataset, num_workers=4)
       
       with timer():
           for i, batch in enumerate(loader):
               if i >= 100:  # Test first 100 batches
                   break
       
       print(f"Config {config}: done")

Tips and Best Practices
------------------------

1. **Choose appropriate block sizes**: Larger blocks (128-512) work better for sequential data access, smaller blocks (16-64) for more randomness.

2. **Use fetch_factor > 1** for better I/O efficiency, especially with slow storage.

3. **Set prefetch_factor = fetch_factor + 1** in DataLoader for optimal performance.

4. **For validation**, use ``Streaming`` strategy for deterministic results.

5. **For large datasets**, consider using fewer workers but higher fetch_factor to reduce memory overhead.

6. **Profile your pipeline** to find the optimal configuration for your specific data and hardware setup.
