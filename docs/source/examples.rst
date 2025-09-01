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

For multi-modal single-cell data (e.g., gene expression + protein measurements), 
you can use the :class:`MultiIndexable` class to keep different data modalities 
synchronized during indexing:

.. code-block:: python

   import numpy as np
   from scdataset import scDataset, BlockShuffling, MultiIndexable
   from torch.utils.data import DataLoader
   
   # Simulate multi-modal data
   n_cells = 1000
   gene_data = np.random.randn(n_cells, 2000)     # Gene expression
   protein_data = np.random.randn(n_cells, 100)   # Protein measurements  
   metadata = np.random.randn(n_cells, 10)        # Cell metadata
   
   # Method 1: Using keyword arguments
   multimodal_data = MultiIndexable(
       genes=gene_data,
       proteins=protein_data, 
       metadata=metadata
   )
   
   # Method 2: Using dictionary as positional argument  
   data_dict = {
       'genes': gene_data,
       'proteins': protein_data,
       'metadata': metadata
   }
   multimodal_data = MultiIndexable(data_dict)
   
   # Create dataset - all modalities will be indexed together
   dataset = scDataset(
       multimodal_data,
       BlockShuffling(block_size=8),
       batch_size=32
   )
   
   # Use with DataLoader
   loader = DataLoader(dataset, batch_size=None, num_workers=4)
   
   for batch in loader:
       genes = batch['genes']        # Shape: (32, 2000)
       proteins = batch['proteins']  # Shape: (32, 100)  
       meta = batch['metadata']      # Shape: (32, 10)
       
       print(f"Genes: {genes.shape}, Proteins: {proteins.shape}, Meta: {meta.shape}")
       # All correspond to the same 32 cells
       break

Alternative approach with custom fetch function (for AnnData objects):

.. code-block:: python

   def fetch_multimodal(adata, indices):
       # Fetch both gene expression and protein data
       gene_data = adata[indices].X.toarray()
       protein_data = adata[indices].obsm['protein'].toarray()
       
       return MultiIndexable(
           genes=gene_data,
           proteins=protein_data
       )
   
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
   
   # Create dataset with custom batch callback
   dataset = scDataset(
   )
   
   for batch in DataLoader(dataset, batch_size=None):

Custom Processing for HuggingFace Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def extract_hf_batch(fetched_data, batch_indices):

   def process_hf_batch(batch_dict):
   
   dataset = scDataset(
   )

Working with MultiIndexable
----------------------------

The :class:`MultiIndexable` class provides a convenient way to group multiple 
indexable objects that should be indexed together. This is particularly useful 
for multi-modal data or features and labels.

Basic MultiIndexable Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scdataset import MultiIndexable, scDataset, Streaming
   from torch.utils.data import DataLoader
   
   # Create sample data
   features = np.random.randn(1000, 50)  # Features
   labels = np.random.randint(0, 3, 1000)  # Labels
   
   # Group them together
   data = MultiIndexable(features, labels, names=['X', 'y'])
   
   # Or using dictionary syntax
   data = MultiIndexable(X=features, y=labels)
   
   # Create dataset
   dataset = scDataset(data, Streaming(), batch_size=64)
   loader = DataLoader(dataset, batch_size=None)
   
   for batch in loader:
       X_batch = batch['X']  # or batch[0]
       y_batch = batch['y']  # or batch[1]
       print(f"Features: {X_batch.shape}, Labels: {y_batch.shape}")
       break

Multi-Modal Single-Cell Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simulate CITE-seq data (RNA + protein)
   n_cells = 5000
   rna_data = np.random.randn(n_cells, 2000)      # Gene expression
   protein_data = np.random.randn(n_cells, 50)    # Surface proteins
   cell_types = np.random.choice(['T', 'B', 'NK'], n_cells)  # Labels
   
   # Group all modalities
   cite_seq_data = MultiIndexable(
       rna=rna_data,
       proteins=protein_data,
       cell_types=cell_types
   )
   
   # Use with class-balanced sampling
   from scdataset import ClassBalancedSampling
   strategy = ClassBalancedSampling(cell_types, total_size=2000, block_size=16)
   dataset = scDataset(cite_seq_data, strategy, batch_size=32)
   
   for batch in dataset:
       rna = batch['rna']           # RNA expression for 32 cells
       proteins = batch['proteins'] # Protein expression for same 32 cells  
       types = batch['cell_types']  # Cell type labels for same 32 cells
       # All data is synchronized - same cells across modalities
       break

Subsetting and Indexing
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create MultiIndexable
   data = MultiIndexable(
       features=np.random.randn(1000, 100),
       labels=np.random.randint(0, 5, 1000),
       metadata=np.random.randn(1000, 10)
   )
   
   # Access individual indexables
   features = data['features']  # or data[0]
   labels = data['labels']      # or data[1] 
   
   # Subset by sample indices - returns new MultiIndexable
   subset = data[100:200]       # Samples 100-199 from all modalities
   train_data = data[train_indices]  # Training subset
   
   # Check properties
   print(f"Original length: {len(data)}")      # 1000 samples
   print(f"Subset length: {len(subset)}")      # 100 samples  
   print(f"Number of modalities: {data.count}") # 3 modalities
   print(f"Modality names: {data.names}")      # ['features', 'labels', 'metadata']

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
               BlockShuffling(block_size=8, indices=train_idx),
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
               batch_size=None,
               num_workers=self.num_workers,
               prefetch_factor=self.train_dataset.fetch_factor + 1
           )
       
       def val_dataloader(self):
           return DataLoader(
               self.val_dataset,
               batch_size=None,
               num_workers=self.num_workers,
               prefetch_factor=self.val_dataset.fetch_factor + 1
           )

Tips and Best Practices
------------------------

1. **Choose appropriate block sizes**: Larger blocks (128-512) work better for sequential data access, smaller blocks (4-16) for more randomness.

2. **Use fetch_factor > 1** for better I/O efficiency, especially with slow storage.

3. **Set prefetch_factor = fetch_factor + 1** in DataLoader for optimal performance.

4. **For validation**, use ``Streaming`` strategy for deterministic results.

5. **Profile your pipeline** to find the optimal configuration for your specific data and hardware setup.
