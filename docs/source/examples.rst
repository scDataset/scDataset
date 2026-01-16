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

Working with AnnCollection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AnnCollection`` from ``anndata.experimental`` allows you to lazily concatenate multiple AnnData 
objects without loading all data into memory. This is particularly useful for large-scale 
single-cell studies spanning multiple files (e.g., different patients, batches, or plates).

**Basic AnnCollection Setup:**

.. code-block:: python

    import anndata as ad
    from anndata.experimental import AnnCollection
    import numpy as np
    import scipy.sparse as sp
    from functools import partial
    from scdataset import scDataset, BlockShuffling, MultiIndexable
    from torch.utils.data import DataLoader

    # Load multiple AnnData files (backed mode for memory efficiency)
    file_paths = [
        "plate_1.h5ad",
        "plate_2.h5ad", 
        "plate_3.h5ad"
    ]
    
    adatas = [ad.read_h5ad(f, backed='r') for f in file_paths]
    
    # Create AnnCollection for lazy concatenation
    collection = AnnCollection(adatas)
    
    print(f"Total cells: {len(collection)}")
    print(f"Number of genes: {collection.shape[1]}")

**Complete Pipeline with AnnCollection:**

.. code-block:: python

    import torch
    
    def fetch_transform_adata(batch, columns=None):
        """
        Transform AnnData batch to MultiIndexable.
        
        This function handles backed AnnData (lazy loading) by materializing
        the data into memory and converting sparse matrices to dense.
        
        Parameters
        ----------
        batch : AnnData
            The fetched AnnData slice from AnnCollection
        columns : list of str, optional
            Observation columns to include in the output
            
        Returns
        -------
        MultiIndexable
            Contains 'X' (expression matrix) and any requested columns
        """
        # Materialize backed data into memory
        batch = batch.to_memory()
        
        # Get expression matrix
        X = batch.X
        if sp.issparse(X):
            X = X.toarray()
        
        # Build output dictionary
        data_dict = {'X': X}
        if columns is not None:
            for col in columns:
                data_dict[col] = batch.obs[col].values
        
        return MultiIndexable(data_dict)
    
    def to_tensor_batch(batch):
        """Convert batch to PyTorch tensors with normalization."""
        X = torch.from_numpy(batch['X']).float()
        
        # Log normalize
        X = torch.log1p(X)
        
        # Z-score normalize per gene
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
        
        # Convert plate labels to integers
        plate = batch['plate']
        plate_to_idx = {'plate_1': 0, 'plate_2': 1, 'plate_3': 2}
        y = torch.tensor([plate_to_idx[p] for p in plate]).long()
        
        return X, y
    
    # Create dataset with transforms
    dataset = scDataset(
        collection,
        strategy=BlockShuffling(block_size=8),
        batch_size=64,
        fetch_factor=16,
        fetch_transform=partial(fetch_transform_adata, columns=['plate']),
        batch_transform=to_tensor_batch
    )
    
    # Create optimized DataLoader
    loader = DataLoader(
        dataset,
        batch_size=None,        # scDataset handles batching
        num_workers=4,
        prefetch_factor=17,     # fetch_factor + 1
        pin_memory=True         # For GPU training
    )
    
    # Training loop
    for X, y in loader:
        # X: (64, n_genes) normalized tensor
        # y: (64,) plate labels
        print(f"Batch X: {X.shape}, y: {y.shape}")
        break

**Train/Validation Split with AnnCollection:**

.. code-block:: python

    from sklearn.model_selection import train_test_split
    
    # Get total number of cells
    n_cells = len(collection)
    indices = np.arange(n_cells)
    
    # Split indices
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=42
    )
    
    # Common transform function
    fetch_fn = partial(fetch_transform_adata, columns=['cell_type', 'plate'])
    
    # Training dataset with shuffling
    train_dataset = scDataset(
        collection,
        BlockShuffling(indices=train_idx, block_size=8),
        batch_size=64,
        fetch_factor=16,
        fetch_transform=fetch_fn
    )
    
    # Validation dataset with streaming (deterministic)
    val_dataset = scDataset(
        collection,
        Streaming(indices=val_idx),
        batch_size=64,
        fetch_factor=16,
        fetch_transform=fetch_fn
    )
    
    train_loader = DataLoader(train_dataset, batch_size=None, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=4)

**Memory-Efficient Tips for AnnCollection:**

1. **Use backed mode**: Always load AnnData files with ``backed='r'`` to avoid loading entire files into memory.

2. **Use fetch_transform**: Materialize data in ``fetch_transform`` rather than ``batch_transform`` to benefit from larger fetches.

3. **Higher fetch_factor**: For backed data, use ``fetch_factor=16`` or higher to amortize I/O overhead.

4. **Block shuffling**: Use ``BlockShuffling`` with appropriate block size to balance randomness vs I/O efficiency.

5. **Common genes**: Use ``join_vars='inner'`` when creating ``AnnCollection`` to ensure all files have the same features.

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
        BlockShuffling(block_size=8),
        batch_size=64,
        batch_callback=extract_hf_batch,
        batch_transform=process_hf_batch
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
