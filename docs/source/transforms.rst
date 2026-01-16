Data Transforms and Callbacks
=============================

``scDataset`` provides a flexible data transformation pipeline through four hook points:
``fetch_callback``, ``fetch_transform``, ``batch_callback``, and ``batch_transform``.
Understanding when and how to use each one is key to efficient data loading.

.. contents:: Table of Contents
   :local:
   :depth: 2

Built-in Transform Functions
----------------------------

``scDataset`` includes pre-built transform functions for common use cases:

.. code-block:: python

    from scdataset.transforms import fetch_transform_adata, fetch_transform_hf

**fetch_transform_adata**
    Transforms an AnnData batch into a :class:`~scdataset.MultiIndexable` object.
    Handles sparse matrices, backed data materialization, and optional observation columns.

**fetch_transform_hf**
    Converts HuggingFace sparse gene expression data to dense tensors or 
    :class:`~scdataset.MultiIndexable` objects.

See the :ref:`examples below <builtin-transform-examples>` for detailed usage.

Overview
--------

The data loading pipeline in ``scDataset`` follows this flow:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────────────┐
    │                      scDataset Pipeline                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Strategy generates indices                                  │
    │           ↓                                                     │
    │  2. fetch_callback(collection, indices) → raw_data              │
    │     [or default: collection[indices]]                           │
    │           ↓                                                     │
    │  3. fetch_transform(raw_data) → transformed_data                │
    │     [Applied to ENTIRE fetch: batch_size × fetch_factor]        │
    │           ↓                                                     │
    │  4. batch_callback(transformed_data, batch_indices) → batch     │
    │     [or default: transformed_data[batch_indices]]               │
    │           ↓                                                     │
    │  5. batch_transform(batch) → final_batch                        │
    │     [Applied to EACH batch: batch_size samples]                 │
    │           ↓                                                     │
    │  6. yield final_batch                                           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

The Four Hook Points
--------------------

fetch_callback
~~~~~~~~~~~~~~

**Purpose**: Custom function to fetch data from the collection using indices.

**Signature**: ``(data_collection, indices) -> fetched_data``

**When to use**:

- When your data collection doesn't support standard indexing
- When you need special handling for batch vs single indexing
- When working with custom data formats (e.g., BioNeMo sparse matrices)

**Default behavior**: ``data_collection[indices]``

**Example** - Custom fetch for a database:

.. code-block:: python

    def fetch_from_database(db_connection, indices):
        """Fetch rows from a database by indices."""
        query = f"SELECT * FROM data WHERE id IN ({','.join(map(str, indices))})"
        return db_connection.execute(query).fetchall()
    
    dataset = scDataset(
        db_connection,
        Streaming(),
        batch_size=64,
        fetch_callback=fetch_from_database
    )

**Example** - BioNeMo sparse matrices:

.. code-block:: python

    from bionemo.scdl.util.torch_dataloader_utils import collate_sparse_matrix_batch
    
    def fetch_callback_bionemo(data_collection, idx):
        """Handle BioNeMo's sparse matrix format."""
        if isinstance(idx, int):
            return collate_sparse_matrix_batch([data_collection[idx]]).to_dense()
        else:
            batch = [data_collection[int(i)] for i in idx]
            return collate_sparse_matrix_batch(batch).to_dense()
    
    dataset = scDataset(
        bionemo_data,
        BlockShuffling(),
        batch_size=64,
        fetch_callback=fetch_callback_bionemo
    )

fetch_transform
~~~~~~~~~~~~~~~

**Purpose**: Transform data after fetching, before splitting into batches.

**Signature**: ``(fetched_data) -> transformed_data``

**When to use**:

- Converting data formats (e.g., AnnData to numpy)
- Materializing lazy/backed data into memory
- Operations that are more efficient on larger chunks
- Creating MultiIndexable objects for downstream indexing

**Applied to**: Entire fetch (``batch_size × fetch_factor`` samples)

.. _builtin-transform-examples:

Built-in: fetch_transform_adata
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`~scdataset.transforms.fetch_transform_adata` for AnnData and AnnCollection:

.. code-block:: python

    from scdataset import scDataset, BlockShuffling
    from scdataset.transforms import fetch_transform_adata
    from functools import partial
    
    # Basic usage - returns MultiIndexable with 'X' key
    dataset = scDataset(
        ann_collection,
        BlockShuffling(),
        batch_size=64,
        fetch_transform=fetch_transform_adata
    )
    
    # With observation columns
    fetch_fn = partial(fetch_transform_adata, columns=['cell_type', 'batch'])
    dataset = scDataset(
        ann_collection,
        BlockShuffling(),
        batch_size=64,
        fetch_transform=fetch_fn
    )

Built-in: fetch_transform_hf
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :func:`~scdataset.transforms.fetch_transform_hf` for HuggingFace sparse datasets:

.. code-block:: python

    from scdataset import scDataset, Streaming
    from scdataset.transforms import fetch_transform_hf
    from functools import partial
    
    # Returns dense tensor (default)
    dataset = scDataset(
        hf_dataset,
        Streaming(),
        batch_size=64,
        fetch_transform=fetch_transform_hf
    )
    
    # With custom gene count
    fetch_fn = partial(fetch_transform_hf, num_genes=62713)
    
    # With dict output format for multi-modal data
    fetch_fn = partial(
        fetch_transform_hf, 
        output_format='dict', 
        obs_columns=['cell_type', 'batch']
    )

Custom Transform Example
^^^^^^^^^^^^^^^^^^^^^^^^

For custom data formats, write your own transform:

.. code-block:: python

    import scipy.sparse as sp
    from scdataset import MultiIndexable
    
    def custom_fetch_transform(batch, columns=None):
        """Transform custom batch to MultiIndexable."""
        batch = batch.to_memory()  # Materialize if backed
        
        X = batch.X
        if sp.issparse(X):
            X = X.toarray()
        
        data_dict = {'X': X}
        if columns is not None:
            for col in columns:
                data_dict[col] = batch.obs[col].values
        
        return MultiIndexable(data_dict)

batch_callback
~~~~~~~~~~~~~~

**Purpose**: Custom function to extract a batch from transformed data.

**Signature**: ``(transformed_data, batch_indices) -> batch``

**When to use**:

- When transformed data doesn't support standard indexing
- When you need special slicing logic
- Rarely needed if fetch_transform produces indexable output

**Default behavior**: ``transformed_data[batch_indices]``

**Example** - Custom batch extraction:

.. code-block:: python

    def custom_batch_callback(data, indices):
        """Extract batch with custom logic."""
        # Maybe data is a custom container
        return {
            'features': data.get_features(indices),
            'labels': data.get_labels(indices)
        }
    
    dataset = scDataset(
        custom_data,
        Streaming(),
        batch_size=64,
        batch_callback=custom_batch_callback
    )

batch_transform
~~~~~~~~~~~~~~~

**Purpose**: Transform each individual batch before yielding.

**Signature**: ``(batch) -> transformed_batch``

**When to use**:

- Normalization per batch
- Data augmentation
- Converting to model-ready format
- Adding noise or other batch-level operations

**Applied to**: Each individual batch (``batch_size`` samples)

**Example** - Normalization and augmentation:

.. code-block:: python

    import torch
    
    def batch_transform(batch):
        """Normalize and augment batch."""
        X, y = batch['X'], batch['labels']
        
        # Convert to tensor if needed
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        
        # Normalize per sample
        X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-8)
        
        # Add small noise for regularization (training only)
        if training:
            X = X + torch.randn_like(X) * 0.01
        
        return X, torch.from_numpy(y).long()
    
    dataset = scDataset(
        data,
        BlockShuffling(),
        batch_size=64,
        batch_transform=batch_transform
    )

**Example** - Log transformation for gene expression:

.. code-block:: python

    import numpy as np
    
    def log_transform(batch):
        """Apply log1p transformation to gene expression."""
        return np.log1p(batch)
    
    dataset = scDataset(
        gene_expression_data,
        Streaming(),
        batch_size=64,
        batch_transform=log_transform
    )

Complete Pipeline Example
-------------------------

Here's a complete example showing all four hooks working together:

.. code-block:: python

    from scdataset import scDataset, BlockShuffling, MultiIndexable
    from torch.utils.data import DataLoader
    from functools import partial
    import torch
    import scipy.sparse as sp
    
    # fetch_callback: Custom indexing for AnnCollection
    def ann_fetch_callback(collection, indices):
        """Fetch from AnnCollection with sorted indices for better I/O."""
        # scDataset already sorts indices, but we could add logging here
        return collection[indices]
    
    # fetch_transform: Convert to MultiIndexable
    def ann_fetch_transform(batch, columns=None):
        """Materialize AnnData and create MultiIndexable."""
        batch = batch.to_memory()
        X = batch.X
        if sp.issparse(X):
            X = X.toarray()
        
        data_dict = {'X': X}
        if columns:
            for col in columns:
                data_dict[col] = batch.obs[col].values
        
        return MultiIndexable(data_dict)
    
    # batch_callback: Default is fine for MultiIndexable
    # (MultiIndexable supports standard indexing)
    
    # batch_transform: Normalize and convert to tensors
    def to_tensor_batch(batch):
        """Convert batch to training-ready tensors."""
        X = torch.from_numpy(batch['X']).float()
        y = torch.from_numpy(batch['cell_type']).long()
        
        # Log normalize
        X = torch.log1p(X)
        
        # Z-score normalize per gene (column)
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-8)
        
        return X, y
    
    # Create dataset with all transforms
    dataset = scDataset(
        ann_collection,
        strategy=BlockShuffling(block_size=8),
        batch_size=64,
        fetch_factor=16,
        fetch_callback=ann_fetch_callback,
        fetch_transform=partial(ann_fetch_transform, columns=['cell_type']),
        batch_transform=to_tensor_batch
    )
    
    # Use with DataLoader
    loader = DataLoader(
        dataset,
        batch_size=None,  # scDataset handles batching
        num_workers=4,
        prefetch_factor=17  # fetch_factor + 1
    )
    
    # Training loop
    for X, y in loader:
        # X: (64, n_genes) normalized tensor
        # y: (64,) cell type labels
        loss = model(X, y)
        ...

Best Practices
--------------

1. **Use fetch_transform for heavy operations**: Operations like densifying sparse 
   matrices or loading data from disk are more efficient when applied to larger 
   chunks (entire fetch) rather than individual batches.

2. **Use batch_transform for sample-wise operations**: Normalization, augmentation, 
   and format conversion that operate per-sample belong in batch_transform.

3. **Return indexable objects from fetch_transform**: If you use fetch_transform, 
   make sure it returns something that can be indexed (numpy array, tensor, 
   MultiIndexable, etc.) unless you also provide a custom batch_callback.

4. **Use MultiIndexable for multi-modal data**: When your transform creates multiple 
   outputs (X, y, metadata), wrap them in MultiIndexable for synchronized indexing.

5. **Profile your transforms**: Use Python's ``cProfile`` or ``line_profiler`` to 
   ensure transforms aren't bottlenecks.

Common Use Cases
----------------

+------------------------+------------------+-------------------+
| Use Case               | fetch_transform  | batch_transform   |
+========================+==================+===================+
| Densify sparse matrix  | ✓                |                   |
+------------------------+------------------+-------------------+
| Load backed AnnData    | ✓                |                   |
+------------------------+------------------+-------------------+
| Per-sample normalize   |                  | ✓                 |
+------------------------+------------------+-------------------+
| Data augmentation      |                  | ✓                 |
+------------------------+------------------+-------------------+
| Convert to tensor      |                  | ✓                 |
+------------------------+------------------+-------------------+
| Add labels from obs    | ✓                |                   |
+------------------------+------------------+-------------------+
| Log transformation     |                  | ✓                 |
+------------------------+------------------+-------------------+

See Also
--------

* :doc:`examples` - More complete examples
* :doc:`quickstart` - Getting started guide
* :doc:`scdataset` - API reference
