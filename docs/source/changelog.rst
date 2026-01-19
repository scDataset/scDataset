Changelog
=========

[0.3.0] - 2025-01-16
---------------------

**Major Features**
~~~~~~~~~~~~~~~~~~

* **Native DDP support**: Full Distributed Data Parallel support with round-robin
  fetch distribution across ranks. Auto-detects ``torch.distributed`` settings.
  
  * Weighted sampling works with DDP (first implementation to our knowledge)
  * All sampling strategies (Streaming, BlockShuffling, BlockWeightedSampling, 
    ClassBalancedSampling) work seamlessly with DDP
  * No ``DistributedSampler`` needed - partitioning handled internally

* **Built-in transform functions** (``transforms.py``):
  
  * ``fetch_transform_adata()`` - Transform AnnData/AnnCollection to MultiIndexable
  * ``fetch_transform_hf()`` - Convert HuggingFace sparse data to dense tensors

* **Auto-configuration module** (``scdataset.experimental.auto_config``):
  
  * ``suggest_parameters()`` - Automatically suggest optimal ``num_workers``, 
    ``fetch_factor``, and ``block_size`` based on data and system resources

* **Training experiments module** (``training_experiments/``): Comprehensive framework
  for benchmarking data loading strategies on the Tahoe-100M dataset:
  
  * 6 data loading strategies compared: Streaming, Streaming with Buffer, Block
    Shuffling (block_size=4), Random Sampling (block_size=1), Block Weighted
    Sampling (block_size=4), True Weighted Sampling (block_size=1)
  * 4 classification tasks: Cell line, Drug, MOA broad, MOA fine
  * Linear model to avoid confounding effects from model selection
  * Balanced weight computation with min_count_baseline to prevent extreme
    reweighting of rare (cell_line, drug) combinations

**Bug Fixes**
~~~~~~~~~~~~~

* **Fixed unsorted indices issue**: Sampling strategies now automatically sort indices
  to ensure optimal I/O performance. When unsorted indices are provided, a warning
  is issued and indices are sorted automatically. This fix addresses issues with
  disk access patterns that could occur when users passed indices in arbitrary order.
  (Thanks to `@deto <https://github.com/deto>`_ for reporting this issue)

* **Fixed ClassBalancedSampling with indices**: Class-balanced sampling now correctly
  handles subset indices by computing weights only for the specified subset.
  Previously, when ``indices`` was provided, weights could mismatch the subset size.

* **Fixed BlockWeightedSampling weights handling with indices**: When both ``weights``
  and ``indices`` are provided, weights are now properly aligned with the subset.
  Supports both full weights (matching data_collection) that get subsetted, and
  pre-subsetted weights (matching indices length).

**Added**
~~~~~~~~~

* **Unstructured data support in MultiIndexable**:
  
  * New ``unstructured`` parameter to store non-indexable metadata
  * Useful for storing gene names, dataset info, or other metadata
  * Unstructured data is preserved through subsetting operations
  * New ``unstructured_keys`` property to list available keys

* **Jupyter notebook integration in documentation**:
  
  * Added ``nbsphinx`` extension for including Jupyter notebooks in docs
  * Tutorial notebook (``tahoe_tutorial.ipynb``) now available in docs
  * Notebooks are rendered without execution for faster builds

* **Doctest integration for documentation examples**:
  
  * Added ``sphinx.ext.doctest`` extension for testing code blocks in docs
  * New ``tests/test_docstrings.py`` for testing module docstrings
  * Documentation code examples are now automatically tested

* **Comprehensive test suite**:
  
  * Tests for all strategies, MultiIndexable, scDataset, and auto_config
  * Tests for dict-like interface (items, keys, values) in MultiIndexable
  * Tests for error handling and edge cases
  * Tests for doc code snippets from quickstart guide

* **Documentation improvements**:
  
  * New transforms guide (``transforms.rst``) documenting ``fetch_transform``,
    ``batch_transform``, ``fetch_callback``, and ``batch_callback``
  * Comprehensive AnnCollection example in examples
  * Documentation badge added to README and docs
  * Updated benchmarks README with utility documentation

**Changed**
~~~~~~~~~~~

* **Added BlockWeightedSampling to benchmarks**: Now evaluates all sampling strategies

**Dependencies**
~~~~~~~~~~~~~~~~

* Added optional ``[auto]`` extras for auto-configuration: ``pip install scDataset[auto]``
* Added optional ``[docs]`` extras for documentation building: ``pip install scDataset[docs]``
* Added ``[dev]`` extras for development: ``pip install scDataset[dev]``

[0.2.0] - 2025-08-28
---------------------

**Breaking Changes**
~~~~~~~~~~~~~~~~~~~~

* **Completely redesigned API**: scDataset now uses a strategy-based sampling approach instead of modes
* **Constructor changes**: ``scDataset(data_collection, strategy, batch_size, ...)`` replaces old ``scDataset(data_collection, batch_size, ...)``
* **New required parameter**: ``strategy`` - must provide a ``SamplingStrategy`` instance
* ``block_size`` parameter moved to strategies
* **Removed methods**: ``subset()``, ``set_mode()``

**Added**
~~~~~~~~~

* **Strategy-based sampling system**:
  
  * ``SamplingStrategy`` - Abstract base class for all sampling strategies
  * ``Streaming`` - Sequential sampling with optional buffer-level shuffling
  * ``BlockShuffling`` - Block-based shuffling for data locality while maintaining randomization
  * ``BlockWeightedSampling`` - Weighted sampling with configurable block sizes and replacement options
  * ``ClassBalancedSampling`` - Automatic class balancing for imbalanced datasets

* **MultiIndexable class** - Container for multi-modal data with synchronized indexing:
  
  * Supports multiple indexable objects (arrays, lists, etc.) that are indexed together
  * Named and positional access to contained indexables  
  * Useful for gene expression + protein data, features + labels, etc.

**Migration Guide**
~~~~~~~~~~~~~~~~~~~

**Old v0.1.x syntax**::

    # v0.1.x - No longer supported
    dataset = scDataset(data, batch_size=64, block_size=8, fetch_factor=4)
    dataset.subset(train_indices)
    dataset.set_mode('train')

**New v0.2.0 syntax**::

    # v0.2.0 - Strategy-based approach
    from scdataset import scDataset, BlockShuffling
    
    strategy = BlockShuffling(block_size=8, indices=train_indices)
    dataset = scDataset(data, strategy, batch_size=64, fetch_factor=4)