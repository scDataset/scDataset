Changelog
=========

[0.2.0] - 2025-08-28
---------------------

**Breaking Changes**
~~~~~~~~~~~~~~~~~~~~

* **Completely redesigned API**: scDataset now uses a strategy-based sampling approach instead of modes
* **Constructor changes**: ``scDataset(data_collection, strategy, batch_size, ...)`` replaces old ``scDataset(data_collection, batch_size, ...)``
* **Removed methods**: ``subset()``, ``set_mode()``, ``block_size``, ``fetch_factor`` parameters moved to strategies
* **New required parameter**: ``strategy`` - must provide a ``SamplingStrategy`` instance

**Added**
~~~~~~~~~

* **Strategy-based sampling system**:
  
  * ``SamplingStrategy`` - Abstract base class for all sampling strategies
  * ``Streaming`` - Sequential sampling with optional buffer-level shuffling
  * ``BlockShuffling`` - Block-based shuffling for data locality while maintaining randomization
  * ``BlockWeightedSampling`` - Weighted sampling with configurable block sizes and replacement options
  * ``ClassBalancedSampling`` - Automatic class balancing for imbalanced datasets

* **``MultiIndexable`` class** - Container for multi-modal data with synchronized indexing:
  
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