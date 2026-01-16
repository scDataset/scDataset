Distributed & Parallel Training
================================

``scDataset`` provides native support for distributed and parallel training with 
PyTorch. This includes:

- **DistributedDataParallel (DDP)**: Multi-GPU, multi-node training
- **DataParallel (DP)**: Simple multi-GPU on single node
- **DataLoader multiprocessing**: ``num_workers`` for parallel data loading

How DDP Works
-------------

When running in a distributed environment, ``scDataset`` automatically partitions 
data across workers using round-robin assignment. Each worker (rank) processes 
only its assigned portion of the data, ensuring no overlap.

**Key features:**

- **Automatic rank detection**: Works with ``torch.distributed`` environment variables
- **Round-robin partitioning**: Even distribution of data across all ranks
- **Per-epoch shuffling**: Automatic epoch counter ensures different data ordering each epoch
- **Deterministic splits**: Same data partition for a given rank/world_size/seed

Basic DDP Setup
---------------

.. code-block:: python

    import os
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from scdataset import scDataset, BlockShuffling
    
    def setup_distributed():
        """Initialize distributed training."""
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return rank, world_size
    
    def main():
        rank, world_size = setup_distributed()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
        # Load your data source
        adata = load_adata()  # Your data loading function
        
        # Create scDataset - it automatically detects rank and world_size
        dataset = scDataset(
            adata,
            BlockShuffling(block_size=64),
            batch_size=128,
            fetch_callback=my_fetch_fn
        )
        
        # Create DataLoader (no DistributedSampler needed!)
        loader = DataLoader(
            dataset,
            batch_size=None,  # Batching handled by scDataset
            num_workers=2
        )
        
        # Standard DDP model setup
        model = YourModel().to(local_rank)
        model = DDP(model, device_ids=[local_rank])
        
        for epoch in range(num_epochs):
            # No set_epoch needed! Shuffling changes automatically each epoch.
            for batch in loader:
                batch = batch.to(local_rank)
                # Training code here
                pass
        
        dist.destroy_process_group()
    
    if __name__ == "__main__":
        main()

Manual Rank Configuration
-------------------------

You can also manually specify rank and world_size without relying on 
environment variables:

.. code-block:: python

    # Explicit rank configuration
    dataset = scDataset(
        adata,
        BlockShuffling(block_size=64),
        batch_size=128,
        fetch_callback=my_fetch_fn,
        rank=2,          # This worker's rank (0-indexed)
        world_size=4     # Total number of workers
    )

Data Partitioning
-----------------

``scDataset`` uses round-robin partitioning to distribute data across workers:

- **Rank 0**: Indices 0, 4, 8, 12, ...
- **Rank 1**: Indices 1, 5, 9, 13, ...
- **Rank 2**: Indices 2, 6, 10, 14, ...
- **Rank 3**: Indices 3, 7, 11, 15, ...

This ensures even distribution and allows each worker to process its portion 
independently without communication during data loading.

.. code-block:: python

    # Example: Understanding partitioning
    # With 1000 samples and world_size=4:
    # - Each rank processes ~250 samples
    # - Rank 0: samples at positions 0, 4, 8, ... (250 samples)
    # - Rank 1: samples at positions 1, 5, 9, ... (250 samples)
    # etc.

Automatic Epoch Handling
------------------------

.. versionadded:: 0.3.0

``scDataset`` automatically increments an internal epoch counter each time the 
dataset is iterated. This means different shuffling happens automatically each epoch.

.. code-block:: python

    # Different shuffling each epoch automatically.
    for epoch in range(100):
        for batch in loader:
            train_step(batch)
        # Epoch counter auto-increments when iteration completes

For reproducibility across runs, you can set a base seed in the constructor:

.. code-block:: python

    dataset = scDataset(adata, strategy, batch_size=128, seed=42)
    # Same seed = same shuffling sequence across runs

Launching Distributed Training
------------------------------

Use ``torchrun`` to launch distributed training:

**Single Node, Multiple GPUs:**

.. code-block:: bash

    torchrun --nproc_per_node=4 train.py

**Multiple Nodes:**

.. code-block:: bash

    # On node 0 (master):
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=<master_ip> --master_port=29500 train.py
    
    # On node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
        --master_addr=<master_ip> --master_port=29500 train.py

Complete Training Example
-------------------------

Here's a complete example with all components:

.. code-block:: python

    import os
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    import anndata as ad
    from scdataset import scDataset, BlockShuffling
    from scdataset.transforms import fetch_transform_adata
    
    def train():
        # Initialize distributed
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        
        # Load data (each rank loads independently)
        adata = ad.read_h5ad("large_dataset.h5ad", backed='r')
        
        # Create dataset - DDP handled automatically
        dataset = scDataset(
            adata,
            BlockShuffling(block_size=256),
            batch_size=512,
            fetch_callback=lambda d, idx: fetch_transform_adata(d[idx])
        )
        
        loader = DataLoader(dataset, batch_size=None, num_workers=4)
        
        # Model setup
        model = nn.Sequential(
            nn.Linear(adata.n_vars, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(local_rank)
        model = DDP(model, device_ids=[local_rank])
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop - shuffling changes automatically each epoch!
        for epoch in range(10):
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(local_rank)
                
                output = model(batch)
                loss = criterion(output, batch[:, :128])  # Reconstruction
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if batch_idx % 100 == 0 and local_rank == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        dist.destroy_process_group()
    
    if __name__ == "__main__":
        train()

Weighted Sampling with DDP
--------------------------

.. versionadded:: 0.3.0

One of ``scDataset``'s unique advantages is native support for **weighted sampling 
in distributed training scenarios**. This is a capability that PyTorch does not 
provide out of the box.

**The Problem with PyTorch's Approach**

PyTorch provides ``WeightedRandomSampler`` for handling class imbalance and 
``DistributedSampler`` for distributed training, but these two components do not 
work together natively. This limitation has been a persistent challenge in the 
PyTorch ecosystem:

- ``DistributedSampler`` partitions data deterministically across processes
- ``WeightedRandomSampler`` uses probabilistic multinomial sampling
- Combining them naively leads to incorrect behavior or overlapping data across GPUs

The community has produced various workarounds (e.g., ``DistributedSamplerWrapper``
from Catalyst), but these require careful seed management and have subtle edge cases.
As of January 2026, there is an `open PR in PyTorch <https://github.com/pytorch/pytorch/pull/150182>`_ 
to add ``DistributedWeightedRandomSampler``, but this has not yet been merged.

**scDataset's Solution**

With ``scDataset``, weighted sampling and distributed training are **decoupled by design**.
The sampling strategy operates on the data collection level, and DDP partitioning 
is applied automatically on top of that. This means you can use any strategy—including
``BlockWeightedSampling`` and ``ClassBalancedSampling``—without any special configuration:

.. code-block:: python

    import torch.distributed as dist
    from scdataset import scDataset, BlockWeightedSampling
    
    dist.init_process_group(backend="nccl")
    
    # Weighted sampling just works in DDP - no special wrappers needed!
    weights = compute_sample_weights(adata.obs['cell_type'])  # Your weights
    
    dataset = scDataset(
        adata,
        BlockWeightedSampling(
            weights=weights,
            total_size=10000,
            block_size=64
        ),
        batch_size=128,
        fetch_callback=my_fetch_fn
    )
    
    # Each GPU gets a different portion of the weighted-sampled data
    loader = DataLoader(dataset, batch_size=None, num_workers=4)

**Class-Balanced Sampling in DDP**

For the common use case of handling class imbalance, ``ClassBalancedSampling`` 
automatically computes weights inversely proportional to class frequencies:

.. code-block:: python

    from scdataset import scDataset, ClassBalancedSampling
    
    # Automatically balances rare cell types
    dataset = scDataset(
        adata,
        ClassBalancedSampling(
            label_key="cell_type",  # Column in adata.obs
            block_size=64,
            smoothing=0.1  # Prevents over-sampling of very rare classes
        ),
        batch_size=128,
        fetch_callback=my_fetch_fn
    )
    
    # Works seamlessly in DDP - rare classes are represented on all GPUs

**Why This Matters**

To the best of our knowledge, ``scDataset`` is among the first libraries to provide 
efficient on-disk weighted sampling that works natively with DDP. This is possible 
because:

1. **Strategy and distribution are decoupled**: The sampling strategy generates 
   indices, and DDP partitioning is applied afterward
2. **Deterministic seeding**: All ranks use the same base seed plus epoch offset,
   ensuring coordinated but different data across GPUs
3. **On-disk efficiency**: Unlike PyTorch's in-memory approach, ``scDataset`` reads
   only the needed samples from disk, making large-scale weighted sampling practical

DDP with Any Strategy
---------------------

``scDataset`` supports **any sampling strategy in DDP mode**. The distributed 
partitioning layer is orthogonal to the strategy layer:

.. list-table:: Strategy-DDP Compatibility
   :header-rows: 1
   :widths: 30 70
   
   * - Strategy
     - Description
   * - ``Streaming``
     - Simple sequential access, partitioned across GPUs
   * - ``BlockShuffling``
     - Block-shuffled access, each GPU gets different blocks
   * - ``BlockWeightedSampling``
     - Weighted sampling with DDP partitioning
   * - ``ClassBalancedSampling``
     - Class-balanced with automatic DDP support

.. code-block:: python

    # All strategies work the same way in DDP
    
    # Simple streaming
    dataset1 = scDataset(adata, Streaming(), ...)
    
    # Block shuffling
    dataset2 = scDataset(adata, BlockShuffling(block_size=64), ...)
    
    # Weighted sampling
    dataset3 = scDataset(adata, BlockWeightedSampling(weights=w), ...)
    
    # Class-balanced
    dataset4 = scDataset(adata, ClassBalancedSampling(label_key="ct"), ...)
    
    # All automatically partition data across GPUs when run with torchrun

Best Practices
--------------

1. **No DistributedSampler needed**: ``scDataset`` handles partitioning internally

2. **Use ``batch_size=None`` in DataLoader**: Batching is handled by ``scDataset``

3. **Backed mode for large files**: Use ``ad.read_h5ad(path, backed='r')`` to avoid 
   loading entire datasets into memory on each rank

4. **Same data across ranks**: Ensure all ranks can access the same data files

5. **Logging on rank 0 only**: Print/log only from rank 0 to avoid duplicate output

6. **Synchronize when needed**: Use ``dist.barrier()`` for synchronization points

7. **Set seed for reproducibility**: Use the ``seed`` parameter (e.g., ``seed=42``) 
   if you need identical shuffling sequences across different runs

8. **Weighted sampling works out of the box**: Use ``BlockWeightedSampling`` or 
   ``ClassBalancedSampling`` in DDP without any special configuration

DataLoader Multiprocessing (num_workers)
----------------------------------------

``scDataset`` works seamlessly with PyTorch DataLoader's ``num_workers`` parameter
for multiprocessing-based data loading. This enables parallel data fetching and
preprocessing while the GPU trains on the current batch.

**Basic Usage**

.. code-block:: python

    from torch.utils.data import DataLoader
    from scdataset import scDataset, BlockShuffling
    
    dataset = scDataset(
        data,
        BlockShuffling(block_size=64),
        batch_size=128
    )
    
    # Use num_workers for parallel data loading
    loader = DataLoader(
        dataset,
        batch_size=None,  # IMPORTANT: batching handled by scDataset
        num_workers=4,     # Use 4 worker processes
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    for batch in loader:
        # Training code here
        pass

**How It Works**

When ``num_workers > 0``, PyTorch spawns worker processes that load data in parallel:

1. Each worker independently iterates over ``scDataset``
2. Workers use PyTorch's automatic data splitting via ``get_worker_info()``
3. ``scDataset`` detects worker count and ID, partitioning data among workers
4. Batches are prefetched and queued for the main process

**Combining with DDP**

Both DDP partitioning and DataLoader multiprocessing work together:

.. code-block:: python

    # In a DDP training script launched with torchrun
    dist.init_process_group(backend="nccl")
    
    dataset = scDataset(
        adata,
        BlockShuffling(block_size=64),
        batch_size=128
        # rank and world_size auto-detected from torch.distributed
    )
    
    # Each DDP rank uses multiple workers
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4  # 4 workers PER GPU
    )
    
    # With 4 GPUs and 4 workers each, you have 16 worker processes total

**Choosing num_workers**

- Start with ``num_workers=2-4`` and increase if GPU utilization is low
- More workers = more memory usage (each loads data separately)
- For backed HDF5 files, each worker opens its own file handle
- Monitor CPU and memory usage to find the optimal value

**Important Notes**

- Always use ``batch_size=None`` in DataLoader (batching is handled by scDataset)
- For backed AnnData, each worker opens its own file handle
- The ``seed`` parameter ensures reproducibility across workers

DataParallel (DP) Support
-------------------------

For simpler single-node multi-GPU setups, you can use PyTorch's ``DataParallel``
instead of ``DistributedDataParallel``. This is easier to set up but less efficient
than DDP for large-scale training.

.. code-block:: python

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from scdataset import scDataset, BlockShuffling
    
    # DataParallel doesn't need distributed setup
    # scDataset works normally without rank/world_size
    dataset = scDataset(
        data,
        BlockShuffling(block_size=64),
        batch_size=128
    )
    
    loader = DataLoader(dataset, batch_size=None, num_workers=4)
    
    # Wrap model with DataParallel
    model = nn.Sequential(
        nn.Linear(2000, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    
    for batch in loader:
        batch = batch.cuda()  # DataParallel handles GPU distribution
        output = model(batch)
        # Training code here

**When to Use DataParallel vs DDP**

.. list-table::
   :header-rows: 1
   :widths: 20 40 40
   
   * - 
     - DataParallel (DP)
     - DistributedDataParallel (DDP)
   * - Setup
     - Simple, no ``torchrun`` needed
     - Requires distributed initialization
   * - Efficiency
     - Slower due to GIL and data copying
     - Faster, recommended for production
   * - Multi-node
     - No
     - Yes
   * - Data handling
     - Single DataLoader, GPU scatter
     - Per-GPU DataLoader, no scatter
   * - scDataset usage
     - Normal (no rank/world_size)
     - Auto-detected rank/world_size

**Recommendation**: Use DDP for serious training workloads. DataParallel is 
acceptable for quick experiments or when simplicity is more important than
maximum throughput.

Further Reading
---------------

- `PyTorch DDP Tutorial <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_
- `PyTorch DistributedSampler Documentation <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`_
- `PyTorch DataLoader num_workers <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`_
- `PyTorch Issue #77154 <https://github.com/pytorch/pytorch/issues/77154>`_ - Feature request for DistributedWeightedRandomSampler
- `PyTorch PR #150182 <https://github.com/pytorch/pytorch/pull/150182>`_ - Proposed DistributedWeightedRandomSampler (not yet merged)
