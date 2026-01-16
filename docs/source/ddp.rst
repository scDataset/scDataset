Distributed Training
====================

``scDataset`` provides native support for distributed training with PyTorch's 
DistributedDataParallel (DDP). This enables efficient scaling across multiple 
GPUs and nodes.

How It Works
------------

When running in a distributed environment, ``scDataset`` automatically partitions 
data across workers using round-robin assignment. Each worker (rank) processes 
only its assigned portion of the data, ensuring no overlap.

**Key features:**

- **Automatic rank detection**: Works with ``torch.distributed`` environment variables
- **Round-robin partitioning**: Even distribution of data across all ranks
- **Per-epoch shuffling**: Call ``set_epoch()`` to ensure different data ordering each epoch
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
            # IMPORTANT: Call set_epoch for proper shuffling
            dataset.set_epoch(epoch)
            
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

The ``set_epoch()`` Method
--------------------------

When using shuffling strategies with DDP, call ``set_epoch(epoch)`` at the start 
of each epoch. This ensures:

1. Different random permutations each epoch
2. Consistent ordering across workers for the same epoch
3. Reproducible results with the same seed

.. code-block:: python

    for epoch in range(100):
        dataset.set_epoch(epoch)  # New shuffling for each epoch
        for batch in loader:
            train_step(batch)

Without ``set_epoch()``, the same data order would be used every epoch, which 
can harm model convergence.

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
        
        # Training loop
        for epoch in range(10):
            dataset.set_epoch(epoch)  # Important for shuffling
            
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

Best Practices
--------------

1. **Always call ``set_epoch()``**: Essential for proper shuffling across epochs

2. **No DistributedSampler needed**: ``scDataset`` handles partitioning internally

3. **Use ``batch_size=None`` in DataLoader**: Batching is handled by ``scDataset``

4. **Backed mode for large files**: Use ``ad.read_h5ad(path, backed='r')`` to avoid 
   loading entire datasets into memory on each rank

5. **Same data across ranks**: Ensure all ranks can access the same data files

6. **Logging on rank 0 only**: Print/log only from rank 0 to avoid duplicate output

7. **Synchronize when needed**: Use ``dist.barrier()`` for synchronization points
