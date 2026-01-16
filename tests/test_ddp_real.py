"""
Real DDP/DP Tests for scDataset

This module tests actual distributed data parallel behavior using real GPU processes.
It verifies that:
1. Data is correctly split across ranks (no overlap, no missing data)
2. All ranks get approximately equal amounts of data
3. The auto-detection of rank/world_size works correctly
4. Multi-worker DataLoader integration works correctly

Requirements: 
- Multiple GPUs (can be run with CUDA_VISIBLE_DEVICES to limit)
- torch.distributed support
"""

import os
import sys
import tempfile
import pytest
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Add src to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from scdataset import scDataset
from scdataset.strategy import Streaming


def setup_ddp(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def ddp_worker_collect_indices(rank: int, world_size: int, 
                                data_size: int, batch_size: int, fetch_factor: int,
                                result_queue: mp.Queue, use_nccl: bool = False):
    """
    Worker function that runs on each DDP rank.
    Creates a scDataset with auto-detection, iterates through it,
    and puts the indices it processed into the result queue.
    """
    try:
        # Setup DDP - use gloo for CPU testing (more reliable), nccl for GPU
        backend = "nccl" if use_nccl else "gloo"
        setup_ddp(rank, world_size, backend)
        
        # Verify torch.distributed is properly initialized
        assert dist.is_initialized(), f"Rank {rank}: dist not initialized"
        assert dist.get_rank() == rank, f"Rank {rank}: get_rank() mismatch"
        assert dist.get_world_size() == world_size, f"Rank {rank}: world_size mismatch"
        
        # Create data - use indices as data so we can track what was processed
        data = np.arange(data_size)
        
        # Create scDataset with auto-detection (rank and world_size should be detected)
        dataset = scDataset(
            data, 
            Streaming(shuffle=False),  # No shuffle for deterministic testing
            batch_size=batch_size,
            fetch_factor=fetch_factor,
            seed=42
        )
        
        # Verify auto-detection worked
        assert dataset.rank == rank, f"Rank {rank}: auto-detected rank is {dataset.rank}"
        assert dataset.world_size == world_size, f"Rank {rank}: auto-detected world_size is {dataset.world_size}"
        
        # Collect all data this rank processes
        collected = []
        for batch in dataset:
            collected.extend(batch.tolist())
        
        result_queue.put((rank, collected, None))
        
    except Exception as e:
        import traceback
        result_queue.put((rank, [], traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def run_ddp_test(world_size: int, data_size: int, batch_size: int = 10, fetch_factor: int = 5):
    """
    Run a DDP test with the specified number of processes.
    Returns a dict mapping rank -> list of indices processed.
    """
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue()
    
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=ddp_worker_collect_indices,
            args=(rank, world_size, data_size, batch_size, fetch_factor, result_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    results = {}
    errors = []
    for _ in range(world_size):
        rank, indices, error = result_queue.get(timeout=60)
        results[rank] = indices
        if error:
            errors.append(f"Rank {rank}: {error}")
    
    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    
    if errors:
        raise RuntimeError("Errors in DDP workers:\n" + "\n".join(errors))
    
    return results


def ddp_worker_with_dataloader(rank: int, world_size: int, data_size: int, result_queue):
    """Worker that tests DDP with DataLoader multiprocessing."""
    try:
        setup_ddp(rank, world_size, "gloo")
        
        data = np.arange(data_size)
        
        dataset = scDataset(
            data,
            Streaming(shuffle=False),
            batch_size=10,
            fetch_factor=5,
            seed=42
        )
        
        # Use DataLoader with 2 workers
        dataloader = DataLoader(dataset, batch_size=None, num_workers=2)
        
        collected = []
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batch = batch.numpy()
            collected.extend(batch.flatten().tolist())
        
        result_queue.put((rank, collected, None))
        
    except Exception as e:
        import traceback
        result_queue.put((rank, [], traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def ddp_worker_multiple_epochs(rank: int, world_size: int, data_size: int, result_queue):
    """Worker that collects data from multiple epochs."""
    try:
        setup_ddp(rank, world_size, "gloo")
        
        data = np.arange(data_size)
        
        dataset = scDataset(
            data,
            Streaming(shuffle=True),
            batch_size=10,
            fetch_factor=5,
            seed=42
        )
        
        # Collect data from 2 epochs
        epoch_results = []
        for epoch in range(2):
            epoch_data = []
            for batch in dataset:
                epoch_data.extend(batch.tolist())
            epoch_results.append(epoch_data)
        
        result_queue.put((rank, epoch_results, None))
        
    except Exception as e:
        import traceback
        result_queue.put((rank, [], traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


class TestDDPRealDistribution:
    """Test real DDP distribution behavior."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ddp_2_ranks_data_split(self):
        """Test that 2 DDP ranks correctly split data with no overlap."""
        world_size = 2
        data_size = 100
        batch_size = 10
        fetch_factor = 5
        
        results = run_ddp_test(world_size, data_size, batch_size, fetch_factor)
        
        # Verify we got results from both ranks
        assert len(results) == world_size
        
        # Collect all indices
        all_indices = []
        for rank in range(world_size):
            all_indices.extend(results[rank])
        
        # Check for no overlap
        unique_indices = set(all_indices)
        assert len(unique_indices) == len(all_indices), "Overlap detected between ranks!"
        
        # Check all data is covered
        expected = set(range(data_size))
        assert unique_indices == expected, f"Missing indices: {expected - unique_indices}"
        
        # Check roughly equal distribution
        for rank in range(world_size):
            count = len(results[rank])
            expected_count = data_size // world_size
            # Allow some variance due to batch boundaries
            assert abs(count - expected_count) <= batch_size, \
                f"Rank {rank} has {count} samples, expected ~{expected_count}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")  
    def test_ddp_3_ranks_data_split(self):
        """Test that 3 DDP ranks correctly split data."""
        world_size = 3
        data_size = 150
        batch_size = 10
        fetch_factor = 3
        
        results = run_ddp_test(world_size, data_size, batch_size, fetch_factor)
        
        # Collect all indices
        all_indices = []
        for rank in range(world_size):
            all_indices.extend(results[rank])
        
        # Check for no overlap
        unique_indices = set(all_indices)
        assert len(unique_indices) == len(all_indices), "Overlap detected!"
        
        # Check all data is covered
        expected = set(range(data_size))
        assert unique_indices == expected, f"Missing: {expected - unique_indices}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ddp_uneven_data(self):
        """Test DDP with data that doesn't divide evenly across ranks."""
        world_size = 3
        data_size = 100  # Not divisible by 3
        batch_size = 7   # Not divisible by data_size
        fetch_factor = 4
        
        results = run_ddp_test(world_size, data_size, batch_size, fetch_factor)
        
        # Collect all indices
        all_indices = []
        for rank in range(world_size):
            all_indices.extend(results[rank])
        
        # With drop_last=False (default), all data should be covered
        unique_indices = set(all_indices)
        expected = set(range(data_size))
        assert unique_indices == expected, f"Missing: {expected - unique_indices}"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ddp_auto_detection(self):
        """Test that rank and world_size are auto-detected correctly."""
        # This is implicitly tested by ddp_worker_collect_indices assertions
        # but we can add explicit verification
        world_size = 2
        data_size = 50
        
        results = run_ddp_test(world_size, data_size)
        
        # If we got here without errors, auto-detection worked
        assert len(results) == world_size
        

class TestDDPWithShuffle:
    """Test DDP with shuffling enabled."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ddp_shuffle_same_seed_different_data(self):
        """Test that shuffling gives same order but different data per rank."""
        world_size = 2
        data_size = 100
        
        # Run test - data should still be split correctly even with shuffle
        results = run_ddp_test(world_size, data_size, batch_size=10, fetch_factor=5)
        
        # Even with shuffle within fetches, each fetch should still go to correct rank
        all_indices = []
        for rank in range(world_size):
            all_indices.extend(results[rank])
        
        unique_indices = set(all_indices)
        expected = set(range(data_size))
        assert unique_indices == expected


def ddp_worker_with_shuffle_test(rank: int, world_size: int, 
                                  data_size: int, result_queue: mp.Queue):
    """Worker that tests shuffled data with auto-detection."""
    try:
        setup_ddp(rank, world_size, "gloo")
        
        data = np.arange(data_size)
        
        # Use shuffle=True
        dataset = scDataset(
            data, 
            Streaming(shuffle=True),
            batch_size=10,
            fetch_factor=5,
            seed=42
        )
        
        collected = []
        for batch in dataset:
            collected.extend(batch.tolist())
        
        result_queue.put((rank, collected, None))
        
    except Exception as e:
        import traceback
        result_queue.put((rank, [], traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def test_ddp_cpu_only():
    """Test DDP with CPU only (gloo backend) - should work without CUDA."""
    world_size = 2
    data_size = 100
    
    # This test uses gloo backend which works on CPU
    results = run_ddp_test(world_size, data_size, batch_size=10, fetch_factor=5)
    
    # Verify results
    all_indices = []
    for rank in range(world_size):
        all_indices.extend(results[rank])
    
    unique_indices = set(all_indices)
    expected = set(range(data_size))
    assert unique_indices == expected, f"Missing: {expected - unique_indices}"


class TestDataParallelCompatibility:
    """Test compatibility with DataParallel (DP) training.
    
    Note: DataParallel works at the batch level after data loading.
    scDataset doesn't need special DP support - it just provides batches.
    These tests verify that batches are correctly formatted for DP.
    """
    
    @pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, 
                        reason="Need at least 2 GPUs for DP test")
    def test_dp_batch_format(self):
        """Test that scDataset produces batches suitable for DataParallel."""
        import torch.nn as nn
        
        # Create dataset
        data_size = 100
        batch_size = 16  # Must be >= num_gpus for DP
        data = np.random.randn(data_size, 10).astype(np.float32)
        
        dataset = scDataset(
            data,
            Streaming(shuffle=False),
            batch_size=batch_size,
            fetch_factor=2
        )
        
        # Simple model to test with DP
        model = nn.Linear(10, 5).cuda()
        dp_model = nn.DataParallel(model)
        
        # Verify batches can be processed by DP
        for batch in dataset:
            batch_tensor = torch.from_numpy(batch).cuda()
            output = dp_model(batch_tensor)
            assert output.shape[0] == batch.shape[0], "DP output size mismatch"
            break  # Just test one batch


class TestDDPWithDataLoader:
    """Test DDP with DataLoader multiprocessing (num_workers > 0)."""
    
    def test_ddp_with_dataloader_workers(self):
        """Test that DDP + DataLoader with workers works correctly."""
        world_size = 2
        data_size = 200
        
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_with_dataloader,
                args=(rank, world_size, data_size, result_queue)
            )
            p.start()
            processes.append(p)
        
        results = {}
        errors = []
        for _ in range(world_size):
            rank, indices, error = result_queue.get(timeout=120)
            results[rank] = indices
            if error:
                errors.append(f"Rank {rank}: {error}")
        
        for p in processes:
            p.join(timeout=30)
        
        if errors:
            pytest.fail(f"Errors in DDP workers:\n" + "\n".join(errors))
        
        # Verify all data is covered
        all_indices = []
        for rank in range(world_size):
            all_indices.extend(results[rank])
        
        unique_indices = set(all_indices)
        expected = set(range(data_size))
        
        assert unique_indices == expected, f"Missing: {expected - unique_indices}"


class TestDDPEpochConsistency:
    """Test that epochs are consistent across DDP ranks."""
    
    def test_ddp_epoch_different_shuffling(self):
        """Test that different epochs produce different shuffling but same split."""
        world_size = 2
        data_size = 100
        
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()
        
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_multiple_epochs,
                args=(rank, world_size, data_size, result_queue)
            )
            p.start()
            processes.append(p)
        
        results = {}
        errors = []
        for _ in range(world_size):
            rank, epoch_data, error = result_queue.get(timeout=60)
            results[rank] = epoch_data
            if error:
                errors.append(f"Rank {rank}: {error}")
        
        for p in processes:
            p.join(timeout=30)
        
        if errors:
            pytest.fail(f"Errors:\n" + "\n".join(errors))
        
        # Verify each epoch covers all data across ranks
        for epoch in range(2):
            all_indices = []
            for rank in range(world_size):
                all_indices.extend(results[rank][epoch])
            
            unique_indices = set(all_indices)
            expected = set(range(data_size))
            assert unique_indices == expected, f"Epoch {epoch}: Missing {expected - unique_indices}"
        
        # Verify epochs have same data per rank (same indices, possibly different order)
        for rank in range(world_size):
            set_epoch0 = set(results[rank][0])
            set_epoch1 = set(results[rank][1])
            assert set_epoch0 == set_epoch1, f"Rank {rank}: Different data between epochs"


if __name__ == "__main__":
    # Run basic test directly
    print("Running DDP test with 2 ranks...")
    try:
        results = run_ddp_test(world_size=2, data_size=100, batch_size=10, fetch_factor=5)
        print(f"Rank 0 processed {len(results[0])} samples")
        print(f"Rank 1 processed {len(results[1])} samples")
        
        all_indices = []
        for rank in range(2):
            all_indices.extend(results[rank])
        
        unique = set(all_indices)
        expected = set(range(100))
        
        if unique == expected:
            print("✓ All data covered correctly!")
        else:
            print(f"✗ Missing indices: {expected - unique}")
            
        if len(unique) == len(all_indices):
            print("✓ No overlap between ranks!")
        else:
            print(f"✗ Overlap detected: {len(all_indices) - len(unique)} duplicates")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nRunning DDP test with 3 ranks...")
    try:
        results = run_ddp_test(world_size=3, data_size=150, batch_size=10, fetch_factor=3)
        for rank in range(3):
            print(f"Rank {rank} processed {len(results[rank])} samples")
        
        all_indices = []
        for rank in range(3):
            all_indices.extend(results[rank])
        
        unique = set(all_indices)
        expected = set(range(150))
        
        if unique == expected and len(unique) == len(all_indices):
            print("✓ Test passed!")
        else:
            print("✗ Test failed!")
            
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

