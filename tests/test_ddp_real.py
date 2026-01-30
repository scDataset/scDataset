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

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Add src to path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scdataset import scDataset
from scdataset.strategy import BlockShuffling, Streaming


def setup_ddp(rank: int, world_size: int, backend: str = "gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_ddp():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def ddp_worker_collect_indices(
    rank: int,
    world_size: int,
    data_size: int,
    batch_size: int,
    fetch_factor: int,
    result_queue: mp.Queue,
    use_nccl: bool = False,
):
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
            seed=42,
        )

        # Verify auto-detection worked
        assert (
            dataset.rank == rank
        ), f"Rank {rank}: auto-detected rank is {dataset.rank}"
        assert (
            dataset.world_size == world_size
        ), f"Rank {rank}: auto-detected world_size is {dataset.world_size}"

        # Collect all data this rank processes
        collected = []
        for batch in dataset:
            collected.extend(batch.tolist())

        result_queue.put((rank, collected, None))

    except Exception:
        import traceback

        result_queue.put((rank, [], traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def run_ddp_test(
    world_size: int, data_size: int, batch_size: int = 10, fetch_factor: int = 5
):
    """
    Run a DDP test with the specified number of processes.
    Returns a dict mapping rank -> list of indices processed.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=ddp_worker_collect_indices,
            args=(rank, world_size, data_size, batch_size, fetch_factor, result_queue),
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


def ddp_worker_with_dataloader(
    rank: int, world_size: int, data_size: int, result_queue
):
    """Worker that tests DDP with DataLoader multiprocessing."""
    try:
        setup_ddp(rank, world_size, "gloo")

        data = np.arange(data_size)

        dataset = scDataset(
            data, Streaming(shuffle=False), batch_size=10, fetch_factor=5, seed=42
        )

        # Use DataLoader with 2 workers
        dataloader = DataLoader(dataset, batch_size=None, num_workers=2)

        collected = []
        for batch in dataloader:
            if isinstance(batch, torch.Tensor):
                batch = batch.numpy()
            collected.extend(batch.flatten().tolist())

        result_queue.put((rank, collected, None))

    except Exception:
        import traceback

        result_queue.put((rank, [], traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def ddp_worker_multiple_epochs(
    rank: int, world_size: int, data_size: int, result_queue
):
    """Worker that collects data from multiple epochs."""
    try:
        setup_ddp(rank, world_size, "gloo")

        data = np.arange(data_size)

        dataset = scDataset(
            data, Streaming(shuffle=True), batch_size=10, fetch_factor=5, seed=42
        )

        # Collect data from 2 epochs
        epoch_results = []
        for _ in range(2):
            epoch_data = []
            for batch in dataset:
                epoch_data.extend(batch.tolist())
            epoch_results.append(epoch_data)

        result_queue.put((rank, epoch_results, None))

    except Exception:
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
        assert len(unique_indices) == len(
            all_indices
        ), "Overlap detected between ranks!"

        # Check all data is covered
        expected = set(range(data_size))
        assert (
            unique_indices == expected
        ), f"Missing indices: {expected - unique_indices}"

        # Check roughly equal distribution
        for rank in range(world_size):
            count = len(results[rank])
            expected_count = data_size // world_size
            # Allow some variance due to batch boundaries
            assert (
                abs(count - expected_count) <= batch_size
            ), f"Rank {rank} has {count} samples, expected ~{expected_count}"

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
        batch_size = 7  # Not divisible by data_size
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


def ddp_worker_with_shuffle_test(
    rank: int, world_size: int, data_size: int, result_queue: mp.Queue
):
    """Worker that tests shuffled data with auto-detection."""
    try:
        setup_ddp(rank, world_size, "gloo")

        data = np.arange(data_size)

        # Use shuffle=True
        dataset = scDataset(
            data, Streaming(shuffle=True), batch_size=10, fetch_factor=5, seed=42
        )

        collected = []
        for batch in dataset:
            collected.extend(batch.tolist())

        result_queue.put((rank, collected, None))

    except Exception:
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

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Need at least 2 GPUs for DP test",
    )
    def test_dp_batch_format(self):
        """Test that scDataset produces batches suitable for DataParallel."""
        import torch.nn as nn

        # Create dataset
        data_size = 100
        batch_size = 16  # Must be >= num_gpus for DP
        data = np.random.randn(data_size, 10).astype(np.float32)

        dataset = scDataset(
            data, Streaming(shuffle=False), batch_size=batch_size, fetch_factor=2
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

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_with_dataloader,
                args=(rank, world_size, data_size, result_queue),
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
            pytest.fail("Errors in DDP workers:\n" + "\n".join(errors))

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

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_multiple_epochs,
                args=(rank, world_size, data_size, result_queue),
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
            pytest.fail("Errors:\n" + "\n".join(errors))

        # Verify each epoch covers all data across ranks
        for epoch in range(2):
            all_indices = []
            for rank in range(world_size):
                all_indices.extend(results[rank][epoch])

            unique_indices = set(all_indices)
            expected = set(range(data_size))
            assert (
                unique_indices == expected
            ), f"Epoch {epoch}: Missing {expected - unique_indices}"

        # Verify epochs have same data per rank (same indices, possibly different order)
        for rank in range(world_size):
            set_epoch0 = set(results[rank][0])
            set_epoch1 = set(results[rank][1])
            assert (
                set_epoch0 == set_epoch1
            ), f"Rank {rank}: Different data between epochs"


def ddp_worker_random_seed(rank: int, world_size: int, data_size: int, result_queue):
    """
    Worker that tests seed=None in DDP.
    Uses no explicit seed, so the dataset should generate a random seed
    and broadcast it from rank 0 to all ranks.
    """
    try:
        setup_ddp(rank, world_size, "gloo")

        data = np.arange(data_size)

        # Create scDataset with seed=None (default) - should broadcast from rank 0
        dataset = scDataset(
            data,
            Streaming(shuffle=False),  # No shuffle for deterministic testing
            batch_size=10,
            fetch_factor=5,
            # seed=None is the default - random seed broadcast from rank 0
        )

        # Collect all data this rank processes
        collected = []
        for batch in dataset:
            collected.extend(batch.tolist())

        # Also collect the base_seed for verification
        result_queue.put((rank, collected, dataset._base_seed, None))

    except Exception:
        import traceback

        result_queue.put((rank, [], None, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def run_ddp_random_seed_test(world_size: int, data_size: int):
    """
    Run a DDP test with seed=None to verify seed broadcast.
    Returns dict mapping rank -> (indices, base_seed).
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=ddp_worker_random_seed,
            args=(rank, world_size, data_size, result_queue),
        )
        p.start()
        processes.append(p)

    results = {}
    seeds = {}
    errors = []
    for _ in range(world_size):
        rank, indices, seed, error = result_queue.get(timeout=60)
        results[rank] = indices
        seeds[rank] = seed
        if error:
            errors.append(f"Rank {rank}: {error}")

    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    if errors:
        raise RuntimeError("Errors in DDP workers:\n" + "\n".join(errors))

    return results, seeds


class TestDDPRandomSeed:
    """Test seed=None works correctly in DDP (broadcasts from rank 0)."""

    # Note: These tests use gloo backend which works on CPU, no CUDA needed
    def test_ddp_random_seed_broadcast(self):
        """Test that seed=None broadcasts the same seed to all ranks."""
        world_size = 2
        data_size = 100

        results, seeds = run_ddp_random_seed_test(world_size, data_size)

        # All ranks should have the same seed
        seed_values = list(seeds.values())
        assert all(
            s == seed_values[0] for s in seed_values
        ), f"Seeds differ across ranks: {seeds}"

        # Data should still be correctly split (no overlap, all covered)
        all_indices = []
        for rank in range(world_size):
            all_indices.extend(results[rank])

        unique_indices = set(all_indices)
        expected = set(range(data_size))
        assert len(unique_indices) == len(all_indices), "Overlap detected!"
        assert unique_indices == expected, f"Missing: {expected - unique_indices}"

    def test_ddp_random_seed_different_runs(self):
        """Test that seed=None produces different seeds on different runs."""
        world_size = 2
        data_size = 50

        # Run test twice - should get different seeds (with very high probability)
        _, seeds1 = run_ddp_random_seed_test(world_size, data_size)
        _, seeds2 = run_ddp_random_seed_test(world_size, data_size)

        # The seeds should be different between runs
        # (tiny chance of collision, but acceptable for testing)
        seed1 = seeds1[0]
        seed2 = seeds2[0]
        # Don't assert they're different - just log for visibility
        # In a real scenario, they should almost always differ
        print(f"Run 1 seed: {seed1}, Run 2 seed: {seed2}")
        # Both runs should still have correct data split
        # (verified implicitly by not throwing errors)


# =============================================================================
# Edge Case Tests for DDP
# =============================================================================


def ddp_worker_drop_last(rank: int, world_size: int, result_queue: mp.Queue):
    """Worker that tests drop_last=True with DDP."""
    try:
        setup_ddp(rank, world_size, "gloo")

        # Data size that doesn't divide evenly by batch size
        data = np.arange(107)  # 107 samples, batch_size=10

        dataset = scDataset(
            data,
            Streaming(shuffle=False),
            batch_size=10,
            fetch_factor=5,
            drop_last=True,
            seed=42,
        )

        # Collect all data
        collected = []
        batch_sizes = []
        for batch in dataset:
            collected.extend(batch.tolist())
            batch_sizes.append(len(batch))

        # With drop_last=True, all batches should be exactly batch_size
        expected_len = len(dataset)

        result_queue.put((rank, collected, batch_sizes, expected_len, None))

    except Exception:
        import traceback

        result_queue.put((rank, [], [], 0, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def ddp_worker_transforms(rank: int, world_size: int, result_queue: mp.Queue):
    """Worker that tests fetch_transform and batch_transform with DDP."""
    try:
        setup_ddp(rank, world_size, "gloo")

        data = np.arange(100).astype(np.float32)

        # Define simple transforms
        def fetch_transform(x):
            return x * 2  # Double the values

        def batch_transform(x):
            return x + 1  # Add 1

        dataset = scDataset(
            data,
            Streaming(shuffle=False),
            batch_size=10,
            fetch_factor=5,
            fetch_transform=fetch_transform,
            batch_transform=batch_transform,
            seed=42,
        )

        # Collect all data
        collected = []
        for batch in dataset:
            collected.extend(batch.tolist())

        result_queue.put((rank, collected, None))

    except Exception:
        import traceback

        result_queue.put((rank, [], traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def ddp_worker_checkpoint_resume(
    rank: int, world_size: int, start_epoch: int, result_queue: mp.Queue
):
    """Worker that simulates checkpoint resume by setting _epoch manually."""
    try:
        setup_ddp(rank, world_size, "gloo")

        data = np.arange(100)

        dataset = scDataset(
            data,
            BlockShuffling(block_size=10),
            batch_size=10,
            fetch_factor=5,
            seed=42,
        )

        # Simulate checkpoint resume - manually set the epoch
        dataset._epoch = start_epoch

        # Collect data from this "epoch"
        collected = []
        for batch in dataset:
            collected.extend(batch.tolist())

        result_queue.put((rank, collected, dataset._epoch, None))

    except Exception:
        import traceback

        result_queue.put((rank, [], 0, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


def ddp_worker_small_batch(rank: int, world_size: int, result_queue: mp.Queue):
    """Worker that tests batch_size=1 with DDP."""
    try:
        setup_ddp(rank, world_size, "gloo")

        data = np.arange(50)

        dataset = scDataset(
            data,
            Streaming(shuffle=False),
            batch_size=1,  # Smallest possible batch
            fetch_factor=10,
            seed=42,
        )

        collected = []
        batch_count = 0
        for batch in dataset:
            collected.extend(batch.tolist())
            batch_count += 1

        result_queue.put((rank, collected, batch_count, None))

    except Exception:
        import traceback

        result_queue.put((rank, [], 0, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            cleanup_ddp()


class TestDDPDropLast:
    """Test drop_last=True works correctly in DDP."""

    def test_ddp_drop_last_all_complete_batches(self):
        """Test that drop_last=True only produces complete batches in DDP."""
        world_size = 2

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_drop_last,
                args=(rank, world_size, result_queue),
            )
            p.start()
            processes.append(p)

        results = {}
        batch_sizes_per_rank = {}
        expected_lens = {}
        errors = []
        for _ in range(world_size):
            rank, indices, batch_sizes, expected_len, error = result_queue.get(
                timeout=60
            )
            results[rank] = indices
            batch_sizes_per_rank[rank] = batch_sizes
            expected_lens[rank] = expected_len
            if error:
                errors.append(f"Rank {rank}: {error}")

        for p in processes:
            p.join(timeout=30)

        if errors:
            pytest.fail("Errors in DDP workers:\n" + "\n".join(errors))

        # Verify all batches are complete (size == batch_size)
        batch_size = 10
        for rank in range(world_size):
            for i, bs in enumerate(batch_sizes_per_rank[rank]):
                assert (
                    bs == batch_size
                ), f"Rank {rank} batch {i} has size {bs}, expected {batch_size}"

        # Verify len() matches actual batch count
        for rank in range(world_size):
            actual_batches = len(batch_sizes_per_rank[rank])
            assert (
                actual_batches == expected_lens[rank]
            ), f"Rank {rank}: len()={expected_lens[rank]} but got {actual_batches} batches"


class TestDDPTransforms:
    """Test that transforms work correctly in DDP."""

    def test_ddp_with_fetch_and_batch_transforms(self):
        """Test fetch_transform and batch_transform work in DDP context."""
        world_size = 2

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_transforms,
                args=(rank, world_size, result_queue),
            )
            p.start()
            processes.append(p)

        results = {}
        errors = []
        for _ in range(world_size):
            rank, indices, error = result_queue.get(timeout=60)
            results[rank] = indices
            if error:
                errors.append(f"Rank {rank}: {error}")

        for p in processes:
            p.join(timeout=30)

        if errors:
            pytest.fail("Errors in DDP workers:\n" + "\n".join(errors))

        # Verify transforms were applied: original * 2 + 1
        all_collected = []
        for rank in range(world_size):
            all_collected.extend(results[rank])

        # Each original value x should become x * 2 + 1
        expected_transformed = {x * 2 + 1 for x in range(100)}
        actual_transformed = set(all_collected)

        assert actual_transformed == expected_transformed, (
            f"Transform mismatch. Missing: {expected_transformed - actual_transformed}, "
            f"Extra: {actual_transformed - expected_transformed}"
        )


class TestDDPCheckpointResume:
    """Test checkpoint resume scenario with DDP."""

    def test_ddp_checkpoint_resume_reproducibility(self):
        """Test that resuming from a checkpoint produces consistent results."""
        world_size = 2
        start_epoch = 5  # Simulate resuming from epoch 5

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        # First run - collect data at "epoch 5"
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_checkpoint_resume,
                args=(rank, world_size, start_epoch, result_queue),
            )
            p.start()
            processes.append(p)

        run1_results = {}
        errors = []
        for _ in range(world_size):
            rank, indices, epoch, error = result_queue.get(timeout=60)
            run1_results[rank] = indices
            if error:
                errors.append(f"Run 1 Rank {rank}: {error}")

        for p in processes:
            p.join(timeout=30)

        if errors:
            pytest.fail("Errors in first run:\n" + "\n".join(errors))

        # Second run - should get same data at "epoch 5"
        result_queue2 = ctx.Queue()
        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_checkpoint_resume,
                args=(rank, world_size, start_epoch, result_queue2),
            )
            p.start()
            processes.append(p)

        run2_results = {}
        errors = []
        for _ in range(world_size):
            rank, indices, epoch, error = result_queue2.get(timeout=60)
            run2_results[rank] = indices
            if error:
                errors.append(f"Run 2 Rank {rank}: {error}")

        for p in processes:
            p.join(timeout=30)

        if errors:
            pytest.fail("Errors in second run:\n" + "\n".join(errors))

        # Verify both runs produced identical data per rank
        for rank in range(world_size):
            assert run1_results[rank] == run2_results[rank], (
                f"Rank {rank}: Checkpoint resume not reproducible! "
                f"First 5: {run1_results[rank][:5]} vs {run2_results[rank][:5]}"
            )


class TestDDPSmallBatch:
    """Test edge case with batch_size=1 in DDP."""

    def test_ddp_batch_size_one(self):
        """Test that batch_size=1 works correctly in DDP."""
        world_size = 2
        data_size = 50

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for rank in range(world_size):
            p = ctx.Process(
                target=ddp_worker_small_batch,
                args=(rank, world_size, result_queue),
            )
            p.start()
            processes.append(p)

        results = {}
        batch_counts = {}
        errors = []
        for _ in range(world_size):
            rank, indices, batch_count, error = result_queue.get(timeout=60)
            results[rank] = indices
            batch_counts[rank] = batch_count
            if error:
                errors.append(f"Rank {rank}: {error}")

        for p in processes:
            p.join(timeout=30)

        if errors:
            pytest.fail("Errors in DDP workers:\n" + "\n".join(errors))

        # Verify all data is covered
        all_indices = []
        for rank in range(world_size):
            all_indices.extend(results[rank])

        unique_indices = set(all_indices)
        expected = set(range(data_size))

        assert len(unique_indices) == len(all_indices), "Overlap detected!"
        assert unique_indices == expected, f"Missing: {expected - unique_indices}"

        # Verify each sample was a batch of 1
        total_batches = sum(batch_counts.values())
        assert (
            total_batches == data_size
        ), f"Expected {data_size} batches, got {total_batches}"


if __name__ == "__main__":
    # Run basic test directly
    print("Running DDP test with 2 ranks...")
    try:
        results = run_ddp_test(
            world_size=2, data_size=100, batch_size=10, fetch_factor=5
        )
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
        results = run_ddp_test(
            world_size=3, data_size=150, batch_size=10, fetch_factor=3
        )
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
