#!/usr/bin/env python3
"""
Test script demonstrating MultiIndexable functionality.

This script shows the various ways to initialize and use the MultiIndexable class
for synchronized indexing of multiple data objects.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from scdataset import MultiIndexable, scDataset, BlockShuffling


def test_basic_functionality():
    """Test basic MultiIndexable functionality."""
    print("=== Testing Basic MultiIndexable Functionality ===")
    
    # Create test data
    x = np.random.randn(100, 10)
    y = np.random.randint(0, 3, 100)
    
    # Test 1: Positional arguments
    multi = MultiIndexable(x, y)
    print(f"1. Positional args - Length: {len(multi)}, Count: {multi.count}")
    
    # Test 2: Positional with names
    multi_named = MultiIndexable(x, y, names=['features', 'labels'])
    print(f"2. Named args - Names: {multi_named.names}")
    
    # Test 3: Dictionary as positional
    data_dict = {'X': x, 'y': y}
    multi_dict = MultiIndexable(data_dict)
    print(f"3. Dict positional - Names: {multi_dict.names}")
    
    # Test 4: Keyword arguments
    multi_kwargs = MultiIndexable(features=x, labels=y)
    print(f"4. Keyword args - Names: {multi_kwargs.names}")
    
    # Test indexing
    subset = multi_named[[0, 1, 2]]
    print(f"5. Subset indexing - Length: {len(subset)}, Names: {subset.names}")
    
    # Test name vs position access
    features_by_name = multi_named['features']
    features_by_pos = multi_named[0]
    print(f"6. Access by name shape: {features_by_name.shape}")
    print(f"   Access by position shape: {features_by_pos.shape}")
    print(f"   Are equal: {np.array_equal(features_by_name, features_by_pos)}")


def test_multimodal_example():
    """Test the classical X, y setup that motivated MultiIndexable."""
    print("\n=== Testing Classical X, y Setup ===")
    
    # Simulate a classical ML setup
    n_samples = 1000
    X = np.random.randn(n_samples, 50)  # Features
    y = np.random.randint(0, 3, n_samples)  # Labels
    sample_ids = np.arange(n_samples)  # Sample identifiers
    
    # Create MultiIndexable - this ensures X, y, and sample_ids stay synchronized
    data = MultiIndexable(X, y, sample_ids, names=['features', 'labels', 'ids'])
    
    print(f"Data length: {len(data)}")
    print(f"Data names: {data.names}")
    
    # Test subset - indices are synchronized across all arrays
    subset_indices = [0, 10, 20, 30, 40]
    subset = data[subset_indices]
    
    print(f"Subset features shape: {subset['features'].shape}")
    print(f"Subset labels shape: {subset['labels'].shape}")
    print(f"Subset IDs: {subset['ids']}")
    
    # Verify synchronization - check that the IDs match the subset indices
    print(f"IDs match indices: {np.array_equal(subset['ids'], subset_indices)}")


def test_with_scdataset():
    """Test MultiIndexable integration with scDataset."""
    print("\n=== Testing MultiIndexable with scDataset ===")
    
    # Create multi-modal single-cell data
    n_cells = 1000
    gene_expr = np.random.randn(n_cells, 2000)  # Gene expression
    protein_expr = np.random.randn(n_cells, 100)  # Protein measurements
    cell_types = np.random.randint(0, 5, n_cells)  # Cell type labels
    
    # Group data with MultiIndexable
    multimodal_data = MultiIndexable(
        genes=gene_expr,
        proteins=protein_expr,
        cell_types=cell_types
    )
    
    # Create dataset with synchronized sampling
    dataset = scDataset(
        multimodal_data,
        BlockShuffling(block_size=64),
        batch_size=32
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Get a batch
    batch = next(iter(dataset))
    print(f"Batch type: {type(batch)}")
    print(f"Gene batch shape: {batch['genes'].shape}")
    print(f"Protein batch shape: {batch['proteins'].shape}")
    print(f"Cell types batch shape: {batch['cell_types'].shape}")
    
    # Verify all modalities have same batch size (synchronized)
    batch_sizes = [
        batch['genes'].shape[0],
        batch['proteins'].shape[0], 
        batch['cell_types'].shape[0]
    ]
    print(f"All batch sizes equal: {len(set(batch_sizes)) == 1}")


def test_dictionary_methods():
    """Test dictionary-like methods when names are provided."""
    print("\n=== Testing Dictionary-like Methods ===")
    
    data = {
        'X': np.random.randn(100, 50),
        'y': np.random.randint(0, 3, 100),
        'metadata': np.arange(100)
    }
    
    multi = MultiIndexable(data)
    
    print(f"Keys: {list(multi.keys())}")
    print(f"Values shapes: {[v.shape if hasattr(v, 'shape') else len(v) for v in multi.values()]}")
    print(f"Items: {[(k, v.shape if hasattr(v, 'shape') else len(v)) for k, v in multi.items()]}")
    
    # Test iteration
    print("Iteration over items:")
    for i, (name, data_array) in enumerate(multi.items()):
        shape = data_array.shape if hasattr(data_array, 'shape') else len(data_array)
        print(f"  {name}: {shape}")
        if i >= 2:  # Just show first 3
            break


if __name__ == "__main__":
    test_basic_functionality()
    test_multimodal_example()
    test_with_scdataset()
    test_dictionary_methods()
    print("\n✅ All tests completed successfully!")
