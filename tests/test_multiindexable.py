"""
Tests for MultiIndexable class.

This module tests the MultiIndexable class from scdataset.multiindexable.
"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from scdataset import MultiIndexable

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def features_labels():
    """Create features and labels arrays."""
    features = np.random.randn(100, 50)
    labels = np.random.randint(0, 3, 100)
    return features, labels


@pytest.fixture
def multi_modal_data():
    """Create multi-modal data (genes, proteins, metadata)."""
    n_samples = 100
    return {
        "genes": np.random.randn(n_samples, 2000),
        "proteins": np.random.randn(n_samples, 100),
        "metadata": np.random.randn(n_samples, 10),
    }


# =============================================================================
# Basic Construction Tests
# =============================================================================


class TestMultiIndexableConstruction:
    """Tests for MultiIndexable construction."""

    def test_init_with_positional_args(self, features_labels):
        """Test initialization with positional arguments."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels)

        assert len(multi) == 100
        assert multi.count == 2
        assert multi.names is None

    def test_init_with_names(self, features_labels):
        """Test initialization with names."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        assert multi.names == ["X", "y"]

    def test_init_with_dict_positional(self, multi_modal_data):
        """Test initialization with dictionary as positional argument."""
        multi = MultiIndexable(multi_modal_data)

        assert len(multi) == 100
        assert multi.count == 3
        assert set(multi.names) == set(multi_modal_data.keys())

    def test_init_with_kwargs(self, multi_modal_data):
        """Test initialization with keyword arguments."""
        multi = MultiIndexable(
            genes=multi_modal_data["genes"],
            proteins=multi_modal_data["proteins"],
            metadata=multi_modal_data["metadata"],
        )

        assert len(multi) == 100
        assert multi.count == 3

    def test_init_with_unstructured(self, features_labels):
        """Test initialization with unstructured data."""
        features, labels = features_labels
        unstructured = {
            "gene_names": ["Gene_" + str(i) for i in range(50)],
            "dataset_name": "TestDataset",
        }

        multi = MultiIndexable(
            features, labels, names=["X", "y"], unstructured=unstructured
        )

        assert multi.unstructured == unstructured
        assert multi.unstructured["dataset_name"] == "TestDataset"

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        arr1 = np.zeros((100, 10))
        arr2 = np.zeros((50, 5))  # Different length

        with pytest.raises(ValueError, match="same length"):
            MultiIndexable(arr1, arr2)

    def test_mismatched_names_raises_error(self, features_labels):
        """Test that mismatched names count raises ValueError."""
        features, labels = features_labels

        with pytest.raises(ValueError):
            MultiIndexable(features, labels, names=["X"])  # Only 1 name for 2 arrays

    def test_mixed_args_and_kwargs_raises_error(self, features_labels):
        """Test that mixing positional and keyword args raises TypeError."""
        features, labels = features_labels

        with pytest.raises(TypeError):
            MultiIndexable(features, extra=labels)

    def test_unstructured_must_be_dict(self, features_labels):
        """Test that unstructured must be a dictionary."""
        features, labels = features_labels

        with pytest.raises(TypeError, match="dictionary"):
            MultiIndexable(features, labels, unstructured="not a dict")


# =============================================================================
# Indexing Tests
# =============================================================================


class TestMultiIndexableIndexing:
    """Tests for MultiIndexable indexing operations."""

    def test_integer_index_returns_indexable(self, features_labels):
        """Test that integer indexing returns the indexable itself."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        assert_array_equal(multi[0], features)
        assert_array_equal(multi[1], labels)

    def test_string_index_returns_indexable(self, features_labels):
        """Test that string indexing returns the indexable by name."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        assert_array_equal(multi["X"], features)
        assert_array_equal(multi["y"], labels)

    def test_slice_returns_new_multi_indexable(self, features_labels):
        """Test that slicing returns a new MultiIndexable."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        subset = multi[10:20]

        assert isinstance(subset, MultiIndexable)
        assert len(subset) == 10
        assert_array_equal(subset["X"], features[10:20])
        assert_array_equal(subset["y"], labels[10:20])

    def test_array_index_returns_new_multi_indexable(self, features_labels):
        """Test that array indexing returns a new MultiIndexable."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        indices = np.array([0, 5, 10, 15])
        subset = multi[indices]

        assert isinstance(subset, MultiIndexable)
        assert len(subset) == 4
        assert_array_equal(subset["X"], features[indices])

    def test_list_index_returns_new_multi_indexable(self, features_labels):
        """Test that list indexing returns a new MultiIndexable."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        indices = [0, 5, 10, 15]
        subset = multi[indices]

        assert isinstance(subset, MultiIndexable)
        assert len(subset) == 4

    def test_slice_preserves_unstructured(self, features_labels):
        """Test that slicing preserves unstructured data."""
        features, labels = features_labels
        unstructured = {"info": "test", "gene_names": ["A", "B", "C"]}
        multi = MultiIndexable(
            features, labels, names=["X", "y"], unstructured=unstructured
        )

        subset = multi[10:20]

        assert subset.unstructured == unstructured
        assert subset.unstructured["info"] == "test"

    def test_invalid_string_index_raises_error(self, features_labels):
        """Test that invalid string index raises KeyError."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        with pytest.raises(KeyError):
            _ = multi["invalid_name"]

    def test_negative_index(self, features_labels):
        """Test negative indexing."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        # Integer negative index should return the last indexable
        assert_array_equal(multi[-1], labels)


# =============================================================================
# Property Tests
# =============================================================================


class TestMultiIndexableProperties:
    """Tests for MultiIndexable properties."""

    def test_len(self, multi_modal_data):
        """Test __len__ method."""
        multi = MultiIndexable(multi_modal_data)
        assert len(multi) == 100

    def test_count(self, multi_modal_data):
        """Test count property."""
        multi = MultiIndexable(multi_modal_data)
        assert multi.count == 3

    def test_names_property(self, features_labels):
        """Test names property."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])
        assert multi.names == ["X", "y"]

    def test_unstructured_property_default(self, features_labels):
        """Test unstructured property default value."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels)
        assert multi.unstructured == {}

    def test_unstructured_keys_property(self, features_labels):
        """Test unstructured_keys property."""
        features, labels = features_labels
        unstructured = {"key1": "val1", "key2": "val2"}
        multi = MultiIndexable(features, labels, unstructured=unstructured)

        assert set(multi.unstructured_keys) == {"key1", "key2"}


# =============================================================================
# Repr Tests
# =============================================================================


class TestMultiIndexableRepr:
    """Tests for MultiIndexable string representation."""

    def test_repr_basic(self, features_labels):
        """Test __repr__ method."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        repr_str = repr(multi)
        assert "MultiIndexable" in repr_str
        assert "100" in repr_str  # Length
        assert "names=['X', 'y']" in repr_str  # Names shown when provided

    def test_repr_with_unstructured(self, features_labels):
        """Test __repr__ includes unstructured keys."""
        features, labels = features_labels
        multi = MultiIndexable(
            features, labels, names=["X", "y"], unstructured={"meta": "data"}
        )

        repr_str = repr(multi)
        assert "unstructured" in repr_str.lower() or "meta" in repr_str


# =============================================================================
# Integration Tests
# =============================================================================


class TestMultiIndexableIntegration:
    """Integration tests for MultiIndexable."""

    def test_iteration_over_subset(self, multi_modal_data):
        """Test iterating over subsets."""
        multi = MultiIndexable(multi_modal_data)

        # Simulate batch iteration
        batch_size = 10
        for i in range(0, len(multi), batch_size):
            batch = multi[i : i + batch_size]
            assert len(batch) <= batch_size
            assert batch.count == 3

    def test_chained_indexing(self, multi_modal_data):
        """Test chained indexing operations."""
        multi = MultiIndexable(multi_modal_data)

        # First subset
        subset1 = multi[20:80]
        assert len(subset1) == 60

        # Second subset of first
        subset2 = subset1[10:20]
        assert len(subset2) == 10

        # Verify data integrity
        assert_array_equal(
            subset2["genes"],
            multi_modal_data["genes"][30:40],  # 20 + 10 to 20 + 20
        )

    def test_with_sparse_indices(self, multi_modal_data):
        """Test with sparse index selection."""
        multi = MultiIndexable(multi_modal_data)

        sparse_indices = [0, 10, 20, 50, 99]
        subset = multi[sparse_indices]

        assert len(subset) == 5
        assert_array_equal(subset["genes"], multi_modal_data["genes"][sparse_indices])

    def test_boolean_indexing(self, features_labels):
        """Test boolean array indexing."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        # Create boolean mask
        mask = labels == 0
        subset = multi[mask]

        assert len(subset) == np.sum(mask)
        assert_array_equal(subset["X"], features[mask])


# =============================================================================
# Edge Cases
# =============================================================================


class TestMultiIndexableEdgeCases:
    """Test edge cases for MultiIndexable."""

    def test_single_indexable(self):
        """Test with single indexable."""
        data = np.random.randn(100, 50)
        multi = MultiIndexable(data, names=["X"])

        assert len(multi) == 100
        assert multi.count == 1
        assert_array_equal(multi["X"], data)

    def test_empty_indexables(self):
        """Test with empty arrays."""
        empty1 = np.array([]).reshape(0, 10)
        empty2 = np.array([]).reshape(0, 5)

        multi = MultiIndexable(empty1, empty2, names=["A", "B"])
        assert len(multi) == 0
        assert multi.count == 2

    def test_single_sample(self):
        """Test with single sample."""
        features = np.random.randn(1, 50)
        labels = np.array([0])

        multi = MultiIndexable(features, labels, names=["X", "y"])
        assert len(multi) == 1

        # Subsetting single sample
        subset = multi[0:1]
        assert len(subset) == 1

    def test_with_lists(self):
        """Test with Python lists instead of numpy arrays."""
        list1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        list2 = [0, 1, 2]

        multi = MultiIndexable(list1, list2, names=["A", "B"])
        assert len(multi) == 3

    def test_large_unstructured(self):
        """Test with large unstructured metadata."""
        features = np.random.randn(100, 50)
        large_unstructured = {
            "gene_names": ["Gene_" + str(i) for i in range(50)],
            "cell_types": ["Type_" + str(i) for i in range(100)],
            "metadata": {"nested": {"data": [1, 2, 3]}, "list": list(range(1000))},
        }

        multi = MultiIndexable(features, names=["X"], unstructured=large_unstructured)

        # Verify unstructured is preserved through subsetting
        subset = multi[10:20]
        assert subset.unstructured == large_unstructured
        assert subset.unstructured["metadata"]["nested"]["data"] == [1, 2, 3]


class TestMultiIndexableDictInterface:
    """Test dict-like interface methods."""

    def test_items_with_names(self, features_labels):
        """Test items() method when names are provided."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        items = list(multi.items())
        assert len(items) == 2
        assert items[0][0] == "X"
        assert items[1][0] == "y"
        assert_array_equal(items[0][1], features)
        assert_array_equal(items[1][1], labels)

    def test_items_without_names(self, features_labels):
        """Test items() method when names are not provided."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels)

        items = list(multi.items())
        assert len(items) == 2
        assert items[0][0] == 0  # Index instead of name
        assert items[1][0] == 1

    def test_keys_with_names(self, features_labels):
        """Test keys() method with named indexables."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        keys = multi.keys()
        assert keys == ["X", "y"]

    def test_keys_without_names(self, features_labels):
        """Test keys() method without names."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels)

        keys = multi.keys()
        assert keys == [0, 1]

    def test_values(self, features_labels):
        """Test values() method."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        values = multi.values()
        assert len(values) == 2
        assert_array_equal(values[0], features)
        assert_array_equal(values[1], labels)


class TestMultiIndexableErrorHandling:
    """Test error handling in MultiIndexable."""

    def test_no_indexables_raises_error(self):
        """Test that creating with no indexables raises error."""
        with pytest.raises(TypeError, match="Must provide at least one indexable"):
            MultiIndexable()

    def test_indexable_without_len_raises_error(self):
        """Test that indexable without len() raises error."""

        class NoLen:
            def __getitem__(self, idx):
                return idx

        with pytest.raises(TypeError, match="support len"):
            MultiIndexable(NoLen())

    def test_index_out_of_range_negative(self, features_labels):
        """Test that out of range negative index raises error."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels)

        with pytest.raises(IndexError, match="out of range"):
            _ = multi[-10]

    def test_key_not_found_with_names(self, features_labels):
        """Test that accessing missing key raises KeyError."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels, names=["X", "y"])

        with pytest.raises(KeyError, match="not found"):
            _ = multi["missing"]

    def test_key_access_without_names(self, features_labels):
        """Test that accessing by key without names raises error."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels)  # No names

        with pytest.raises(KeyError, match="No named indexables"):
            _ = multi["X"]

    def test_repr_without_names_shows_count(self, features_labels):
        """Test that repr shows count when names not provided."""
        features, labels = features_labels
        multi = MultiIndexable(features, labels)  # No names

        repr_str = repr(multi)
        assert "count=2" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
