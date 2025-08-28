"""
Multi-indexable data structure for synchronized indexing.

This module provides the :class:`MultiIndexable` class for grouping multiple
indexable objects that should be indexed together using the same indices.
This is particularly useful for multi-modal data or when working with
features and labels that need to stay synchronized.

.. autosummary::
   :toctree: generated/

   MultiIndexable
"""

from typing import Union, List, Dict, Any, Optional, Sequence
import numpy as np


class MultiIndexable:
    """
    Container for multiple indexable objects that should be indexed together.
    
    This class allows you to group multiple indexable objects (arrays, lists, etc.)
    and index them synchronously. It's particularly useful for scenarios like:
    
    - Multi-modal single-cell data (gene expression + protein data)
    - Features and labels (X, y) that need to stay aligned
    - Multiple data modalities that share the same sample dimension
    
    The class supports both positional and named access to the contained indexables,
    and ensures all indexables have the same length along the first dimension.
    
    Parameters
    ----------
    *indexables : indexable objects or dict
        Variable number of indexable objects that should be indexed together,
        OR a single dictionary where keys become names and values are indexables.
        All indexables must have the same length in the first dimension.
    names : list of str, optional
        Names for the indexables when using positional arguments.
        Must have the same length as the number of indexables.
        Cannot be used with dictionary input.
    **named_indexables : dict, optional
        Named indexable objects passed as keyword arguments.
        Cannot be used together with positional indexables.
        
    Attributes
    ----------
    names : list of str or None
        Names of the indexables if provided, None otherwise.
    count : int
        Number of indexables contained in this object.
        
    Raises
    ------
    ValueError
        If indexables have different lengths along the first dimension,
        or if the number of names doesn't match the number of indexables.
    TypeError
        If both positional and keyword indexables are provided.
        
    Examples
    --------
    Create with positional arguments:
    
    >>> import numpy as np
    >>> x = np.random.randn(100, 50)
    >>> y = np.random.randint(0, 3, 100)
    >>> multi = MultiIndexable(x, y, names=['features', 'labels'])
    >>> len(multi)
    100
    >>> multi.count
    2
    
    Create with dictionary as positional argument:
    
    >>> data_dict = {
    ...     'genes': np.random.randn(100, 2000),
    ...     'proteins': np.random.randn(100, 100)
    ... }
    >>> multi = MultiIndexable(data_dict)
    >>> subset = multi[10:20]  # Get samples 10-19 from both modalities
    >>> subset['genes'].shape
    (10, 2000)
    
    Create with keyword arguments:
    
    >>> multi = MultiIndexable(
    ...     genes=np.random.randn(100, 2000),
    ...     proteins=np.random.randn(100, 100)
    ... )
    >>> multi.names
    ['genes', 'proteins']
    
    Access by name or position:
    
    >>> multi = MultiIndexable(x, y, names=['x', 'y'])
    >>> same_x1 = multi[0]      # Access by position
    >>> same_x2 = multi['x']    # Access by name
    >>> np.array_equal(same_x1, same_x2)
    True
    
    Use with scDataset:
    
    >>> from scdataset import scDataset, Streaming
    >>> dataset = scDataset(multi, Streaming(), batch_size=32)
    >>> for batch in dataset:
    ...     genes, proteins = batch[0], batch[1]  # or batch['genes'], batch['proteins']
    ...     break
        
    See Also
    --------
    scdataset.scDataset : Main dataset class that can use MultiIndexable objects
    """
    
    def __init__(
        self, 
        *indexables, 
        names: Optional[List[str]] = None,
        **named_indexables
    ):
        """
        Initialize MultiIndexable with indexable objects.
        
        Can be initialized in four ways:
        1. Positional: MultiIndexable(x, y, z)
        2. Positional with names: MultiIndexable(x, y, names=['x', 'y'])  
        3. Dictionary as positional: MultiIndexable({'x': x_data, 'y': y_data})
        4. Named keywords: MultiIndexable(x=x_data, y=y_data)
        """
        # Handle different initialization patterns
        if indexables and named_indexables:
            raise TypeError("Cannot provide both positional and named indexables")
        
        # Check for single dictionary as positional argument
        if (len(indexables) == 1 and isinstance(indexables[0], dict) 
            and not named_indexables and names is None):
            # Dictionary passed as positional argument
            data_dict = indexables[0]
            self._names = list(data_dict.keys())
            self._indexables = list(data_dict.values())
            self._mapping = data_dict.copy()
        elif named_indexables:
            # Dictionary-style initialization
            self._names = list(named_indexables.keys())
            self._indexables = list(named_indexables.values())
            self._mapping = named_indexables.copy()
        elif indexables:
            # Positional initialization
            self._indexables = list(indexables)
            if names is not None:
                if len(names) != len(indexables):
                    raise ValueError(
                        f"Length of names ({len(names)}) must match number of "
                        f"indexables ({len(indexables)})"
                    )
                self._names = list(names)
                self._mapping = dict(zip(self._names, self._indexables))
            else:
                self._names = None
                self._mapping = None
        else:
            raise TypeError("Must provide at least one indexable object")
        
        # Validate that all indexables have the same length
        if not self._indexables:
            raise ValueError("No indexables provided")
            
        try:
            first_len = len(self._indexables[0])
        except TypeError:
            raise TypeError("All indexables must support len() operation")
            
        for i, indexable in enumerate(self._indexables[1:], start=1):
            try:
                curr_len = len(indexable)
            except TypeError:
                raise TypeError(f"Indexable at position {i} does not support len() operation")
                
            if curr_len != first_len:
                name_info = f" ('{self._names[i]}')" if self._names else ""
                raise ValueError(
                    f"All indexables must have the same length. "
                    f"First indexable has length {first_len}, but indexable {i}{name_info} "
                    f"has length {curr_len}"
                )
    
    @property
    def names(self) -> Optional[List[str]]:
        """Names of the indexables, if provided."""
        return self._names.copy() if self._names else None
    
    @property 
    def count(self) -> int:
        """Number of indexables contained in this object."""
        return len(self._indexables)
    
    def __getitem__(self, key: Union[int, str, slice, Sequence[int], np.ndarray]):
        """
        Index the MultiIndexable object.
        
        Parameters
        ----------
        key : int, str, slice, or array-like
            - int: Return the indexable at that position
            - str: Return the indexable with that name (if names provided)
            - slice/array: Return new MultiIndexable with subsets at those sample indices
            
        Returns
        -------
        object or MultiIndexable
            - Single indexable if key is int or str
            - New MultiIndexable with subsets if key represents sample indices
        """
        if isinstance(key, int):
            # Return the indexable at position key
            if key < 0:
                key = len(self._indexables) + key
            if not 0 <= key < len(self._indexables):
                raise IndexError(f"Index {key} out of range for {len(self._indexables)} indexables")
            return self._indexables[key]
        
        elif isinstance(key, str):
            # Return the named indexable
            if self._mapping is None:
                raise KeyError(f"No named indexables available. Available indices: 0-{len(self._indexables)-1}")
            if key not in self._mapping:
                raise KeyError(f"Key '{key}' not found. Available keys: {list(self._mapping.keys())}")
            return self._mapping[key]
        
        else:
            # Sample indices - return new MultiIndexable with subsets
            try:
                subset_indexables = [indexable[key] for indexable in self._indexables]
            except (IndexError, TypeError) as e:
                raise IndexError(f"Invalid indices for sample selection: {e}")
            
            # Preserve names if any
            if self._mapping:
                return MultiIndexable(**dict(zip(self._names, subset_indexables)))
            else:
                return MultiIndexable(*subset_indexables)
    
    def __len__(self) -> int:
        """Return the number of samples (length of first dimension)."""
        return len(self._indexables[0]) if self._indexables else 0
    
    def __repr__(self) -> str:
        """Return string representation of the MultiIndexable."""
        n_samples = len(self)
        if self._names:
            indexable_info = f"names={self._names}"
        else:
            indexable_info = f"count={self.count}"
        return f"MultiIndexable({indexable_info}, samples={n_samples})"
    
    def __iter__(self):
        """Iterate over indexables."""
        return iter(self._indexables)
    
    def items(self):
        """
        Iterate over (name, indexable) pairs.
        
        Yields
        ------
        tuple
            (name, indexable) pairs if names are available,
            (index, indexable) pairs otherwise.
        """
        if self._names:
            for name, indexable in zip(self._names, self._indexables):
                yield name, indexable
        else:
            for i, indexable in enumerate(self._indexables):
                yield i, indexable
    
    def keys(self):
        """
        Get the names or indices of indexables.
        
        Returns
        -------
        list
            List of names if available, list of indices otherwise.
        """
        return self._names.copy() if self._names else list(range(len(self._indexables)))
    
    def values(self):
        """
        Get the indexable objects.
        
        Returns
        -------
        list
            List of indexable objects.
        """
        return self._indexables.copy()
