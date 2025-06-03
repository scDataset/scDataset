from typing import Optional, List, Union, Callable
import warnings

import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

class scDataset(IterableDataset):
    """
    Iterable PyTorch Dataset for single-cell data collections with flexible batching,
    shuffling, and transformation options.

    Parameters
    ----------
    data_collection : object
        The data collection to sample from (e.g., AnnCollection, HuggingFace Dataset, numpy array, etc.).
    batch_size : int
        Number of samples per minibatch.
    block_size : int, default=1
        Number of samples per block for shuffling.
    fetch_factor : int, default=1
        Multiplier for fetch size relative to batch size.
    shuffle : bool, default=False
        Whether to shuffle data before batching.
    drop_last : bool, default=False
        Whether to drop the last incomplete batch.
    sort_before_fetch : bool, optional
        Whether to sort indices before fetching.
    shuffle_before_yield : bool, optional
        Whether to shuffle within fetched blocks before yielding batches.
    fetch_transform : Callable, optional
        Function to transform data after fetching.
    batch_transform : Callable, optional
        Function to transform each batch before yielding.
    fetch_callback : Callable, optional
        Custom function to fetch data given indices.
    batch_callback : Callable, optional
        Custom function to fetch batch data given indices.
    """
    def __init__(
        self, data_collection, 
        batch_size: int, block_size: int = 1, 
        fetch_factor: int = 1, shuffle: bool = False, drop_last: bool = False,
        sort_before_fetch: Optional[bool] = None, shuffle_before_yield: Optional[bool] = None,
        fetch_transform: Optional[Callable] = None, batch_transform: Optional[Callable] = None,
        fetch_callback: Optional[Callable] = None, batch_callback: Optional[Callable] = None
    ):
        """
        Initialize the scDataset.
        """
        if shuffle:
            if sort_before_fetch is None:
                sort_before_fetch = True
            if shuffle_before_yield is None:
                shuffle_before_yield = True
        if (not shuffle) and shuffle_before_yield:
            raise ValueError("shuffle_before_yield=True requires shuffle=True")
        if (fetch_factor == 1) and shuffle_before_yield:
            warnings.warn("shuffle_before_yield=True and fetch_factor=1, this will return the same batch unless some downstream logic is applied")
        if (not sort_before_fetch) and shuffle_before_yield:
            warnings.warn("shuffle_before_yield=True and sort_before_fetch=False, this decrease the fetching speed and yield the same result. Consider setting sort_before_fetch=True")    
        if shuffle and (not drop_last):
            raise ValueError("shuffle=True requires drop_last=True")
        if (not shuffle) and drop_last:
            raise NotImplementedError("shuffle=False and drop_last=True is not implemented")
                
        self.batch_size = batch_size
        self.block_size = block_size
        self.fetch_factor = fetch_factor
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.sort_before_fetch = sort_before_fetch
        self.shuffle_before_yield = shuffle_before_yield
        self.fetch_size = self.batch_size * self.fetch_factor
        
        # Store callback functions
        self.fetch_transform = fetch_transform
        self.batch_transform = batch_transform
        self.fetch_callback = fetch_callback
        self.batch_callback = batch_callback
        
        self.collection = data_collection
        
        self.indices = np.arange(len(self.collection))
        
    def __len__(self):
        """
        Return the number of batches in the dataset.
        """
        if self.shuffle and self.drop_last:
            # When shuffling and dropping last, we remove remainder to make dataset
            # size divisible by fetch_size
            adjusted_size = len(self.indices) - (len(self.indices) % self.fetch_size)
            return adjusted_size // self.batch_size
        elif not self.drop_last:
            # Include the last batch even if incomplete
            return (len(self.indices) + self.batch_size - 1) // self.batch_size
        else:
            # Just drop the last incomplete batch
            return len(self.indices) // self.batch_size
        
    def __iter__(self):
        """
        Yield batches of data according to the current configuration.
        """
        worker_info = get_worker_info()
        indices = self.indices
        g = None
        if self.shuffle:
            if worker_info is None:
                g = np.random.default_rng()
            else:
                seed = worker_info.seed - worker_info.id
                g = np.random.default_rng(seed=seed)
                
            remainder = len(indices) % self.fetch_size
            
            # Drop randomly selected indices
            if self.drop_last and (remainder != 0):
                remove_indices = g.choice(indices, size=remainder, replace=False)
                mask = ~np.isin(indices, remove_indices)
                indices = indices[mask]
            
            blocks = indices.reshape(-1, self.block_size)
            blocks = g.permutation(blocks, axis=0)
            fetches = blocks.reshape(-1, self.fetch_size)

            if worker_info is not None:
                per_worker = len(fetches) // worker_info.num_workers
                remainder = len(fetches) % worker_info.num_workers
                
                # Distribute remainder among workers
                if worker_info.id < remainder:
                    # First 'remainder' workers get one extra fetch
                    start = worker_info.id * (per_worker + 1)
                    end = start + per_worker + 1
                else:
                    # Other workers get the base number of fetches
                    start = worker_info.id * per_worker + remainder
                    end = start + per_worker
                
                fetches = fetches[start:end]
                
            if self.sort_before_fetch:
                fetches = np.sort(fetches, axis=1)
            
            for fetch_ids in fetches:
                # Use custom fetch callback if provided, otherwise use default indexing
                if self.fetch_callback is not None:
                    data = self.fetch_callback(self.collection, fetch_ids)
                else:
                    data = self.collection[fetch_ids]

                # Call fetch transform if provided
                if self.fetch_transform is not None:
                    data = self.fetch_transform(data)

                if self.shuffle_before_yield:
                    # Shuffle the indices
                    shuffle_indices = g.permutation(len(fetch_ids))
                else:
                    shuffle_indices = np.arange(len(fetch_ids))
                
                # Yield batches
                for i in range(0, len(fetch_ids), self.batch_size):
                    # Use custom batch callback if provided, otherwise use default indexing
                    if self.batch_callback is not None:
                        batch_ids = shuffle_indices[i:i + self.batch_size]
                        batch_data = self.batch_callback(data, batch_ids)
                    else:
                        batch_ids = shuffle_indices[i:i + self.batch_size]
                        batch_data = data[batch_ids]
                    
                    # Call batch transform if provided
                    if self.batch_transform is not None:
                        batch_data = self.batch_transform(batch_data)
                        
                    yield batch_data
                
        else: # Not shuffling indices before fetching
            if worker_info is not None:
                per_worker = len(indices) // worker_info.num_workers
                remainder = len(indices) % worker_info.num_workers
                
                # Distribute remainder among workers
                if worker_info.id < remainder:
                    # First 'remainder' workers get one extra fetch
                    start = worker_info.id * (per_worker + 1)
                    end = start + per_worker + 1
                else:
                    # Other workers get the base number of fetches
                    start = worker_info.id * per_worker + remainder
                    end = start + per_worker
                    
                indices = indices[start:end]

            for i in range(0, len(indices), self.fetch_size):
                ids = indices[i:i + self.fetch_size]
                # Use custom fetch callback if provided, otherwise use default indexing
                if self.fetch_callback is not None:
                    data = self.fetch_callback(self.collection, ids)
                else:
                    data = self.collection[ids]
                
                # Call fetch transform if provided
                if self.fetch_transform is not None:
                    data = self.fetch_transform(data)
                
                # Yield batches
                for j in range(0, len(ids), self.batch_size):
                    # Use custom batch callback if provided, otherwise use default indexing
                    if self.batch_callback is not None:
                        batch_indices = slice(j, j + self.batch_size)
                        batch_data = self.batch_callback(data, batch_indices)
                    else:
                        batch_data = data[j:j + self.batch_size]
                    
                    # Call batch transform if provided
                    if self.batch_transform is not None:
                        batch_data = self.batch_transform(batch_data)
                        
                    yield batch_data

    def set_mode(self, mode):
        """
        Set dataset mode.

        Args:
            mode (str): One of 'train', 'training', 'eval', 'val', 'evaluation', 'test', 'testing'.

        Raises:
            ValueError: If mode is not recognized.
        """
        mode = mode.lower()
        if mode in ['train', 'training']:
            self.shuffle = True
            self.drop_last = True
            self.sort_before_fetch = True
            self.shuffle_before_yield = True
        elif mode in ['eval', 'val', 'evaluation', 'test', 'testing']:
            self.shuffle = False
            self.drop_last = False
            self.sort_before_fetch = False
            self.shuffle_before_yield = False
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Must be 'train', 'training', 'eval', 'val', 'evaluation', 'test', or 'testing'.")

    def subset(self, indices: Union[List[int], np.ndarray]):
        """
        Subset the dataset to only include the specified indices.

        Parameters
        ----------
        indices : List[int] or numpy.ndarray
            Indices to subset the dataset to.
        """
        if not isinstance(indices, (list, np.ndarray)):
            raise TypeError("indices must be a list of integers or a numpy array")
            
        if isinstance(indices, list):
            if any(not isinstance(i, int) for i in indices):
                raise TypeError("All elements in indices must be integers")
        elif not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Numpy array must contain integers")
            
        if any(i < 0 or i >= len(self.collection) for i in indices):
            raise IndexError("Indices out of bounds")
        
        self.indices = np.array(indices)
        
    def reset_indices(self):
        """
        Reset the dataset to include all indices.
        """
        self.indices = np.arange(len(self.collection))