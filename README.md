# scDataset

[![PyPI version](https://badge.fury.io/py/scDataset.svg)](https://pypi.org/project/scDataset/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01883-b31b1b.svg)](https://arxiv.org/abs/2506.01883)

Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics

---

![scDataset architecture](https://github.com/Kidara/scDataset/raw/main/figures/scdataset.png)

**scDataset** is a flexible and efficient PyTorch `IterableDataset` for large-scale single-cell omics datasets. It supports a variety of data formats (e.g., AnnData, HuggingFace Datasets, NumPy arrays) and is designed for high-throughput deep learning workflows. While optimized for single-cell data, it is general-purpose and can be used with any dataset.

## Features

- **Flexible Data Source Support**: Integrates seamlessly with AnnData, HuggingFace Datasets, NumPy arrays, PyTorch Datasets, and more.
- **Scalable**: Handles datasets with billions of samples without loading everything into memory.
- **Efficient Data Loading**: Block sampling and batched fetching optimize random access for large datasets.
- **Dynamic Splitting**: Split datasets into train/validation/test dynamically, without duplicating data or rewriting files.
- **Custom Hooks**: Apply transformations at fetch or batch time via user-defined callbacks.

## Installation

Install the latest release from PyPI:

```bash
pip install scDataset
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/Kidara/scDataset.git
```

## Usage

### Minimal Integration

You can use `scDataset` as a drop-in replacement for your existing dataset class with minimal changes:

```python
from scdataset import scDataset
from torch.utils.data import DataLoader

dataset = MyDataset()
sc_dataset = scDataset(data_collection=dataset, batch_size=64)
loader = DataLoader(sc_dataset, batch_size=None)  # scDataset handles batching internally
```

> **Note:** Set `batch_size=None` in the DataLoader to delegate batching to `scDataset`.

### Dataset Splitting

Split datasets into training, validation, and test sets dynamically, without duplicating or rewriting data:

```python
dataset = MyDataset()
train_indices = ...  # e.g., a list or numpy array of indices

sc_dataset = scDataset(data_collection=dataset, batch_size=64)
sc_dataset.subset(train_indices)  # Specify indices for the split
loader = DataLoader(sc_dataset, batch_size=None)
```

### Training and Evaluation Modes

Switch between training (random access) and evaluation (streaming) modes at any time:

```python
sc_dataset = scDataset(data_collection=dataset, batch_size=64)
sc_dataset.set_mode('train')  # or 'eval'
loader = DataLoader(sc_dataset, batch_size=None)
```

### Block Sampling and Batched Fetching

Optimize data loading for large datasets by configuring `block_size` and `fetch_factor`:

```python
sc_dataset = scDataset(
    data_collection=dataset,
    batch_size=64,
    block_size=16,
    fetch_factor=8
)
loader = DataLoader(
    sc_dataset,
    batch_size=None,
    prefetch_factor=9  # fetch_factor + 1
)
```
We recommend setting `prefetch_factor` to `fetch_factor + 1` for efficient data loading. For parameter details, see the [original paper](https://arxiv.org/abs/2506.01883).

### Custom Hooks

Apply custom transformations at fetch or batch time using hooks. This enables on-the-fly preprocessing without modifying the original dataset.

#### Hook Overview

- **`fetch_callback(collection, ids)`**:  
  Customizes how samples are fetched from the underlying data collection.  
  Use this if your collection does not support batched indexing or requires special access logic.  
  - **Input:** the data collection and an array of indices  
  - **Output:** the fetched data

- **`fetch_transform(fetched_data)`**:  
  Transforms each fetched sample (e.g., sparse-to-dense conversion, normalization).  
  - **Input:** the fetched data  
  - **Output:** the transformed data

- **`batch_callback(fetched_data, ids)`**:  
  Selects or arranges a minibatch from the fetched/transformed data. Similar to `fetch_callback`, but operates on already fetched data.
  - **Input:** the fetched/transformed data and a list of batch indices  
  - **Output:** the batch to yield

- **`batch_transform(batch)`**:  
  Applies final processing to each batch before yielding (e.g., collation, augmentation).  
  - **Input:** the batch  
  - **Output:** the processed batch

```python
def fetch_transform(sample):
    # Example: convert sparse to dense, normalization, etc.
    return sample

def batch_transform(batch):
    # Example: batch-level augmentation or collation
    return batch

sc_dataset = scDataset(
    data_collection=dataset,
    batch_size=64,
    fetch_transform=fetch_transform,
    batch_transform=batch_transform
)
```

#### Full Example

```python
dataset = MyDataset()
train_indices = ...  # training indices
val_indices = ...    # validation indices

sc_dataset = scDataset(
    data_collection=dataset,
    batch_size=64,
    block_size=16,
    fetch_factor=8
)

# Training
sc_dataset.subset(train_indices)
sc_dataset.set_mode('train')
train_loader = DataLoader(
    sc_dataset,
    batch_size=None,
    num_workers=4,
    prefetch_factor=9
)
for batch in train_loader:
    # Training code here
    ...

# Validation
sc_dataset.subset(val_indices)
sc_dataset.set_mode('eval')
val_loader = DataLoader(
    sc_dataset,
    batch_size=None,
    num_workers=4,
    prefetch_factor=9
)
for batch in val_loader:
    # Validation code here
    ...
```

> **Important:**  
> `subset()` and `set_mode()` modify the `scDataset` object in-place.  
> - Call them before each use (not just before DataLoader creation).
> - Complete all code using the current split/mode before changing to another.
> - **Do not** interleave usage of different splits or modes with the same `scDataset` object, as its state will change.

## Citing

If you use `scDataset` in your research, please cite the following paper:

```
@article{scdataset2025,
  title={scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics},
  author={D'Ascenzo, Davide and Cultrera di Montesano, Sebastiano},
  journal={arXiv preprint arXiv:2506.01883},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open issues or pull requests on [GitHub](https://github.com/Kidara/scDataset).

---
