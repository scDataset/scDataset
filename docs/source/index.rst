scDataset Documentation
======================

.. image:: https://badge.fury.io/py/scDataset.svg
   :target: https://pypi.org/project/scDataset/
   :alt: PyPI version

.. image:: https://img.shields.io/badge/License-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/arXiv-2506.01883-b31b1b.svg
   :target: https://arxiv.org/abs/2506.01883
   :alt: arXiv

**Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics**

.. image:: https://github.com/Kidara/scDataset/raw/main/figures/scdataset.png
   :alt: scDataset architecture
   :align: center

``scDataset`` is a flexible and efficient PyTorch ``IterableDataset`` for large-scale single-cell omics datasets. 
It supports a variety of data formats (e.g., AnnData, HuggingFace Datasets, NumPy arrays) and is designed for 
high-throughput deep learning workflows. While optimized for single-cell data, it is general-purpose and can be 
used with any dataset.

Key Features
------------

✨ **Flexible Data Source Support**: Integrates seamlessly with AnnData, HuggingFace Datasets, NumPy arrays, PyTorch Datasets, and more.

🚀 **Scalable**: Handles datasets with billions of samples without loading everything into memory.

⚡ **Efficient Data Loading**: Block sampling and batched fetching optimize random access for large datasets.

🔄 **Dynamic Splitting**: Split datasets into train/validation/test dynamically, without duplicating data or rewriting files.

🎯 **Custom Hooks**: Apply transformations at fetch or batch time via user-defined callbacks.

Quick Start
-----------

Install from PyPI:

.. code-block:: bash

   pip install scDataset

Basic usage:

.. code-block:: python

   from scdataset import scDataset, Streaming
   from torch.utils.data import DataLoader
   import numpy as np

   # Create sample data
   data = np.random.randn(10000, 2000)  # 10k cells, 2k genes
   
   # Create dataset with streaming strategy
   dataset = scDataset(data, Streaming(), batch_size=64)
   
   # Use with PyTorch DataLoader
   loader = DataLoader(dataset, num_workers=4)
   
   for batch in loader:
       print(f"Batch shape: {batch.shape}")
       break

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   quickstart
   examples
   scdataset
   
.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:
   
   citation
   contributing
   changelog

