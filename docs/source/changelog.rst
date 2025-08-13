Changelog
=========

This document tracks all notable changes to ``scDataset``.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

[0.2.0] - 2025-08-13
---------------------

Added
~~~~~

* Core ``scDataset`` class with flexible sampling strategies
* Sampling strategies:
  
  * ``Streaming`` - Sequential sampling without shuffling
  * ``BlockShuffling`` - Block-based shuffling for locality
  * ``BlockWeightedSampling`` - Weighted sampling with blocks
  * ``ClassBalancedSampling`` - Automatic class balancing

* Support for multiple data formats:
  
  * NumPy arrays
  * AnnData objects
  * HuggingFace Datasets
  * PyTorch Datasets
  * Any object with ``__getitem__`` and ``__len__``

* Performance optimizations:
  
  * Block-based data fetching
  * Configurable fetch factors
  * Multiprocessing support

* Customization features:
  
  * Custom fetch callbacks
  * Custom batch callbacks
  * Fetch and batch transforms

* Comprehensive documentation and examples
* Test suite with >90% coverage
* GitHub Actions CI/CD pipeline

Technical Details
~~~~~~~~~~~~~~~~~

* Minimum Python version: 3.8
* Core dependencies: ``torch >= 1.2.0``, ``numpy >= 1.17.0``
* Compatible with PyTorch DataLoader
* Supports both eager and lazy data loading patterns

Known Issues
~~~~~~~~~~~~
* Performance may vary depending on storage backend (SSD vs HDD)

.. note::
   This changelog will be updated with each release.
   See the `GitHub releases page <https://github.com/scDataset/scDataset/releases>`_
   for the most up-to-date information.
