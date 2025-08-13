Installation
============

Requirements
------------

``scDataset`` requires Python 3.8 or higher and the following dependencies:

* ``torch >= 1.2.0``
* ``numpy >= 1.17.0``

Optional dependencies for specific data formats:

* ``anndata`` - for AnnData support
* ``datasets`` - for HuggingFace Datasets support

Install from PyPI
-----------------

The easiest way to install ``scDataset`` is from PyPI:

.. code-block:: bash

   pip install scDataset

This will install the latest stable release along with all required dependencies.

Install from GitHub
-------------------

To get the latest development version, install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/Kidara/scDataset.git

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/Kidara/scDataset.git
   cd scDataset
   pip install -e .

To install development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

Verify Installation
-------------------

You can verify your installation by importing the package:

.. code-block:: python

   import scdataset
   print(scdataset.__version__)

Or run a quick test:

.. code-block:: python

   from scdataset import scDataset, Streaming
   import numpy as np
   
   # Create test data
   data = np.random.randn(100, 50)
   dataset = scDataset(data, Streaming(), batch_size=10)
   
   # Test iteration
   for batch in dataset:
       print(f"Batch shape: {batch.shape}")
       break
   
   print("Installation successful!")

Troubleshooting
---------------

**ImportError: No module named 'torch'**
   Make sure PyTorch is installed. Visit `pytorch.org <https://pytorch.org/get-started/locally/>`_ for installation instructions.

**Performance Issues**
   For best performance with large datasets, consider installing:
   
   .. code-block:: bash
   
      pip install numba  # For faster numerical operations
