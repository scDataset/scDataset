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