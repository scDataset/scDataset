Citation
========

If you use ``scDataset`` in your research, please cite our paper:

BibTeX
------

.. code-block:: bibtex

   @article{scdataset2025,
     title={scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics},
     author={D'Ascenzo, Davide and Cultrera di Montesano, Sebastiano},
     journal={arXiv preprint arXiv:2506.01883},
     year={2025}
   }

Paper Abstract
--------------

Modern single-cell datasets now comprise hundreds of millions of cells, presenting significant challenges for training deep learning models that require shuffled, memory-efficient data loading. While the AnnData format is the community standard for storing single-cell datasets, existing data loading solutions for AnnData are often inadequate: some require loading all data into memory, others convert to dense formats that increase storage demands, and many are hampered by slow random disk access. We present scDataset, a PyTorch IterableDataset that operates directly on one or more AnnData files without the need for format conversion. The core innovation is a combination of block sampling and batched fetching, which together balance randomness and I/O efficiency. On the Tahoe 100M dataset, scDataset achieves up to a 48× speed-up over AnnLoader, a 27× speed-up over HuggingFace Datasets, and an 18× speed-up over BioNeMo in single-core settings. These advances democratize large-scale single-cell model training for the broader research community.

Links
-----

* `arXiv Paper <https://arxiv.org/abs/2506.01883>`_
* `GitHub Repository <https://github.com/scDataset/scDataset>`_
* `PyPI Package <https://pypi.org/project/scDataset/>`_

Related Work
------------

``scDataset`` builds upon and complements several important tools in the single-cell analysis ecosystem:

* `AnnData <https://anndata.readthedocs.io/>`_ - Annotated data format for single-cell data
* `Scanpy <https://scanpy.readthedocs.io/>`_ - Single-cell analysis in Python
* `HuggingFace Datasets <https://huggingface.co/docs/datasets/>`_ - Dataset library for machine learning
* `PyTorch <https://pytorch.org/>`_ - Deep learning framework
