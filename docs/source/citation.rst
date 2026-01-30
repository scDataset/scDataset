Citation
========

If you use ``scDataset`` in your research, please cite our paper:

BibTeX
------

.. code-block:: bibtex

   @article{scdataset2025,
     title={scDataset: Scalable Data Loading for Deep Learning on Large-Scale Single-Cell Omics},
     author={D'Ascenzo, Davide and Cultrera di Montesano, Sebastiano},
     journal={arXiv:2506.01883},
     year={2025}
   }

Paper Abstract
--------------

Training deep learning models on single-cell datasets with hundreds of millions of cells requires loading data from disk, as these datasets exceed available memory. While random sampling provides the data diversity needed for effective training, it is prohibitively slow due to the random access pattern overhead, whereas sequential streaming achieves high throughput but introduces biases that degrade model performance. We present scDataset, a PyTorch data loader that enables efficient training from on-disk data with seamless integration across diverse storage formats. Our approach combines block sampling and batched fetching to achieve quasi-random sampling that balances I/O efficiency with minibatch diversity. On Tahoe-100M, a dataset of 100 million cells, scDataset achieves more than two orders of magnitude speedup compared to true random sampling while working directly with AnnData files. We provide theoretical bounds on minibatch diversity and empirically show that scDataset matches the performance of true random sampling across multiple classification tasks.

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
