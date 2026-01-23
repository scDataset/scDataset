API Reference
=============

This section provides detailed documentation for all classes and functions in ``scDataset``.

Main Dataset Class
------------------

.. currentmodule:: scdataset

.. autosummary::
   :toctree: generated/
   :nosignatures:

   scDataset

Multi-Modal Data Support
-------------------------

.. currentmodule:: scdataset

.. autosummary::
   :toctree: generated/
   :nosignatures:

   MultiIndexable

Transform Functions
-------------------

.. currentmodule:: scdataset.transforms

.. autosummary::
   :toctree: generated/
   :nosignatures:

   adata_to_mindex
   hf_tahoe_to_tensor

Sampling Strategies
-------------------

.. currentmodule:: scdataset.strategy

.. autosummary::
   :toctree: generated/
   :nosignatures:

   SamplingStrategy
   Streaming
   BlockShuffling
   BlockWeightedSampling
   ClassBalancedSampling

Experimental Features
---------------------

.. warning::

   Features in the experimental module are subject to change and may be
   modified significantly or removed entirely in future releases.

.. currentmodule:: scdataset.experimental

.. autosummary::
   :toctree: generated/
   :nosignatures:

   suggest_parameters
   estimate_sample_size
