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

.. autoclass:: scDataset
   :members:
   :undoc-members:
   :show-inheritance:

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

Base Strategy Class
~~~~~~~~~~~~~~~~~~~

.. autoclass:: SamplingStrategy
   :members:
   :undoc-members:
   :show-inheritance:

Sequential Strategies
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Streaming
   :members:
   :undoc-members:
   :show-inheritance:

Shuffling Strategies
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: BlockShuffling
   :members:
   :undoc-members:
   :show-inheritance:

Weighted Sampling
~~~~~~~~~~~~~~~~~

.. autoclass:: BlockWeightedSampling
   :members:
   :undoc-members:
   :show-inheritance:

Class Balanced Sampling
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ClassBalancedSampling
   :members:
   :undoc-members:
   :show-inheritance:
