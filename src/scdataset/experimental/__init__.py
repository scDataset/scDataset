"""
Experimental module for scDataset.

This module contains experimental features that are not yet fully stable
and may change in future versions. Use with caution in production code.

.. warning::

   Features in this module are experimental and subject to change.

Available Features
------------------

**Auto-Configuration**

.. autosummary::
   :toctree: generated/

   suggest_parameters
   estimate_sample_size

The auto-configuration utilities help determine optimal parameters
for scDataset based on system resources and data characteristics.
"""

from .auto_config import (
    estimate_sample_size,
    suggest_parameters,
)

__all__ = [
    "estimate_sample_size",
    "suggest_parameters",
]
