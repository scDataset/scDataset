"""
Models for multi-task classification on Tahoe-100M dataset.
"""

from training_experiments.models.linear import (
    MultiTaskLinearClassifier,
    MultiTaskLoss,
)

__all__ = ["MultiTaskLinearClassifier", "MultiTaskLoss"]
