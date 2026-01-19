"""
Training experiments for scDataset on Tahoe-100M dataset.

This module contains real-world training experiments comparing different data loading
strategies on the Tahoe-100M dataset for multi-task classification.

Tasks:
    1. Cell line classification
    2. Drug classification
    3. Mechanism of action (broad labels)
    4. Mechanism of action (fine labels)

Data Loading Strategies:
    1. Streaming - Sequential access without shuffling
    2. Streaming with Buffer - Sequential with buffer-level shuffling (HuggingFace/Ray style)
    3. Block Shuffling - scDataset with block_size=4
    4. Random Sampling - Full shuffling (block_size=1)
    5. Block Weighted Sampling - Weighted with block_size=4
    6. True Weighted Sampling - Weighted with block_size=1

Model:
    Simple multi-task linear classifier (no MLP) to avoid confounding effects
    from model selection or hyperparameter tuning.
"""

from training_experiments.data import LabelEncoder, TahoeDataLoader
from training_experiments.models import MultiTaskLinearClassifier
from training_experiments.trainers import MultiTaskTrainer
from training_experiments.utils import compute_balanced_weights

__all__ = [
    "TahoeDataLoader",
    "LabelEncoder",
    "MultiTaskLinearClassifier",
    "MultiTaskTrainer",
    "compute_balanced_weights",
]

__version__ = "0.1.0"
