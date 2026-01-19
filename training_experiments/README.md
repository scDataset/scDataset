# Training Experiments for scDataset

This module contains real-world training experiments comparing different data loading
strategies on the Tahoe-100M dataset for multi-task classification.

## Overview

We train a simple **linear model** (no MLP) to avoid confounding effects from model
selection or hyperparameter tuning. The model consists of 4 separate linear layers,
one for each task, combined for computational efficiency since training is I/O
bottlenecked rather than compute-bound.

## Tasks

We evaluate 4 classification tasks:

1. **Cell Line Classification** - Classify cells by cell line
2. **Drug Classification** - Classify cells by drug treatment
3. **Mechanism of Action (Broad)** - Classify by broad MOA labels (provided by dataset authors)
4. **Mechanism of Action (Fine)** - Classify by fine-grained MOA labels (provided by dataset authors)

## Data Loading Strategies

We compare 6 data loading strategies:

1. **Streaming** - Sequential access without shuffling (`Streaming(shuffle=False)`)

2. **Streaming with Buffer** - Sequential with buffer-level shuffling like HuggingFace/Ray
   (`Streaming(shuffle=True)`)

3. **Block Shuffling** - scDataset with block_size=4, fetch_factor=16
   (`BlockShuffling(block_size=4)`)

4. **Random Sampling** - Full shuffling without replacement (block_size=1, fetch_factor=16)
   (`BlockShuffling(block_size=1)`)

5. **Block Weighted Sampling** - Weighted sampling with replacement using balanced weights
   for (cell_line, drug) combinations, with block_size=4, fetch_factor=16
   (`BlockWeightedSampling(block_size=4)`)

6. **True Weighted Sampling** - Weighted sampling with block_size=1 to mimic true weighted
   sampling, fetch_factor=16 (`BlockWeightedSampling(block_size=1)`)

## Weight Calculation

For weighted sampling strategies, weights are computed to balance (cell_line, drug)
tuple combinations, similar to `sklearn.utils.class_weight.compute_class_weight` but
with a twist to handle very low count combinations.

**Problem**: Some combinations have 20,000 cells while others have only 30 cells.
Simple balanced sampling would oversample the 30-cell combinations, risking dataset
memorization.

**Solution**: We add a constant factor of 1000 to all combination counts before
computing the balancing weights. This suppresses extreme reweighting:

```python
# For each unique (cell_line, drug) combination:
count = actual_count + MIN_COUNT_BASELINE  # MIN_COUNT_BASELINE = 1000
weight = total_cells / (n_combinations * count)
```

This ensures that rare combinations are upsampled, but not so aggressively as to
cause memorization issues.

## Directory Structure

```
training_experiments/
├── README.md                    # This file
├── __init__.py                  # Package init
├── data/
│   ├── __init__.py
│   ├── loader.py               # Tahoe dataset loading utilities
│   └── label_encoder.py        # Label encoding for all 4 tasks
├── models/
│   ├── __init__.py
│   └── linear.py               # Multi-task linear classifier
├── trainers/
│   ├── __init__.py
│   └── multitask.py            # Multi-task trainer
├── utils/
│   ├── __init__.py
│   └── weights.py              # Weight computation utilities
├── experiments/
│   ├── __init__.py
│   ├── base.py                 # Base experiment runner
│   └── run_all.py              # Run all experiments
├── configs/
│   └── default.yaml            # Default configuration
└── tests/
    ├── test_weights.py         # Test weight computation
    ├── test_models.py          # Test model architecture
    └── test_data_loading.py    # Test data loading strategies
```

## Usage

### Run a single experiment

```bash
python -m training_experiments.experiments.run_all --strategy streaming --epochs 10
```

### Run all experiments

```bash
python -m training_experiments.experiments.run_all --all --epochs 10
```

## Dataset

This module expects the Tahoe-100M dataset in h5ad format at:
`/home/kidara/raid/volume/vevo-data/2025-02-25/original_h5ad/`

Training uses plates 1-13, testing uses plate 14.

## Requirements

- PyTorch
- scDataset
- anndata
- scikit-learn
- numpy
- pandas
