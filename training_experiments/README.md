# Training Experiments for scDataset

This module contains real-world training experiments comparing different data loading
strategies on the Tahoe-100M dataset for multi-task classification.

## Overview

We train a simple linear model to avoid confounding effects from model
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

3. **Block Shuffling** - scDataset with block_size=16, fetch_factor=256
   (`BlockShuffling(block_size=16)`)

4. **Random Sampling** - Full shuffling without replacement (block_size=1, fetch_factor=256)
   (`BlockShuffling(block_size=1)`)

5. **Block Weighted Sampling** - Weighted sampling with replacement using balanced weights
   for (cell_line, drug) combinations, with block_size=16, fetch_factor=256
   (`BlockWeightedSampling(block_size=16)`)

6. **True Weighted Sampling** - Weighted sampling with block_size=1 to mimic true weighted
   sampling, fetch_factor=256 (`BlockWeightedSampling(block_size=1)`)

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

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

This module expects the Tahoe-100M dataset in h5ad format at:
`/path/to/tahoe-100m/h5ad/`

Training uses plates 1-13, testing uses plate 14.

You can modify the path in `configs/default.yaml` or `data/loader.py`.

## Running Experiments

### Run a single experiment

```bash
python -m training_experiments.experiments.run_all --strategy streaming --epochs 1
```

Available strategies:
- `streaming` - Sequential without shuffling
- `streaming_buffer` - Sequential with buffer shuffling
- `block_shuffling` - Block shuffling with block_size=16
- `random` - Random sampling (block_size=1)
- `weighted_block` - Weighted sampling with block_size=16
- `weighted_true` - Weighted sampling with block_size=1

### Run all experiments

```bash
python -m training_experiments.experiments.run_all --all --epochs 1
```

### Configuration

The experiments use a YAML configuration file to set all parameters. The default config
is loaded from `configs/default.yaml`.

**Using a custom config:**
```bash
python -m training_experiments.experiments.run_all --config my_config.yaml --all
```

**Overriding specific values from CLI:**
```bash
python -m training_experiments.experiments.run_all --batch_size 128 --epochs 1 --all
```

CLI arguments always override config file values.

**Configuration file structure (`configs/default.yaml`):**

```yaml
# Dataset paths
data:
  h5ad_dir: "/path/to/h5ad/files"
  label_dir: null  # Uses bundled mappings by default

# Training parameters
training:
  batch_size: 64
  num_epochs: 1
  learning_rate: 0.001

# scDataset parameters
scdataset:
  fetch_factor: 256
  num_workers: 8

# Weight computation for weighted sampling
weights:
  min_count_baseline: 1000

# Strategy-specific configurations
strategies:
  streaming:
    enabled: true
    shuffle: false
  streaming_buffer:
    enabled: true
    shuffle: true
  block_shuffling:
    enabled: true
    block_size: 16
  random_sampling:
    enabled: true
    block_size: 1
  block_weighted:
    enabled: true
    block_size: 16
  true_weighted:
    enabled: true
    block_size: 1

# Output
output:
  save_dir: "./results"
  log_interval: 100
```

## Pilot Experiments (Quick Testing)

Before running the full experiments, you can run a pilot to verify all strategies work.
The pilot uses the same code as the full experiment, just with a different config:

```bash
# Run pilot with all 6 strategies (100K train cells, 50K test cells, 1 epoch)
python -m training_experiments.experiments.run_all --config training_experiments/configs/pilot.yaml --all

# Run pilot with specific strategy
python -m training_experiments.experiments.run_all --config training_experiments/configs/pilot.yaml --strategy streaming

# Override cell counts from command line
python -m training_experiments.experiments.run_all --config training_experiments/configs/pilot.yaml --all --max_train_cells 50000 --max_test_cells 25000
```

The pilot config (`configs/pilot.yaml`):
- Uses only plate 1 for training and plate 14 for testing
- Limits to 100,000 training cells and 50,000 test cells
- Runs for a single epoch
- Tests all 6 strategies to ensure they work without errors

This is useful for:
- Verifying the installation and data paths are correct
- Quick debugging and development
- Testing code changes before full experiments

## Plotting Results

The module includes plotting utilities for visualizing experiment results.

### Using the Notebook

Open and run `notebooks/analyze_results.ipynb` for interactive analysis:

```bash
jupyter notebook notebooks/analyze_results.ipynb
```

This notebook provides:
- Per-task performance comparison (Macro F1-score)
- Training curves across epochs
- Time comparison across strategies

### Programmatic Usage

```python
from training_experiments.analysis.plotting import (
    plot_comparison,
    plot_training_curves,
    plot_time_comparison
)

# Load results
import pandas as pd
df = pd.read_csv('results/summary.csv')

# Create comparison plot (2x2 grid for all tasks)
fig, axes = plot_comparison(df, title='Strategy Comparison')
fig.savefig('figures/comparison.pdf')

# Plot training curves for a specific task
fig, ax = plot_training_curves(results, task='cell_line')
fig.savefig('figures/training_curves.pdf')
```

### Available Plot Functions

- `plot_comparison()` - Create 2x2 comparison plot for all tasks
- `plot_training_curves()` - Plot training curves over epochs
- `plot_time_comparison()` - Compare training time across strategies

## Directory Structure

```
training_experiments/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── __init__.py                  # Package init
├── analysis/
│   ├── __init__.py             # Result loading utilities
│   └── plotting.py             # Plotting utilities
├── configs/
│   ├── default.yaml            # Default configuration
│   └── pilot.yaml              # Pilot experiment configuration
├── data/
│   ├── __init__.py
│   ├── loader.py               # Tahoe dataset loading utilities
│   ├── label_encoder.py        # Label encoding for all 4 tasks
│   └── mappings/               # Label mapping pickle files
├── experiments/
│   ├── __init__.py
│   └── run_all.py              # Run experiments (full or pilot via config)
├── models/
│   ├── __init__.py
│   └── linear.py               # Multi-task linear classifier
├── notebooks/
│   └── analyze_results.ipynb   # Interactive analysis notebook
├── results/                     # Experiment outputs (gitignored)
├── tests/
│   ├── test_config.py          # Test configuration loading
│   ├── test_data_loading.py    # Test data loading strategies
│   ├── test_models.py          # Test model architecture
│   └── test_weights.py         # Test weight computation
├── trainers/
│   ├── __init__.py
│   └── multitask.py            # Multi-task trainer
└── utils/
    ├── __init__.py
    ├── config.py               # Configuration loading utilities
    └── weights.py              # Weight computation utilities
```

## Running Tests

```bash
pytest training_experiments/tests/ -v
```

## Reproducing Results

To fully reproduce the results from the paper:

1. **Download the dataset** from the official source
2. **Update paths** in `configs/default.yaml`
3. **Run all experiments**:
   ```bash
   python -m training_experiments.experiments.run_all --all --epochs 1
   ```
4. **Generate plots**:
   Open `notebooks/analyze_results.ipynb` in Jupyter
