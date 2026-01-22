"""
Experiment runners for training experiments.

To run experiments, use:
    python -m training_experiments.experiments.run_all --all --epochs 1
"""

# Note: We don't import run_all here to avoid RuntimeWarning when
# running as `python -m training_experiments.experiments.run_all`.
# Users should run the module directly or import explicitly:
#   from training_experiments.experiments.run_all import run_experiment

__all__ = ["run_experiment", "run_all_experiments", "main"]
