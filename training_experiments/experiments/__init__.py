"""
Experiment runners for training experiments.

To run experiments, use:
    python -m training_experiments.experiments.run_all --all --epochs 1

For pilot experiments (quick testing), use the pilot config:
    python -m training_experiments.experiments.run_all --config training_experiments/configs/pilot.yaml --all
"""

# Note: We don't import run_all here to avoid RuntimeWarning when
# running as `python -m training_experiments.experiments.run_all`.
# Users should run the module directly or import explicitly:
#   from training_experiments.experiments.run_all import run_experiment

__all__ = [
    "run_experiment",
    "run_all_experiments",
    "main",
]
