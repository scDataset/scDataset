"""
Run all training experiments comparing data loading strategies.

This script runs experiments with all 6 data loading strategies on the
Tahoe-100M dataset for multi-task classification.

Strategies:
1. Streaming - Sequential access without shuffling
2. Streaming with Buffer - Sequential with buffer-level shuffling
3. Block Shuffling - scDataset with block_size=4
4. Random Sampling - Full shuffling (block_size=1)
5. Block Weighted Sampling - Weighted with block_size=4
6. True Weighted Sampling - Weighted with block_size=1

Usage:
    python -m training_experiments.experiments.run_all --all --epochs 1
    python -m training_experiments.experiments.run_all --config my_config.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, Optional

import torch

from training_experiments.data.loader import (
    ALL_STRATEGIES,
    create_dataloaders,
)
from training_experiments.models.linear import TASK_NAMES, create_model
from training_experiments.trainers.multitask import (
    MultiTaskTrainer,
    save_results,
)
from training_experiments.utils.config import (
    get_block_size_for_strategy,
    load_config,
    merge_config_with_args,
)

# Suppress RuntimeWarning about sys.modules (harmless, occurs with python -m)
warnings.filterwarnings(
    "ignore",
    message=".*found in sys.modules after import.*",
    category=RuntimeWarning,
)


def setup_logging(verbose: bool = True) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger("training_experiments")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def run_experiment(
    strategy_name: str,
    num_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    fetch_factor: int = 256,
    num_workers: int = 8,
    min_count_baseline: int = 1000,
    block_size: Optional[int] = None,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    log_interval: int = 100,
    train_plates: Optional[list] = None,
    test_plates: Optional[list] = None,
    max_train_cells: Optional[int] = None,
    max_test_cells: Optional[int] = None,
    verbose: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Run a single experiment with specified strategy.

    Parameters
    ----------
    strategy_name : str
        One of: 'streaming', 'streaming_buffer', 'block_shuffling',
        'random_sampling', 'block_weighted', 'true_weighted'
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    fetch_factor : int
        Fetch factor for scDataset
    num_workers : int
        Number of workers for DataLoader
    min_count_baseline : int
        Minimum count baseline for weight computation
    block_size : int, optional
        Block size for block-based strategies. If None, uses defaults.
    device : str
        Device to use for training
    save_dir : str, optional
        Directory to save results
    log_interval : int
        Print progress every N batches
    train_plates : list, optional
        List of plate numbers for training. Default: [1-13]
    test_plates : list, optional
        List of plate numbers for testing. Default: [14]
    max_train_cells : int, optional
        Maximum number of training cells (pilot mode). If None, uses all cells.
    max_test_cells : int, optional
        Maximum number of test cells (pilot mode). If None, uses all cells.
    verbose : bool
        Whether to print progress information
    logger : logging.Logger, optional
        Logger instance. If None, uses print statements.

    Returns
    -------
    dict
        Experiment results
    """
    # Use logger if provided, otherwise fall back to print
    if logger is None:
        logger = logging.getLogger("training_experiments")
        if not logger.handlers:
            logger = setup_logging(verbose)

    logger.info("=" * 60)
    logger.info(f"Running experiment: {strategy_name}")
    logger.info("=" * 60)

    # Create experiment directory
    if save_dir:
        exp_name = f"{strategy_name}_{batch_size}bs_{learning_rate}lr"
        exp_dir = os.path.join(save_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
    else:
        exp_dir = None

    experiment_start = time.time()

    try:
        # Create data loaders
        logger.info("Creating data loaders...")

        train_loader, test_loader, label_encoder, feature_dim = create_dataloaders(
            strategy_name=strategy_name,
            batch_size=batch_size,
            fetch_factor=fetch_factor,
            num_workers=num_workers,
            min_count_baseline=min_count_baseline,
            block_size=block_size,
            train_plates=train_plates,
            test_plates=test_plates,
            max_train_cells=max_train_cells,
            max_test_cells=max_test_cells,
            verbose=verbose,
        )

        # Get task dimensions
        task_dims = label_encoder.get_task_dims()

        logger.info(f"Task dimensions: {task_dims}")
        logger.info(f"Feature dimension: {feature_dim}")

        # Create model
        logger.info("Creating multi-task linear model...")

        model = create_model(input_dim=feature_dim, task_dims=task_dims)

        logger.info(f"Model parameters: {model.count_parameters():,}")

        # Create trainer
        trainer = MultiTaskTrainer(model=model, device=device)

        # Train model
        logger.info("Starting training...")

        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            log_interval=log_interval,
            save_dir=exp_dir,
        )

        # Final evaluation
        logger.info("Performing final evaluation...")

        final_metrics = trainer.evaluate(test_loader)

        experiment_time = time.time() - experiment_start

        # Compile results
        results = {
            "status": "success",
            "experiment_config": {
                "strategy_name": strategy_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "fetch_factor": fetch_factor,
                "num_workers": num_workers,
                "min_count_baseline": min_count_baseline,
                "block_size": block_size,
                "device": device,
                "train_plates": train_plates,
                "test_plates": test_plates,
                "max_train_cells": max_train_cells,
                "max_test_cells": max_test_cells,
            },
            "data_info": {"task_dims": task_dims, "feature_dim": feature_dim},
            "training_results": training_results,
            "final_metrics": final_metrics,
            "experiment_time_seconds": experiment_time,
        }

        # Save results
        if exp_dir:
            results_file = os.path.join(exp_dir, "results.pkl")
            save_results(results, results_file)

            # Also save as JSON for easy viewing
            json_results = {
                k: v
                for k, v in results.items()
                if k not in ["training_results"]  # Skip large history
            }
            json_file = os.path.join(exp_dir, "results.json")
            with open(json_file, "w") as f:
                json.dump(json_results, f, indent=2, default=str)

        logger.info("Experiment completed successfully!")
        logger.info(f"Total time: {experiment_time:.1f} seconds")
        logger.info("Final metrics:")
        for task_name in TASK_NAMES:
            acc = final_metrics.get(f"{task_name}_accuracy", 0)
            f1 = final_metrics.get(f"{task_name}_f1_macro", 0)
            logger.info(f"  {task_name}: Acc={acc:.4f}, F1={f1:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error in experiment: {str(e)}")
        import traceback

        traceback.print_exc()

        error_results = {
            "status": "failed",
            "experiment_config": {
                "strategy_name": strategy_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
            "error": str(e),
            "experiment_time_seconds": time.time() - experiment_start,
        }

        if exp_dir:
            error_file = os.path.join(exp_dir, "error.json")
            with open(error_file, "w") as f:
                json.dump(error_results, f, indent=2)

        return error_results


def run_all_experiments(
    num_epochs: int = 1,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    fetch_factor: int = 256,
    num_workers: int = 8,
    min_count_baseline: int = 1000,
    block_size: Optional[int] = None,
    device: str = "cuda",
    save_dir: str = "./results",
    strategies: Optional[list] = None,
    train_plates: Optional[list] = None,
    test_plates: Optional[list] = None,
    max_train_cells: Optional[int] = None,
    max_test_cells: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all experiments with all strategies.

    Parameters
    ----------
    num_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    fetch_factor : int
        Fetch factor for scDataset
    num_workers : int
        Number of workers for DataLoader
    min_count_baseline : int
        Minimum count baseline for weight computation
    block_size : int, optional
        Block size for block-based strategies. If None, uses defaults.
    device : str
        Device to use for training
    save_dir : str
        Directory to save results
    strategies : list, optional
        List of strategies to run. If None, runs all strategies.
    train_plates : list, optional
        List of plate numbers for training. Default: [1-13]
    test_plates : list, optional
        List of plate numbers for testing. Default: [14]
    max_train_cells : int, optional
        Maximum number of training cells (pilot mode). If None, uses all cells.
    max_test_cells : int, optional
        Maximum number of test cells (pilot mode). If None, uses all cells.
    verbose : bool
        Whether to print progress information

    Returns
    -------
    dict
        Dictionary of results for each strategy
    """
    if strategies is None:
        strategies = ALL_STRATEGIES

    all_results = {}

    for strategy_name in strategies:
        results = run_experiment(
            strategy_name=strategy_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            fetch_factor=fetch_factor,
            num_workers=num_workers,
            min_count_baseline=min_count_baseline,
            block_size=block_size,
            device=device,
            save_dir=save_dir,
            train_plates=train_plates,
            test_plates=test_plates,
            max_train_cells=max_train_cells,
            max_test_cells=max_test_cells,
            verbose=verbose,
        )
        all_results[strategy_name] = results

    # Save summary
    if save_dir:
        summary_file = os.path.join(save_dir, "summary.json")
        summary = {
            strategy: {
                "status": r.get("status"),
                "final_accuracy": {
                    task: r.get("final_metrics", {}).get(f"{task}_accuracy", 0)
                    for task in TASK_NAMES
                }
                if r.get("status") == "success"
                else None,
                "time": r.get("experiment_time_seconds"),
            }
            for strategy, r in all_results.items()
        }
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Tahoe-100M multi-task training experiments"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. If not specified, uses default config.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=ALL_STRATEGIES,
        help="Specific strategy to run. If not specified, runs all.",
    )

    parser.add_argument("--all", action="store_true", help="Run all strategies")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--fetch_factor",
        type=int,
        default=None,
        help="Fetch factor for scDataset (overrides config)",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for DataLoader (overrides config)",
    )

    parser.add_argument(
        "--min_count_baseline",
        type=int,
        default=None,
        help="Minimum count baseline for weight computation (overrides config)",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block size for block-based strategies (overrides config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save results (overrides config)",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=None,
        help="Print progress every N batches (overrides config)",
    )

    # Pilot mode arguments
    parser.add_argument(
        "--max_train_cells",
        type=int,
        default=None,
        help="Max training cells for pilot mode (overrides config)",
    )

    parser.add_argument(
        "--max_test_cells",
        type=int,
        default=None,
        help="Max test cells for pilot mode (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Merge config with CLI arguments (CLI takes precedence)
    merged = merge_config_with_args(
        config,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fetch_factor=args.fetch_factor,
        num_workers=args.num_workers,
        min_count_baseline=args.min_count_baseline,
        block_size=args.block_size,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        max_train_cells=args.max_train_cells,
        max_test_cells=args.max_test_cells,
    )

    # Set up logging
    logger = setup_logging(verbose=True)

    logger.info("Configuration loaded:")
    for key, value in merged.items():
        logger.info(f"  {key}: {value}")

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = "cpu"

    logger.info(f"Using device: {args.device}")

    if args.all or args.strategy is None:
        # Run all strategies
        logger.info("Running all strategies...")
        results = run_all_experiments(
            num_epochs=merged["num_epochs"],
            batch_size=merged["batch_size"],
            learning_rate=merged["learning_rate"],
            fetch_factor=merged["fetch_factor"],
            num_workers=merged["num_workers"],
            min_count_baseline=merged["min_count_baseline"],
            block_size=merged["block_size"],
            device=args.device,
            save_dir=merged["save_dir"],
            train_plates=merged["train_plates"],
            test_plates=merged["test_plates"],
            max_train_cells=merged["max_train_cells"],
            max_test_cells=merged["max_test_cells"],
        )

        logger.info("=" * 60)
        logger.info("ALL EXPERIMENTS COMPLETED")
        logger.info("=" * 60)

        for strategy, r in results.items():
            status = r.get("status", "unknown")
            if status == "success":
                logger.info(f"{strategy}:")
                for task in TASK_NAMES:
                    acc = r["final_metrics"].get(f"{task}_accuracy", 0)
                    logger.info(f"  {task}: {acc:.4f}")
            else:
                logger.warning(f"{strategy}: FAILED")
    else:
        # Get strategy-specific block_size from config if not overridden
        strategy_block_size = get_block_size_for_strategy(
            config, args.strategy, args.block_size
        )

        # Run single strategy
        run_experiment(
            strategy_name=args.strategy,
            num_epochs=merged["num_epochs"],
            batch_size=merged["batch_size"],
            learning_rate=merged["learning_rate"],
            fetch_factor=merged["fetch_factor"],
            num_workers=merged["num_workers"],
            min_count_baseline=merged["min_count_baseline"],
            block_size=strategy_block_size,
            device=args.device,
            save_dir=merged["save_dir"],
            log_interval=merged["log_interval"],
            train_plates=merged["train_plates"],
            test_plates=merged["test_plates"],
            max_train_cells=merged["max_train_cells"],
            max_test_cells=merged["max_test_cells"],
            logger=logger,
        )


if __name__ == "__main__":
    main()
