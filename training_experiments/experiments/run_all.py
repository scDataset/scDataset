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
"""

import argparse
import json
import os
import time
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


def run_experiment(
    strategy_name: str,
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    fetch_factor: int = 16,
    num_workers: int = 12,
    min_count_baseline: int = 1000,
    device: str = "cuda",
    save_dir: Optional[str] = None,
    log_interval: int = 100,
    verbose: bool = True,
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
    device : str
        Device to use for training
    save_dir : str, optional
        Directory to save results
    log_interval : int
        Print progress every N batches
    verbose : bool
        Whether to print progress information

    Returns
    -------
    dict
        Experiment results
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running experiment: {strategy_name}")
        print(f"{'='*60}")

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
        if verbose:
            print("Creating data loaders...")

        train_loader, test_loader, label_encoder, feature_dim = create_dataloaders(
            strategy_name=strategy_name,
            batch_size=batch_size,
            fetch_factor=fetch_factor,
            num_workers=num_workers,
            min_count_baseline=min_count_baseline,
            verbose=verbose,
        )

        # Get task dimensions
        task_dims = label_encoder.get_task_dims()

        if verbose:
            print(f"Task dimensions: {task_dims}")
            print(f"Feature dimension: {feature_dim}")

        # Create model
        if verbose:
            print("Creating multi-task linear model...")

        model = create_model(input_dim=feature_dim, task_dims=task_dims)

        if verbose:
            print(f"Model parameters: {model.count_parameters():,}")

        # Create trainer
        trainer = MultiTaskTrainer(model=model, device=device)

        # Train model
        if verbose:
            print("Starting training...")

        training_results = trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            log_interval=log_interval,
            save_dir=exp_dir,
        )

        # Final evaluation
        if verbose:
            print("\nPerforming final evaluation...")

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
                "device": device,
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

        if verbose:
            print("\nExperiment completed successfully!")
            print(f"Total time: {experiment_time:.1f} seconds")
            print("\nFinal metrics:")
            for task_name in TASK_NAMES:
                acc = final_metrics.get(f"{task_name}_accuracy", 0)
                f1 = final_metrics.get(f"{task_name}_f1_macro", 0)
                print(f"  {task_name}: Acc={acc:.4f}, F1={f1:.4f}")

        return results

    except Exception as e:
        print(f"Error in experiment: {str(e)}")
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
    num_epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    fetch_factor: int = 16,
    num_workers: int = 12,
    min_count_baseline: int = 1000,
    device: str = "cuda",
    save_dir: str = "./results",
    strategies: Optional[list] = None,
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
    device : str
        Device to use for training
    save_dir : str
        Directory to save results
    strategies : list, optional
        List of strategies to run. If None, runs all strategies.
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
            device=device,
            save_dir=save_dir,
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
        "--strategy",
        type=str,
        default=None,
        choices=ALL_STRATEGIES,
        help="Specific strategy to run. If not specified, runs all.",
    )

    parser.add_argument("--all", action="store_true", help="Run all strategies")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )

    parser.add_argument(
        "--fetch_factor", type=int, default=16, help="Fetch factor for scDataset"
    )

    parser.add_argument(
        "--num_workers", type=int, default=12, help="Number of workers for DataLoader"
    )

    parser.add_argument(
        "--min_count_baseline",
        type=int,
        default=1000,
        help="Minimum count baseline for weight computation",
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
        default="/home/kidara/raid/volume/scDataset/training_experiments/results",
        help="Directory to save results",
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Print progress every N batches (0 to disable)",
    )

    args = parser.parse_args()

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"

    print(f"Using device: {args.device}")

    if args.all or args.strategy is None:
        # Run all strategies
        print("Running all strategies...")
        results = run_all_experiments(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            fetch_factor=args.fetch_factor,
            num_workers=args.num_workers,
            min_count_baseline=args.min_count_baseline,
            device=args.device,
            save_dir=args.save_dir,
        )

        print("\n" + "=" * 60)
        print("ALL EXPERIMENTS COMPLETED")
        print("=" * 60)

        for strategy, r in results.items():
            status = r.get("status", "unknown")
            if status == "success":
                print(f"\n{strategy}:")
                for task in TASK_NAMES:
                    acc = r["final_metrics"].get(f"{task}_accuracy", 0)
                    print(f"  {task}: {acc:.4f}")
            else:
                print(f"\n{strategy}: FAILED")
    else:
        # Run single strategy
        run_experiment(
            strategy_name=args.strategy,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            fetch_factor=args.fetch_factor,
            num_workers=args.num_workers,
            min_count_baseline=args.min_count_baseline,
            device=args.device,
            save_dir=args.save_dir,
            log_interval=args.log_interval,
        )


if __name__ == "__main__":
    main()
