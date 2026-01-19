"""
Utilities for analyzing and visualizing experiment results.

This module provides functions for loading experiment results and generating
publication-quality plots comparing different data loading strategies.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Strategy display names for plots
STRATEGY_DISPLAY_NAMES = {
    "streaming": "Streaming",
    "streaming_buffer": "Streaming (buffer)",
    "block_shuffling": "Block size = 4\nFetch factor = 16",
    "random_sampling": "Random Sampling",
    "block_weighted": "Block Weighted\n(block_size = 4)",
    "true_weighted": "True Weighted\n(block_size = 1)",
}

# Task display names for plots
TASK_DISPLAY_NAMES = {
    "cell_line": "Cell line",
    "drug": "Drug",
    "moa_broad": "MOA (broad)",
    "moa_fine": "MOA (fine)",
}

# Default y-axis limits for each task (can be adjusted based on results)
DEFAULT_YLIM = {
    "cell_line": (0.0, 1.0),
    "drug": (0.0, 0.1),
    "moa_broad": (0.0, 0.3),
    "moa_fine": (0.0, 0.15),
}


def load_experiment_results(
    results_dir: str, strategies: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load experiment results from a directory.

    Parameters
    ----------
    results_dir : str
        Directory containing experiment results
    strategies : list, optional
        List of strategy names to load. If None, loads all available.

    Returns
    -------
    dict
        Dictionary mapping strategy names to their results
    """
    results = {}

    results_path = Path(results_dir)

    if strategies is None:
        strategies = list(STRATEGY_DISPLAY_NAMES.keys())

    for strategy in strategies:
        # Look for results in subdirectories
        for subdir in results_path.iterdir():
            if subdir.is_dir() and strategy in subdir.name:
                # Try to load results.pkl
                pkl_file = subdir / "results.pkl"
                if pkl_file.exists():
                    with open(pkl_file, "rb") as f:
                        results[strategy] = pickle.load(f)
                    break

                # Fallback to results.json
                json_file = subdir / "results.json"
                if json_file.exists():
                    with open(json_file) as f:
                        results[strategy] = json.load(f)
                    break

    return results


def extract_metrics(
    results: Dict[str, Dict[str, Any]], metric_name: str = "f1_macro"
) -> pd.DataFrame:
    """
    Extract metrics from experiment results into a DataFrame.

    Parameters
    ----------
    results : dict
        Dictionary of experiment results
    metric_name : str
        Metric to extract ('accuracy', 'f1_macro', 'f1_weighted')

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: Task, Method, Value
    """
    tasks = ["cell_line", "drug", "moa_broad", "moa_fine"]

    data = []

    for strategy, result in results.items():
        if result.get("status") != "success":
            continue

        final_metrics = result.get("final_metrics", {})

        for task in tasks:
            metric_key = f"{task}_{metric_name}"
            value = final_metrics.get(metric_key, 0.0)

            data.append(
                {
                    "Task": TASK_DISPLAY_NAMES[task],
                    "Task_key": task,
                    "Method": STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                    "Strategy_key": strategy,
                    "Value": value,
                }
            )

    return pd.DataFrame(data)


def compute_summary_statistics(
    results_list: List[Dict[str, Dict[str, Any]]], metric_name: str = "f1_macro"
) -> pd.DataFrame:
    """
    Compute summary statistics from multiple experiment runs.

    Parameters
    ----------
    results_list : list
        List of experiment results dictionaries (from multiple runs)
    metric_name : str
        Metric to summarize

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: Task, Method, Mean, Std
    """
    all_data = []

    for results in results_list:
        df = extract_metrics(results, metric_name)
        all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Group and compute statistics
    summary = (
        combined.groupby(["Task", "Task_key", "Method", "Strategy_key"])
        .agg(Mean=("Value", "mean"), Std=("Value", "std"), Count=("Value", "count"))
        .reset_index()
    )

    # Fill NaN std with 0 for single runs
    summary["Std"] = summary["Std"].fillna(0)

    return summary


def create_comparison_dataframe(
    results: Dict[str, Dict[str, Any]], metric_name: str = "f1_macro"
) -> pd.DataFrame:
    """
    Create a comparison DataFrame for plotting.

    Parameters
    ----------
    results : dict
        Dictionary of experiment results
    metric_name : str
        Metric to compare

    Returns
    -------
    pandas.DataFrame
        DataFrame formatted for plotting with seaborn
    """
    tasks = ["cell_line", "drug", "moa_broad", "moa_fine"]
    strategies = [
        "streaming",
        "streaming_buffer",
        "block_shuffling",
        "random_sampling",
        "block_weighted",
        "true_weighted",
    ]

    data = []

    for strategy in strategies:
        if strategy not in results:
            continue

        result = results[strategy]
        if result.get("status") != "success":
            continue

        final_metrics = result.get("final_metrics", {})

        for task in tasks:
            metric_key = f"{task}_{metric_name}"
            value = final_metrics.get(metric_key, 0.0)

            data.append(
                {
                    "Task": TASK_DISPLAY_NAMES[task],
                    "Method": STRATEGY_DISPLAY_NAMES[strategy],
                    "Macro F1-score": value,
                    "Error": 0.0,  # Can be updated with std if multiple runs
                }
            )

    return pd.DataFrame(data)


def get_time_comparison(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract training time comparison from results.

    Parameters
    ----------
    results : dict
        Dictionary of experiment results

    Returns
    -------
    pandas.DataFrame
        DataFrame with training times
    """
    data = []

    for strategy, result in results.items():
        if result.get("status") != "success":
            continue

        data.append(
            {
                "Method": STRATEGY_DISPLAY_NAMES.get(strategy, strategy),
                "Strategy_key": strategy,
                "Time (s)": result.get("experiment_time_seconds", 0),
            }
        )

    return pd.DataFrame(data)


def print_results_table(
    results: Dict[str, Dict[str, Any]], metric_name: str = "f1_macro"
) -> None:
    """
    Print a formatted table of results.

    Parameters
    ----------
    results : dict
        Dictionary of experiment results
    metric_name : str
        Metric to display
    """
    tasks = ["cell_line", "drug", "moa_broad", "moa_fine"]

    print(f"\n{'='*80}")
    print(
        f"{'Strategy':<25} | {'Cell Line':>10} | {'Drug':>10} | {'MOA Broad':>10} | {'MOA Fine':>10}"
    )
    print(f"{'-'*80}")

    for strategy, result in results.items():
        if result.get("status") != "success":
            continue

        final_metrics = result.get("final_metrics", {})
        values = []

        for task in tasks:
            metric_key = f"{task}_{metric_name}"
            value = final_metrics.get(metric_key, 0.0)
            values.append(f"{value:.4f}")

        name = STRATEGY_DISPLAY_NAMES.get(strategy, strategy).replace("\n", " ")
        print(
            f"{name:<25} | {values[0]:>10} | {values[1]:>10} | {values[2]:>10} | {values[3]:>10}"
        )

    print(f"{'='*80}\n")
