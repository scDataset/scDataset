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

# Strategy display names for plots (base templates)
# These will be dynamically updated with actual values from results.json
STRATEGY_DISPLAY_NAMES = {
    "streaming": "Streaming",
    "streaming_buffer": "Streaming (buffer)",
    "block_shuffling": "Block Shuffling",  # Will be updated dynamically
    "random_sampling": "Random Sampling",
    "block_weighted": "Block Weighted",  # Will be updated dynamically
    "true_weighted": "True Weighted",
}


def get_strategy_display_name(
    strategy: str, result: Optional[Dict[str, Any]] = None
) -> str:
    """
    Get display name for a strategy, dynamically using config values if available.

    Parameters
    ----------
    strategy : str
        Strategy key name
    result : dict, optional
        Result dictionary containing experiment_config

    Returns
    -------
    str
        Human-readable display name with actual block_size/fetch_factor values
    """
    # Base names
    base_names = {
        "streaming": "Streaming",
        "streaming_buffer": "Streaming (buffer)",
        "random_sampling": "Random Sampling",
        "true_weighted": "True Weighted",
        "block_shuffling": "Block Shuffling",
        "block_weighted": "Block Weighted",
    }

    base_name = base_names.get(strategy, strategy)

    # If no result or no config, return base name
    if result is None:
        return base_name

    config = result.get("experiment_config", {})
    block_size = config.get("block_size")
    fetch_factor = config.get("fetch_factor")

    # For block_shuffling, show "Block size = X\nFetch factor = Y"
    if strategy == "block_shuffling":
        lines = []
        if block_size is not None:
            lines.append(f"Block size = {block_size}")
        if fetch_factor is not None:
            lines.append(f"Fetch factor = {fetch_factor}")
        if lines:
            return "\n".join(lines)
        return base_name

    # For block_weighted, show similar format
    if strategy == "block_weighted":
        parts = []
        if block_size is not None:
            parts.append(f"block_size = {block_size}")
        if parts:
            return f"Block Weighted\n({', '.join(parts)})"
        return base_name

    # For true_weighted, show block_size=1
    if strategy == "true_weighted":
        if block_size is not None:
            return f"True Weighted\n(block_size = {block_size})"
        return base_name

    # For random_sampling, just use base name
    if strategy == "random_sampling":
        return base_name

    return base_name


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
                # Prefer results.json (editable) over results.pkl
                json_file = subdir / "results.json"
                if json_file.exists():
                    with open(json_file) as f:
                        results[strategy] = json.load(f)
                    break

                # Fallback to results.pkl
                pkl_file = subdir / "results.pkl"
                if pkl_file.exists():
                    with open(pkl_file, "rb") as f:
                        results[strategy] = pickle.load(f)
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
                    "Method": get_strategy_display_name(strategy, result),
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
                    "Method": get_strategy_display_name(strategy, result),
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
                "Method": get_strategy_display_name(strategy, result),
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

        name = get_strategy_display_name(strategy, result).replace("\n", " ")
        print(
            f"{name:<25} | {values[0]:>10} | {values[1]:>10} | {values[2]:>10} | {values[3]:>10}"
        )

    print(f"{'='*80}\n")
