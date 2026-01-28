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


def load_multiple_run_results(
    run_dirs: List[str], strategies: Optional[List[str]] = None
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Load experiment results from multiple run directories.

    Use this function when you have results from multiple experiment runs
    (e.g., different random seeds) and want to compute error bars.

    Parameters
    ----------
    run_dirs : list of str
        List of directories containing experiment results from different runs.
        Each directory should have the same structure as expected by
        load_experiment_results().
    strategies : list, optional
        List of strategy names to load. If None, loads all available.

    Returns
    -------
    list of dict
        List of results dictionaries, one per run.

    Examples
    --------
    >>> run_dirs = [
    ...     "results/run_seed42",
    ...     "results/run_seed43",
    ...     "results/run_seed44",
    ... ]
    >>> results_list = load_multiple_run_results(run_dirs)
    >>> df = create_comparison_dataframe_from_multiple_runs(results_list)
    """
    results_list = []
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        if run_path.exists():
            results = load_experiment_results(run_dir, strategies)
            if results:
                results_list.append(results)
    return results_list


def create_comparison_dataframe_from_multiple_runs(
    results_list: List[Dict[str, Dict[str, Any]]], metric_name: str = "f1_macro"
) -> pd.DataFrame:
    """
    Create a comparison DataFrame from multiple runs with automatic error bars.

    This function computes mean and standard deviation across runs for each
    strategy/task combination, providing automatic error bars for plotting.

    Parameters
    ----------
    results_list : list of dict
        List of experiment results dictionaries from multiple runs.
        Use load_multiple_run_results() to create this list.
    metric_name : str
        Metric to compare ('accuracy', 'f1_macro', 'f1_weighted')

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: Task, Method, Macro F1-score, Error
        where Error is the standard deviation across runs.

    Examples
    --------
    >>> results_list = load_multiple_run_results(["run1", "run2", "run3"])
    >>> df = create_comparison_dataframe_from_multiple_runs(results_list)
    >>> plot_comparison(df)  # Will automatically show error bars
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

    # Collect all values per (strategy, task) across runs
    from collections import defaultdict

    import numpy as np

    values_by_key = defaultdict(list)
    display_names = {}  # Cache display names

    for results in results_list:
        for strategy in strategies:
            if strategy not in results:
                continue

            result = results[strategy]
            if result.get("status") != "success":
                continue

            final_metrics = result.get("final_metrics", {})
            display_names[strategy] = get_strategy_display_name(strategy, result)

            for task in tasks:
                metric_key = f"{task}_{metric_name}"
                value = final_metrics.get(metric_key, 0.0)
                key = (strategy, task)
                values_by_key[key].append(value)

    # Compute mean and std for each (strategy, task)
    data = []
    for (strategy, task), values in values_by_key.items():
        values_arr = np.array(values)
        mean_val = np.mean(values_arr)
        std_val = np.std(values_arr, ddof=1) if len(values_arr) > 1 else 0.0

        data.append(
            {
                "Task": TASK_DISPLAY_NAMES[task],
                "Method": display_names.get(strategy, strategy),
                "Macro F1-score": mean_val,
                "Error": std_val,
                "N_runs": len(values_arr),
            }
        )

    return pd.DataFrame(data)


def create_comparison_dataframe(
    results: Dict[str, Dict[str, Any]], metric_name: str = "f1_macro"
) -> pd.DataFrame:
    """
    Create a comparison DataFrame for plotting.

    For multiple runs with error bars, use create_comparison_dataframe_from_multiple_runs()
    instead.

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
                    "Error": 0.0,  # For single run; use create_comparison_dataframe_from_multiple_runs for error bars
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


def create_comparison_dataframe_from_manual_runs(
    manual_runs: List[Dict[str, Dict[str, float]]],
    method_names: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Create a comparison DataFrame from manually entered multi-run data with automatic error bars.

    This function allows you to manually enter results from multiple experiment runs
    and automatically computes mean and standard deviation for error bars.

    Parameters
    ----------
    manual_runs : list of dict
        List of dictionaries, one per run. Each dictionary maps strategy names to
        their task metrics. The metrics should be a dict with keys:
        'cell_line', 'drug', 'moa_broad', 'moa_fine'.

        Example:
        [
            {  # Run 1
                'streaming': {'cell_line': 0.935, 'drug': 0.002, 'moa_broad': 0.204, 'moa_fine': 0.055},
                'block_shuffling': {'cell_line': 0.937, 'drug': 0.031, 'moa_broad': 0.270, 'moa_fine': 0.103},
            },
            {  # Run 2
                'streaming': {'cell_line': 0.936, 'drug': 0.003, 'moa_broad': 0.194, 'moa_fine': 0.054},
                'block_shuffling': {'cell_line': 0.938, 'drug': 0.030, 'moa_broad': 0.268, 'moa_fine': 0.104},
            },
        ]

    method_names : dict, optional
        Dictionary mapping strategy keys to display names.
        If None, uses STRATEGY_DISPLAY_NAMES.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: Task, Method, Macro F1-score, Error, N_runs
        where Error is the standard deviation across runs.

    Examples
    --------
    >>> manual_runs = [
    ...     {'streaming': {'cell_line': 0.935, 'drug': 0.002, 'moa_broad': 0.204, 'moa_fine': 0.055}},
    ...     {'streaming': {'cell_line': 0.936, 'drug': 0.003, 'moa_broad': 0.194, 'moa_fine': 0.054}},
    ... ]
    >>> df = create_comparison_dataframe_from_manual_runs(manual_runs)
    >>> plot_comparison(df)  # Will show error bars!
    """
    from collections import defaultdict

    import numpy as np

    if method_names is None:
        method_names = STRATEGY_DISPLAY_NAMES

    tasks = ["cell_line", "drug", "moa_broad", "moa_fine"]

    # Collect all values per (strategy, task) across runs
    values_by_key = defaultdict(list)

    for run_data in manual_runs:
        for strategy, metrics in run_data.items():
            for task in tasks:
                if task in metrics:
                    key = (strategy, task)
                    values_by_key[key].append(metrics[task])

    # Compute mean and std for each (strategy, task)
    data = []
    for (strategy, task), values in values_by_key.items():
        values_arr = np.array(values)
        mean_val = np.mean(values_arr)
        std_val = np.std(values_arr, ddof=1) if len(values_arr) > 1 else 0.0

        display_name = method_names.get(strategy, strategy)

        data.append(
            {
                "Task": TASK_DISPLAY_NAMES[task],
                "Method": display_name,
                "Macro F1-score": mean_val,
                "Error": std_val,
                "N_runs": len(values_arr),
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
