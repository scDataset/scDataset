"""
Plotting utilities for experiment results.

This module provides functions for creating plots comparing different
data loading strategies across classification tasks.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def plot_comparison(
    df: pd.DataFrame,
    tasks: Optional[List[str]] = None,
    ylim_dict: Optional[Dict[str, Tuple[float, float]]] = None,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    metric_column: str = "Macro F1-score",
    error_column: Optional[str] = "Error",
    palette: Optional[str] = None,
):
    """
    Create a 2x2 comparison plot for all classification tasks.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns: Task, Method, {metric_column}, {error_column}
    tasks : list, optional
        List of task names in display order. Uses default order if None.
    ylim_dict : dict, optional
        Dictionary mapping task names to (ymin, ymax) tuples
    figsize : tuple
        Figure size (width, height)
    title : str, optional
        Overall figure title
    metric_column : str
        Name of the column containing metric values
    error_column : str, optional
        Name of the column containing error values for error bars
    palette : str, optional
        Seaborn color palette name

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if tasks is None:
        tasks = ["Cell line", "Drug", "MOA (broad)", "MOA (fine)"]

    # Default y-limits per task
    default_ylim = {
        "Cell line": (0.0, 1.0),
        "Drug": (0.0, 0.1),
        "MOA (broad)": (0.0, 0.3),
        "MOA (fine)": (0.0, 0.15),
    }

    if ylim_dict is None:
        ylim_dict = default_ylim

    # Set style
    sns.set(style="whitegrid", context="talk")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    # Get method order
    methods = df["Method"].unique()

    # Color palette
    if palette is None:
        colors = sns.color_palette("tab10", len(methods))
    else:
        colors = sns.color_palette(palette, len(methods))

    for idx, task in enumerate(tasks):
        ax = axes[idx]
        task_data = df[df["Task"] == task]

        if task_data.empty:
            ax.set_title(task)
            ax.set_ylabel(metric_column)
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Get values in consistent order
        values = []
        errors = []
        labels = []

        for method in methods:
            method_data = task_data[task_data["Method"] == method]
            if not method_data.empty:
                values.append(method_data[metric_column].values[0])
                if error_column and error_column in method_data.columns:
                    errors.append(method_data[error_column].values[0])
                else:
                    errors.append(0)
                labels.append(method)

        x = np.arange(len(labels))

        # Create bars
        bars = ax.bar(
            x,
            values,
            yerr=errors if any(e > 0 for e in errors) else None,
            color=colors[: len(labels)],
            capsize=5,
            edgecolor="black",
            linewidth=0.5,
        )

        # Customize
        ax.set_title(task, fontweight="bold")
        ax.set_ylabel(metric_column)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)

        # Set y-limits
        if task in ylim_dict:
            ax.set_ylim(ylim_dict[task])

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    return fig, axes


def plot_training_curves(
    results: Dict[str, Dict[str, Any]],
    task: str = "cell_line",
    metric: str = "f1_macro",
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot training curves for all strategies.

    Parameters
    ----------
    results : dict
        Dictionary of experiment results
    task : str
        Task to plot
    metric : str
        Metric to plot
    figsize : tuple
        Figure size

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from . import STRATEGY_DISPLAY_NAMES, TASK_DISPLAY_NAMES

    sns.set(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=figsize)

    for strategy, result in results.items():
        if result.get("status") != "success":
            continue

        history = result.get("history", {})
        metric_key = f"val_{task}_{metric}"

        if metric_key in history:
            values = history[metric_key]
            epochs = list(range(1, len(values) + 1))

            label = STRATEGY_DISPLAY_NAMES.get(strategy, strategy).replace("\n", " ")
            ax.plot(epochs, values, label=label, marker="o", markersize=4)

    task_name = TASK_DISPLAY_NAMES.get(task, task)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{task_name} - Macro F1-score")
    ax.set_title(f"Training Curves: {task_name}")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()

    return fig, ax


def plot_time_comparison(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
):
    """
    Plot training time comparison across strategies.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns: Method, Time (s)
    figsize : tuple
        Figure size

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    colors = sns.color_palette("tab10", len(df))

    bars = ax.bar(x, df["Time (s)"], color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Method")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Training Time Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Method"], rotation=45, ha="right", fontsize=9)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    return fig, ax
