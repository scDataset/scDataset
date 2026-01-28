"""
Plotting utilities for experiment results.

This module provides functions for creating plots comparing different
data loading strategies across classification tasks.
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _setup_publication_style():
    """Set up clean, professional fonts and plot aesthetics."""
    plt.rcParams.update(
        {
            # Font settings - Arial/Helvetica are standard for publications
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            # Better text rendering
            "text.usetex": False,
            "mathtext.fontset": "dejavusans",
            # Clean axes
            "axes.linewidth": 1.2,
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            # Tick parameters
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#cccccc",
            # Figure
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _compute_smart_ylim(
    values: np.ndarray, padding_ratio: float = 0.15
) -> Tuple[float, float]:
    """
    Compute smart y-axis limits that zoom in on the data range.

    Parameters
    ----------
    values : array-like
        Data values to compute limits for.
    padding_ratio : float, default=0.15
        Ratio of data range to add as padding above and below.

    Returns
    -------
    tuple
        (ymin, ymax) limits.
    """
    values = np.array(values)
    values = values[~np.isnan(values)]  # Remove NaN values

    if len(values) == 0:
        return (0, 1)

    v_min, v_max = values.min(), values.max()
    v_range = v_max - v_min

    # If all values are nearly the same, use a default range
    if v_range < 1e-6:
        return (v_min - 0.1 * abs(v_min) - 0.01, v_max + 0.1 * abs(v_max) + 0.01)

    padding = v_range * padding_ratio
    ymin = v_min - padding
    ymax = v_max + padding

    # Don't go below 0 if all values are positive
    if v_min >= 0 and ymin < 0:
        ymin = 0

    # Don't go above 1 for ratio metrics (like F1 score) if max is close to 1
    if v_max <= 1.0 and ymax > 1.0:
        ymax = 1.0

    return (ymin, ymax)


def plot_comparison(
    df: pd.DataFrame,
    tasks: Optional[List[str]] = None,
    ylim_dict: Optional[Dict[str, Tuple[float, float]]] = None,
    auto_ylim: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    title: Optional[str] = None,
    metric_column: str = "Macro F1-score",
    error_column: Optional[str] = "Error",
    palette: Optional[str] = None,
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    fontsize_ticks: int = 10,
    hspace: float = 0.30,
    top_margin: float = 0.90,
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
        Dictionary mapping task names to (ymin, ymax) tuples.
        If None and auto_ylim=True, limits are computed automatically.
    auto_ylim : bool, default=True
        If True and ylim_dict is None or task not in ylim_dict, compute
        smart y-axis limits that zoom in on the data range.
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
    fontsize_title : int, default=14
        Font size for titles.
    fontsize_labels : int, default=12
        Font size for axis labels.
    fontsize_ticks : int, default=10
        Font size for tick labels.
    hspace : float, default=0.40
        Vertical spacing between subplot rows.
    top_margin : float, default=0.90
        Top margin for suptitle (0.90 = less space between title and plots).

    Returns
    -------
    tuple
        (fig, axes) matplotlib figure and axes
    """
    # Apply publication style
    _setup_publication_style()

    if tasks is None:
        tasks = ["Cell line", "Drug", "MOA (broad)", "MOA (fine)"]

    if ylim_dict is None:
        ylim_dict = {}

    # Set style
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

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
            ax.set_title(task, fontsize=fontsize_title, fontweight="bold")
            ax.set_ylabel(metric_column, fontsize=fontsize_labels)
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

        # Create bars with white edge for cleaner look
        bars = ax.bar(
            x,
            values,
            yerr=errors if any(e > 0 for e in errors) else None,
            color=colors[: len(labels)],
            capsize=5,
            edgecolor="white",
            linewidth=1.0,
        )

        # Customize
        ax.set_title(task, fontsize=fontsize_title, fontweight="bold")
        ax.set_ylabel(metric_column, fontsize=fontsize_labels, fontweight="medium")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=fontsize_ticks)
        ax.tick_params(axis="y", labelsize=fontsize_ticks)

        # Set y-limits: use provided, or compute smart limits, or default
        if task in ylim_dict:
            ax.set_ylim(ylim_dict[task])
        elif auto_ylim:
            # Compute smart y-limits based on data
            all_values = np.array(values)
            if len(errors) > 0 and any(e > 0 for e in errors):
                # Account for error bars
                all_values = np.concatenate(
                    [
                        all_values,
                        np.array(values) + np.array(errors),
                        np.array(values) - np.array(errors),
                    ]
                )
            ylim = _compute_smart_ylim(all_values)
            ax.set_ylim(ylim)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=fontsize_ticks,
            )

        # Add subtle grid
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=fontsize_title + 2, fontweight="bold", y=0.98)

    # Adjust spacing: reduce space between title and plots, increase between rows
    plt.tight_layout()
    plt.subplots_adjust(top=top_margin, hspace=hspace)

    return fig, axes


def plot_training_curves(
    results: Dict[str, Dict[str, Any]],
    task: str = "cell_line",
    metric: str = "f1_macro",
    figsize: Tuple[int, int] = (10, 6),
    auto_ylim: bool = True,
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    fontsize_ticks: int = 10,
    fontsize_legend: int = 9,
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
    auto_ylim : bool, default=True
        If True, compute smart y-axis limits that zoom in on the data range.
    fontsize_title : int, default=14
        Font size for title.
    fontsize_labels : int, default=12
        Font size for axis labels.
    fontsize_ticks : int, default=10
        Font size for tick labels.
    fontsize_legend : int, default=9
        Font size for legend.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axis
    """
    from . import STRATEGY_DISPLAY_NAMES, TASK_DISPLAY_NAMES

    # Apply publication style
    _setup_publication_style()
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

    fig, ax = plt.subplots(figsize=figsize)

    all_values = []
    for strategy, result in results.items():
        if result.get("status") != "success":
            continue

        history = result.get("history", {})
        metric_key = f"val_{task}_{metric}"

        if metric_key in history:
            values = history[metric_key]
            all_values.extend(values)
            epochs = list(range(1, len(values) + 1))

            label = STRATEGY_DISPLAY_NAMES.get(strategy, strategy).replace("\n", " ")
            ax.plot(epochs, values, label=label, marker="o", markersize=4, linewidth=2)

    task_name = TASK_DISPLAY_NAMES.get(task, task)
    ax.set_xlabel("Epoch", fontsize=fontsize_labels, fontweight="medium")
    ax.set_ylabel(
        f"{task_name} - Macro F1-score", fontsize=fontsize_labels, fontweight="medium"
    )
    ax.set_title(
        f"Training Curves: {task_name}", fontsize=fontsize_title, fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=fontsize_ticks)
    ax.legend(loc="best", fontsize=fontsize_legend)

    # Smart y-limits
    if auto_ylim and len(all_values) > 0:
        ylim = _compute_smart_ylim(np.array(all_values))
        ax.set_ylim(ylim)

    # Add subtle grid and clean spines
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig, ax


def plot_time_comparison(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6),
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    fontsize_ticks: int = 10,
):
    """
    Plot training time comparison across strategies.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with columns: Method, Time (s)
    figsize : tuple
        Figure size
    fontsize_title : int, default=14
        Font size for title.
    fontsize_labels : int, default=12
        Font size for axis labels.
    fontsize_ticks : int, default=10
        Font size for tick labels.

    Returns
    -------
    tuple
        (fig, ax) matplotlib figure and axis
    """
    # Apply publication style
    _setup_publication_style()
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    colors = sns.color_palette("tab10", len(df))

    bars = ax.bar(x, df["Time (s)"], color=colors, edgecolor="white", linewidth=1.0)

    ax.set_xlabel("Method", fontsize=fontsize_labels, fontweight="medium")
    ax.set_ylabel("Time (seconds)", fontsize=fontsize_labels, fontweight="medium")
    ax.set_title("Training Time Comparison", fontsize=fontsize_title, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Method"], rotation=45, ha="right", fontsize=fontsize_ticks)
    ax.tick_params(axis="y", labelsize=fontsize_ticks)

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
            fontsize=fontsize_ticks - 1,
        )

    # Add subtle grid and clean spines
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig, ax
