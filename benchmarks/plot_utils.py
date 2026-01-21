"""
Plotting utilities for scDataset benchmark experiments.

This module provides functions for creating plots
to visualize the performance of scDataset vs baseline loaders.

Plot Types
----------
plot_throughput : Create throughput plot (samples/sec vs block size)
plot_batch_entropy : Create batch entropy plot showing shuffling quality
plot_block_size_by_fetch_factor : Comprehensive plot with fetch factor as hue

Usage
-----
>>> from plot_utils import plot_throughput, plot_batch_entropy
>>> plot_throughput('experiments/block_size_fetch_factor_eval_anncollection_random.csv',
...                 title='scDataset Throughput with AnnData',
...                 save_path='figures/throughput_anndata.pdf')
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


def _format_as_int(x, pos):
    """Format tick labels as integers when possible."""
    if x == int(x):
        return f"{int(x)}"
    else:
        return f"{x}"


def plot_throughput(
    csv_path: str,
    title: str = "scDataset Throughput",
    y_column: str = "samples_per_second",
    y_label: str = "Samples Per Second",
    baseline_label: str = "Random Sampling",
    fig_size: tuple = (10, 6),
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    y_log_scale: bool = True,
    x_log_scale: bool = True,
    color_map: str = "plasma",
    save_path: str = None,
    show_plot: bool = True,
):
    """
    Create a throughput plot showing samples/sec vs block size.

    Generates a plot with separate lines for each fetch_factor value,
    with a horizontal baseline line for the traditional data loader.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with benchmark results.
    title : str, default="scDataset Throughput"
        Title for the plot.
    y_column : str, default="samples_per_second"
        Column to plot on Y axis.
    y_label : str, default="Samples Per Second"
        Label for Y axis.
    baseline_label : str, default="Random Sampling"
        Label for baseline horizontal line in legend.
    fig_size : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    fontsize_title : int, default=14
        Font size for title.
    fontsize_labels : int, default=12
        Font size for axis labels.
    y_log_scale : bool, default=True
        Use log scale for Y axis.
    x_log_scale : bool, default=True
        Use log scale for X axis.
    color_map : str, default="plasma"
        Matplotlib colormap for fetch factor lines.
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_throughput(
    ...     'experiments/block_size_fetch_factor_eval_anncollection_random.csv',
    ...     title='scDataset Throughput with AnnData',
    ...     save_path='figures/throughput_anndata.pdf'
    ... )
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Detect collection type for appropriate baseline naming
    collection_type = df["collection_type"].iloc[0] if "collection_type" in df.columns else "anncollection"

    # Identify baseline loader based on collection type
    if collection_type == "anncollection":
        baseline_loaders = ["AnnLoader"]
    elif collection_type == "huggingface":
        baseline_loaders = ["HuggingFace", "PyTorch DataLoader"]
    elif collection_type == "bionemo":
        baseline_loaders = ["BioNeMo", "PyTorch DataLoader"]
    else:
        baseline_loaders = []

    # Filter for scDataset results (random mode)
    df_scdataset = df[(df["mode"] == "random") & (df["loader"] == "scDataset")].copy()

    # Filter for baseline results
    df_baseline = df[
        (df["mode"] == "random") & (df["loader"].isin(baseline_loaders))
    ].copy()

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # Get unique fetch factors and create a color map
    fetch_factors = sorted(df_scdataset["fetch_factor"].dropna().unique())
    cmap = plt.get_cmap(color_map)
    colors = [cmap(i / max(len(fetch_factors) - 1, 1)) for i in range(len(fetch_factors))]

    # Plot a line for each fetch factor
    for i, ff in enumerate(fetch_factors):
        df_ff = df_scdataset[df_scdataset["fetch_factor"] == ff]
        df_ff = df_ff.sort_values("block_size")

        ff_label = int(ff) if float(ff).is_integer() else ff
        ax.plot(
            df_ff["block_size"],
            df_ff[y_column],
            marker="o",
            label=f"fetch_factor = {ff_label}",
            linewidth=2,
            markersize=8,
            color=colors[i],
        )

    # Add baseline as horizontal line
    if not df_baseline.empty:
        baseline_value = df_baseline[y_column].values[0]
        ax.axhline(
            y=baseline_value,
            color="black",
            linestyle="--",
            linewidth=2,
            label=baseline_label,
        )

    # Format axes
    formatter = FuncFormatter(_format_as_int)

    if x_log_scale:
        ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(formatter)

    if y_log_scale:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.set_ylim(bottom=0)

    # Add labels and title
    ax.set_xlabel("Block Size", fontsize=fontsize_labels)
    ax.set_ylabel(y_label, fontsize=fontsize_labels)
    ax.set_title(title, fontsize=fontsize_title)

    # Add legend
    ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize="medium")

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_batch_entropy(
    csv_path: str,
    title: str = "scDataset Batch Entropy",
    y_column: str = "avg_batch_entropy",
    y_label: str = "Batch Entropy (bits)",
    error_column: str = "std_batch_entropy",
    baseline_label: str = "Random Sampling",
    fig_size: tuple = (10, 6),
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    x_log_scale: bool = True,
    color_map: str = "plasma",
    save_path: str = None,
    show_plot: bool = True,
):
    """
    Create a batch entropy plot showing shuffling quality.

    Displays batch entropy vs block size with separate lines for each
    fetch_factor. Higher entropy indicates better random mixing of samples
    from different sources.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with benchmark results.
    title : str, default="scDataset Batch Entropy"
        Title for the plot.
    y_column : str, default="avg_batch_entropy"
        Column to plot on Y axis.
    y_label : str, default="Batch Entropy (bits)"
        Label for Y axis.
    error_column : str, default="std_batch_entropy"
        Column for error bands.
    baseline_label : str, default="Random Sampling"
        Label for baseline horizontal line in legend.
    fig_size : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    fontsize_title : int, default=14
        Font size for title.
    fontsize_labels : int, default=12
        Font size for axis labels.
    x_log_scale : bool, default=True
        Use log scale for X axis.
    color_map : str, default="plasma"
        Matplotlib colormap for fetch factor lines.
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_batch_entropy(
    ...     'experiments/block_size_fetch_factor_eval_anncollection_random.csv',
    ...     title='scDataset Batch Entropy',
    ...     save_path='figures/entropy_anndata.pdf'
    ... )
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for scDataset results (random mode)
    df_scdataset = df[(df["mode"] == "random") & (df["loader"] == "scDataset")].copy()

    # Filter for baseline results (AnnLoader)
    df_baseline = df[(df["mode"] == "random") & (df["loader"] == "AnnLoader")].copy()

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # Get unique fetch factors and create a color map
    fetch_factors = sorted(df_scdataset["fetch_factor"].dropna().unique())
    cmap = plt.get_cmap(color_map)
    colors = [cmap(i / max(len(fetch_factors) - 1, 1)) for i in range(len(fetch_factors))]

    # Plot a line for each fetch factor with error bands
    for i, ff in enumerate(fetch_factors):
        df_ff = df_scdataset[df_scdataset["fetch_factor"] == ff]
        df_ff = df_ff.sort_values("block_size")

        ff_label = int(ff) if float(ff).is_integer() else ff
        ax.plot(
            df_ff["block_size"],
            df_ff[y_column],
            marker="o",
            label=f"fetch_factor = {ff_label}",
            linewidth=2,
            markersize=8,
            color=colors[i],
        )

        # Add error bands
        if error_column in df_ff.columns:
            ax.fill_between(
                df_ff["block_size"],
                df_ff[y_column] - df_ff[error_column],
                df_ff[y_column] + df_ff[error_column],
                alpha=0.15,
                color=colors[i],
            )

    # Add baseline as horizontal line
    if not df_baseline.empty:
        baseline_value = df_baseline[y_column].values[0]
        ax.axhline(
            y=baseline_value,
            color="black",
            linestyle="--",
            linewidth=2,
            label=baseline_label,
        )
        # Add error band for baseline
        if error_column in df_baseline.columns:
            baseline_error = df_baseline[error_column].values[0]
            if baseline_error > 0:
                ax.axhspan(
                    baseline_value - baseline_error,
                    baseline_value + baseline_error,
                    alpha=0.15,
                    color="black",
                )

    # Format axes
    formatter = FuncFormatter(_format_as_int)

    if x_log_scale:
        ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(formatter)

    # Add labels and title
    ax.set_xlabel("Block Size", fontsize=fontsize_labels)
    ax.set_ylabel(y_label, fontsize=fontsize_labels)
    ax.set_title(title, fontsize=fontsize_title)

    # Add legend
    ax.legend(loc="best", fontsize="medium")

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_block_size_by_fetch_factor(
    csv_path: str,
    y_column: str,
    y_label: str,
    plot_title: str,
    include_error_bars: bool = False,
    error_column: str = None,
    fig_size: tuple = (10, 6),
    fontsize_title: int = 14,
    fontsize_labels: int = 12,
    y_log_scale: bool = False,
    x_log_scale: bool = True,
    color_map: str = "plasma",
    baseline_label: str = "Random Sampling",
    save_path: str = None,
    show_plot: bool = True,
):
    """
    Create a comprehensive plot showing block size vs metric with fetch factor as hue.

    This is the main plotting function that creates publication-quality figures
    for the scDataset paper. It shows scDataset performance with different
    block_size and fetch_factor combinations compared to a baseline.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with benchmark results.
    y_column : str
        Column to plot on Y axis (e.g., 'samples_per_second', 'avg_batch_entropy').
    y_label : str
        Label for Y axis.
    plot_title : str
        Title for the plot.
    include_error_bars : bool, default=False
        Whether to include error bands.
    error_column : str, optional
        Column for error values.
    fig_size : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    fontsize_title : int, default=14
        Font size for title.
    fontsize_labels : int, default=12
        Font size for axis labels.
    y_log_scale : bool, default=False
        Use log scale for Y axis.
    x_log_scale : bool, default=True
        Use log scale for X axis.
    color_map : str, default="plasma"
        Matplotlib colormap for fetch factor lines.
    baseline_label : str, default="Random Sampling"
        Label for baseline horizontal line in legend.
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> # Throughput plot
    >>> fig = plot_block_size_by_fetch_factor(
    ...     csv_path='experiments/block_size_fetch_factor_eval_anncollection_random.csv',
    ...     y_column='samples_per_second',
    ...     y_label='Samples Per Second',
    ...     plot_title='scDataset Throughput with AnnData',
    ...     y_log_scale=True,
    ...     save_path='figures/throughput_anndata.pdf'
    ... )

    >>> # Entropy plot
    >>> fig = plot_block_size_by_fetch_factor(
    ...     csv_path='experiments/block_size_fetch_factor_eval_anncollection_random.csv',
    ...     y_column='avg_batch_entropy',
    ...     y_label='Batch Entropy (bits)',
    ...     plot_title='scDataset Batch Entropy',
    ...     include_error_bars=True,
    ...     error_column='std_batch_entropy',
    ...     save_path='figures/entropy_anndata.pdf'
    ... )
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Detect collection type for appropriate baseline filtering
    collection_type = df["collection_type"].iloc[0] if "collection_type" in df.columns else "anncollection"

    # Filter for scDataset results (random mode)
    df_filtered = df[(df["mode"] == "random") & (df["loader"] == "scDataset")].copy()

    # Identify and filter baseline
    if collection_type == "anncollection":
        df_baseline = df[(df["mode"] == "random") & (df["loader"] == "AnnLoader")]
    elif collection_type == "huggingface":
        df_baseline = df[(df["mode"] == "random") & (df["loader"].isin(["HuggingFace", "PyTorch DataLoader"]))]
    elif collection_type == "bionemo":
        df_baseline = df[(df["mode"] == "random") & (df["loader"].isin(["BioNeMo", "PyTorch DataLoader"]))]
    else:
        df_baseline = pd.DataFrame()

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    # Get unique fetch factors and create a color map
    fetch_factors = sorted(df_filtered["fetch_factor"].dropna().unique())
    cmap = plt.get_cmap(color_map)
    colors = [cmap(i / max(len(fetch_factors) - 1, 1)) for i in range(len(fetch_factors))]

    # Plot a line for each fetch factor with optional error bands
    for i, ff in enumerate(fetch_factors):
        df_ff = df_filtered[df_filtered["fetch_factor"] == ff]
        df_ff = df_ff.sort_values("block_size")

        ff_label = int(ff) if float(ff).is_integer() else ff
        ax.plot(
            df_ff["block_size"],
            df_ff[y_column],
            marker="o",
            label=f"fetch_factor = {ff_label}",
            linewidth=2,
            markersize=8,
            color=colors[i],
        )

        # Add error bands if requested
        if include_error_bars and error_column and error_column in df_ff.columns:
            ax.fill_between(
                df_ff["block_size"],
                df_ff[y_column] - df_ff[error_column],
                df_ff[y_column] + df_ff[error_column],
                alpha=0.15,
                color=colors[i],
            )

    # Add baseline as horizontal line
    if not df_baseline.empty:
        y_value = df_baseline[y_column].values[0]
        ax.axhline(
            y=y_value,
            color="black",
            linestyle="--",
            linewidth=2,
            label=baseline_label,
        )
        # Add error band for baseline if available
        if include_error_bars and error_column and error_column in df_baseline.columns:
            baseline_error = df_baseline[error_column].values[0]
            if baseline_error > 0:
                ax.axhspan(
                    y_value - baseline_error,
                    y_value + baseline_error,
                    alpha=0.15,
                    color="black",
                )

    # Format axes
    formatter = FuncFormatter(_format_as_int)

    if x_log_scale:
        ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(formatter)

    if y_log_scale:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(formatter)
    else:
        # For samples per second, start y-axis at 0
        if "per_second" in y_column:
            ax.set_ylim(bottom=0)

    # Add labels and title
    ax.set_xlabel("Block Size", fontsize=fontsize_labels)
    ax.set_ylabel(y_label, fontsize=fontsize_labels)
    ax.set_title(plot_title, fontsize=fontsize_title)

    # Position legend based on plot type
    if "per_second" in y_column:
        ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), fontsize="medium")
    else:
        ax.legend(loc="best", fontsize="medium")

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def generate_all_benchmark_plots(experiments_dir: str = "experiments", figures_dir: str = "figures"):
    """
    Generate all standard benchmark plots from experiment results.

    Creates throughput and batch entropy plots for all available
    data sources (AnnData, HuggingFace, BioNeMo).

    Parameters
    ----------
    experiments_dir : str, default="experiments"
        Directory containing CSV result files.
    figures_dir : str, default="figures"
        Directory to save generated figures.

    Examples
    --------
    >>> generate_all_benchmark_plots()
    """
    import os

    os.makedirs(figures_dir, exist_ok=True)

    # Define experiments to plot
    experiments = [
        {
            "csv": f"{experiments_dir}/block_size_fetch_factor_eval_anncollection_random.csv",
            "name": "AnnData",
            "has_entropy": True,
        },
        {
            "csv": f"{experiments_dir}/block_size_fetch_factor_eval_huggingface_random.csv",
            "name": "HuggingFace",
            "has_entropy": False,
        },
        {
            "csv": f"{experiments_dir}/block_size_fetch_factor_eval_bionemo_random.csv",
            "name": "BioNeMo",
            "has_entropy": False,
        },
    ]

    for exp in experiments:
        if not os.path.exists(exp["csv"]):
            print(f"Skipping {exp['name']}: {exp['csv']} not found")
            continue

        print(f"\nGenerating plots for {exp['name']}...")

        # Throughput plot
        plot_throughput(
            exp["csv"],
            title=f"scDataset Throughput with {exp['name']}",
            save_path=f"{figures_dir}/throughput_{exp['name'].lower()}.pdf",
            show_plot=False,
        )

        # Batch entropy plot (only for AnnData which has plate info)
        if exp["has_entropy"]:
            plot_batch_entropy(
                exp["csv"],
                title=f"scDataset Batch Entropy",
                save_path=f"{figures_dir}/batch_entropy_{exp['name'].lower()}.pdf",
                show_plot=False,
            )

    print(f"\nAll plots saved to {figures_dir}/")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--generate-all":
        generate_all_benchmark_plots()
    else:
        print("Usage: python plot_utils.py --generate-all")
        print("       Or import functions for custom plotting.")
