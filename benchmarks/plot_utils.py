"""
Plotting utilities for scDataset benchmark experiments.

This module provides functions for creating plots
to visualize the performance of scDataset vs baseline loaders.

Plot Types
----------
plot_throughput : Create throughput plot (samples/sec vs block size)
plot_batch_entropy : Create minibatch entropy plot showing shuffling quality
plot_block_size_by_fetch_factor : Comprehensive plot with fetch factor as hue
plot_tradeoff : Create throughput vs minibatch entropy tradeoff plot (with lines)
plot_tradeoff_scatter : Create scatter tradeoff plot (color=fetch_factor, size=block_size)
plot_streaming_comparison : Create bar plot comparing AnnLoader vs scDataset streaming

Usage
-----
>>> from plot_utils import plot_throughput, plot_batch_entropy, plot_tradeoff
>>> plot_throughput('experiments/block_size_fetch_factor_eval_anncollection_random.csv',
...                 title='scDataset Throughput with AnnData',
...                 save_path='figures/throughput_anndata.pdf')
"""

import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter


class PowerScale(mscale.ScaleBase):
    """
    A matplotlib scale that applies a power transformation.

    For sqrt scale, use power=0.5. For square scale, use power=2.
    The transformation is: y = x^power
    """

    name = 'power'

    def __init__(self, axis, *, power=0.5, **kwargs):
        """
        Initialize the power scale.

        Parameters
        ----------
        axis : matplotlib axis
            The axis to apply the scale to.
        power : float, default=0.5
            The power exponent. Use 0.5 for sqrt scale, 2 for square scale.
        """
        super().__init__(axis, **kwargs)
        self.power = power

    def get_transform(self):
        """Return the transform for this scale."""
        return self.PowerTransform(self.power)

    def set_default_locators_and_formatters(self, axis):
        """Set default locators and formatters for the axis."""
        axis.set_major_locator(MaxNLocator(nbins=8))
        axis.set_major_formatter(ScalarFormatter())

    class PowerTransform(mtransforms.Transform):
        """Transform that applies x^power."""

        input_dims = output_dims = 1

        def __init__(self, power):
            """Initialize with the power exponent."""
            super().__init__()
            self.power = power

        def transform_non_affine(self, a):
            """Apply the forward transform: x^power."""
            # Add small epsilon to avoid issues with zero
            return np.power(np.maximum(a, 1e-10), self.power)

        def inverted(self):
            """Return the inverse transform."""
            return PowerScale.InversePowerTransform(self.power)

    class InversePowerTransform(mtransforms.Transform):
        """Inverse transform that applies x^(1/power)."""

        input_dims = output_dims = 1

        def __init__(self, power):
            """Initialize with the power exponent."""
            super().__init__()
            self.power = power

        def transform_non_affine(self, a):
            """Apply the inverse transform: x^(1/power)."""
            return np.power(np.maximum(a, 1e-10), 1.0 / self.power)

        def inverted(self):
            """Return the forward transform."""
            return PowerScale.PowerTransform(self.power)


# Register the power scale with matplotlib (only once)
try:
    mscale.register_scale(PowerScale)
except ValueError:
    # Already registered, ignore
    pass


def _format_as_int(x, pos):
    """Format tick labels as integers when possible."""
    if x == int(x):
        return f"{int(x)}"
    else:
        return f"{x}"


def _setup_publication_style():
    """Set up clean, professional fonts and plot aesthetics."""
    # Use a clean, professional font stack
    plt.rcParams.update({
        # Font settings - Arial/Helvetica are standard for publications
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans', 'sans-serif'],
        'font.size': 12,
        # Better text rendering
        'text.usetex': False,
        'mathtext.fontset': 'dejavusans',
        # Clean axes
        'axes.linewidth': 1.2,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        # Tick parameters
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': '#cccccc',
        # Figure
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def _get_log_scale_colors(values, cmap_name: str = "plasma", cmap_range: tuple = (0.0, 0.95)):
    """
    Get colors for values using log-scale sampling from colormap.

    Since fetch_factor values are powers of 2, this samples colors
    in log-space for better visual distribution.

    Parameters
    ----------
    values : array-like
        Values to map to colors (e.g., fetch factors like [1, 4, 16, 64, 256]).
    cmap_name : str
        Name of the matplotlib colormap.
    cmap_range : tuple, default=(0.0, 0.95)
        Range of the colormap to use (0.0 to 1.0). The default avoids the
        very bright yellow end of plasma colormap for better visibility.

    Returns
    -------
    list
        List of RGBA colors for each value.
    """
    cmap = plt.get_cmap(cmap_name)
    values = np.array(sorted(values))

    if len(values) <= 1:
        mid_point = (cmap_range[0] + cmap_range[1]) / 2
        return [cmap(mid_point) for _ in values]

    # Use log scale for sampling colors
    log_values = np.log2(np.maximum(values, 1))  # Avoid log(0)
    log_min, log_max = log_values.min(), log_values.max()

    if log_max == log_min:
        mid_point = (cmap_range[0] + cmap_range[1]) / 2
        return [cmap(mid_point) for _ in values]

    # Normalize to [0, 1] in log space
    normalized = (log_values - log_min) / (log_max - log_min)

    # Scale to the specified colormap range
    cmap_min, cmap_max = cmap_range
    scaled = cmap_min + normalized * (cmap_max - cmap_min)

    return [cmap(s) for s in scaled]


def plot_throughput(
    csv_path: str,
    title: str = "scDataset Throughput",
    y_column: str = "samples_per_second",
    y_label: str = "Samples Per Second",
    baseline_label: str = None,  # Auto-detected based on collection type
    fig_size: tuple = (10, 6),
    fontsize_title: int = 18,
    fontsize_labels: int = 16,
    fontsize_ticks: int = 14,
    fontsize_legend: int = 10,
    y_log_scale: bool = True,
    x_log_scale: bool = True,
    color_map: str = "plasma",
    save_path: str = None,
    show_plot: bool = True,
    include_streaming: bool = True,
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
    baseline_label : str, optional
        Label for baseline horizontal line. Auto-detected based on collection type:
        - AnnData: "AnnLoader"
        - HuggingFace: "HuggingFace"
        - BioNeMo: "BioNeMo-SCDL"
    fig_size : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    fontsize_title : int, default=18
        Font size for title.
    fontsize_labels : int, default=16
        Font size for axis labels.
    fontsize_ticks : int, default=14
        Font size for tick labels.
    fontsize_legend : int, default=13
        Font size for legend.
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
    include_streaming : bool, default=True
        Whether to include streaming baseline if present in data.

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
    # Apply Nature-style formatting
    _setup_publication_style()

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Detect collection type for appropriate baseline naming
    collection_type = df["collection_type"].iloc[0] if "collection_type" in df.columns else "anncollection"

    # Identify baseline loader and label based on collection type
    if collection_type == "anncollection":
        baseline_loaders = ["AnnLoader"]
        default_baseline_label = "AnnLoader"
    elif collection_type == "huggingface":
        baseline_loaders = ["HuggingFace", "PyTorch DataLoader"]
        default_baseline_label = "HuggingFace"
    elif collection_type == "bionemo":
        baseline_loaders = ["BioNeMo", "PyTorch DataLoader"]
        default_baseline_label = "BioNeMo-SCDL"
    else:
        baseline_loaders = []
        default_baseline_label = "Baseline"

    # Use provided label or auto-detected default
    if baseline_label is None:
        baseline_label = default_baseline_label

    # Filter for scDataset results (random mode)
    df_scdataset = df[(df["mode"] == "random") & (df["loader"] == "scDataset")].copy()

    # Filter for baseline results
    df_baseline = df[
        (df["mode"] == "random") & (df["loader"].isin(baseline_loaders))
    ].copy()

    # Filter for streaming baseline (if present and requested)
    df_streaming = pd.DataFrame()
    if include_streaming:
        df_streaming = df[
            (df["mode"] == "stream") & (df["loader"].isin(baseline_loaders))
        ].copy()

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Get unique fetch factors and create a color map with log-scale sampling
    fetch_factors = sorted(df_scdataset["fetch_factor"].dropna().unique())
    colors = _get_log_scale_colors(fetch_factors, color_map)

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
            color="#2E86AB",  # Professional blue
            linestyle="--",
            linewidth=2.5,
            label=baseline_label,
        )

    # Add streaming baseline as horizontal line (if present)
    if not df_streaming.empty:
        streaming_value = df_streaming[y_column].values[0]
        ax.axhline(
            y=streaming_value,
            color="#A23B72",  # Professional magenta
            linestyle=":",
            linewidth=2.5,
            label=f"{baseline_label} (streaming)",
        )

    # Format axes
    formatter = FuncFormatter(_format_as_int)

    if x_log_scale:
        ax.set_xscale("log", base=2)
    # Set fixed x-axis ticks
    x_ticks = [1, 4, 16, 64, 256, 1024]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(formatter)

    if y_log_scale:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.set_ylim(bottom=0)

    # Add labels and title with improved font sizes
    ax.set_xlabel("Block Size", fontsize=fontsize_labels, fontweight='medium')
    ax.set_ylabel(y_label, fontsize=fontsize_labels, fontweight='medium')
    ax.set_title(title, fontsize=fontsize_title, fontweight='bold', pad=15)

    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Add legend with improved styling
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        fontsize=fontsize_legend,
        framealpha=0.95,
        edgecolor='#cccccc',
    )
    legend.get_frame().set_linewidth(1.0)

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_batch_entropy(
    csv_path: str,
    title: str = "scDataset Minibatch Entropy",
    y_column: str = "avg_batch_entropy",
    y_label: str = "Minibatch Entropy (bits)",
    error_column: str = "std_batch_entropy",
    baseline_label: str = "Random Sampling",
    fig_size: tuple = (10, 6),
    fontsize_title: int = 18,
    fontsize_labels: int = 16,
    fontsize_ticks: int = 14,
    fontsize_legend: int = 10,
    x_log_scale: bool = True,
    color_map: str = "plasma",
    save_path: str = None,
    show_plot: bool = True,
    include_streaming: bool = True,
):
    """
    Create a minibatch entropy plot showing shuffling quality.

    Displays minibatch entropy vs block size with separate lines for each
    fetch_factor. Higher entropy indicates better random mixing of samples
    from different sources.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with benchmark results.
    title : str, default="scDataset Minibatch Entropy"
        Title for the plot.
    y_column : str, default="avg_batch_entropy"
        Column to plot on Y axis.
    y_label : str, default="Minibatch Entropy (bits)"
        Label for Y axis.
    error_column : str, default="std_batch_entropy"
        Column for error bands.
    baseline_label : str, default="Random Sampling"
        Label for baseline horizontal line in legend.
    fig_size : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    fontsize_title : int, default=18
        Font size for title.
    fontsize_labels : int, default=16
        Font size for axis labels.
    fontsize_ticks : int, default=14
        Font size for tick labels.
    fontsize_legend : int, default=13
        Font size for legend.
    x_log_scale : bool, default=True
        Use log scale for X axis.
    color_map : str, default="plasma"
        Matplotlib colormap for fetch factor lines.
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot.
    include_streaming : bool, default=True
        Whether to include streaming baseline if present in data.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_batch_entropy(
    ...     'experiments/block_size_fetch_factor_eval_anncollection_random.csv',
    ...     title='scDataset Minibatch Entropy',
    ...     save_path='figures/entropy_anndata.pdf'
    ... )
    """
    # Apply Nature-style formatting
    _setup_publication_style()

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for scDataset results (random mode)
    df_scdataset = df[(df["mode"] == "random") & (df["loader"] == "scDataset")].copy()

    # Filter for baseline results (AnnLoader)
    df_baseline = df[(df["mode"] == "random") & (df["loader"] == "AnnLoader")].copy()

    # Filter for streaming baseline (if present and requested)
    df_streaming = pd.DataFrame()
    if include_streaming:
        df_streaming = df[(df["mode"] == "stream") & (df["loader"] == "AnnLoader")].copy()

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Get unique fetch factors and create a color map with log-scale sampling
    fetch_factors = sorted(df_scdataset["fetch_factor"].dropna().unique())
    colors = _get_log_scale_colors(fetch_factors, color_map)

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
            color="#2E86AB",  # Professional blue
            linestyle="--",
            linewidth=2.5,
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
                    color="#2E86AB",
                )

    # Add streaming baseline as horizontal line (if present)
    if not df_streaming.empty:
        streaming_value = df_streaming[y_column].values[0]
        ax.axhline(
            y=streaming_value,
            color="#A23B72",  # Professional magenta
            linestyle=":",
            linewidth=2.5,
            label="Streaming",
        )
        # Add error band for streaming baseline
        if error_column in df_streaming.columns:
            streaming_error = df_streaming[error_column].values[0]
            if streaming_error > 0:
                ax.axhspan(
                    streaming_value - streaming_error,
                    streaming_value + streaming_error,
                    alpha=0.15,
                    color="#A23B72",
                )

    # Format axes
    formatter = FuncFormatter(_format_as_int)

    if x_log_scale:
        ax.set_xscale("log", base=2)
    # Set fixed x-axis ticks
    x_ticks = [1, 4, 16, 64, 256, 1024]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(formatter)

    # Add labels and title with improved font sizes
    ax.set_xlabel("Block Size", fontsize=fontsize_labels, fontweight='medium')
    ax.set_ylabel(y_label, fontsize=fontsize_labels, fontweight='medium')
    ax.set_title(title, fontsize=fontsize_title, fontweight='bold', pad=15)

    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Add legend with improved styling
    legend = ax.legend(
        loc="best",
        fontsize=fontsize_legend,
        framealpha=0.95,
        edgecolor='#cccccc',
    )
    legend.get_frame().set_linewidth(1.0)

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

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
    fontsize_title: int = 18,
    fontsize_labels: int = 16,
    fontsize_ticks: int = 14,
    fontsize_legend: int = 10,
    y_log_scale: bool = False,
    x_log_scale: bool = True,
    color_map: str = "plasma",
    baseline_label: str = None,  # Auto-detected based on collection type
    save_path: str = None,
    show_plot: bool = True,
    include_streaming: bool = True,
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
    fontsize_title : int, default=18
        Font size for title.
    fontsize_labels : int, default=16
        Font size for axis labels.
    fontsize_ticks : int, default=14
        Font size for tick labels.
    fontsize_legend : int, default=13
        Font size for legend.
    y_log_scale : bool, default=False
        Use log scale for Y axis.
    x_log_scale : bool, default=True
        Use log scale for X axis.
    color_map : str, default="plasma"
        Matplotlib colormap for fetch factor lines.
    baseline_label : str, optional
        Label for baseline horizontal line. Auto-detected based on collection type:
        - AnnData: "AnnLoader"
        - HuggingFace: "HuggingFace"
        - BioNeMo: "BioNeMo-SCDL"
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot.
    include_streaming : bool, default=True
        Whether to include streaming baseline if present in data.

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
    ...     y_label='Minibatch Entropy (bits)',
    ...     plot_title='scDataset Minibatch Entropy',
    ...     include_error_bars=True,
    ...     error_column='std_batch_entropy',
    ...     save_path='figures/entropy_anndata.pdf'
    ... )
    """
    # Apply Nature-style formatting
    _setup_publication_style()

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Detect collection type for appropriate baseline filtering
    collection_type = df["collection_type"].iloc[0] if "collection_type" in df.columns else "anncollection"

    # Filter for scDataset results (random mode)
    df_filtered = df[(df["mode"] == "random") & (df["loader"] == "scDataset")].copy()

    # Identify and filter baseline, and determine default label
    if collection_type == "anncollection":
        baseline_loaders = ["AnnLoader"]
        default_baseline_label = "AnnLoader"
    elif collection_type == "huggingface":
        baseline_loaders = ["HuggingFace", "PyTorch DataLoader"]
        default_baseline_label = "HuggingFace"
    elif collection_type == "bionemo":
        baseline_loaders = ["BioNeMo", "PyTorch DataLoader"]
        default_baseline_label = "BioNeMo-SCDL"
    else:
        baseline_loaders = []
        default_baseline_label = "Baseline"

    # Use provided label or auto-detected default
    if baseline_label is None:
        baseline_label = default_baseline_label

    df_baseline = df[(df["mode"] == "random") & (df["loader"].isin(baseline_loaders))]

    # Filter for streaming baseline (if present and requested)
    df_streaming = pd.DataFrame()
    if include_streaming:
        df_streaming = df[(df["mode"] == "stream") & (df["loader"].isin(baseline_loaders))]

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Get unique fetch factors and create a color map with log-scale sampling
    fetch_factors = sorted(df_filtered["fetch_factor"].dropna().unique())
    colors = _get_log_scale_colors(fetch_factors, color_map)

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
            color="#2E86AB",  # Professional blue
            linestyle="--",
            linewidth=2.5,
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
                    color="#2E86AB",
                )

    # Add streaming baseline as horizontal line (if present)
    if not df_streaming.empty:
        streaming_value = df_streaming[y_column].values[0]
        ax.axhline(
            y=streaming_value,
            color="#A23B72",  # Professional magenta
            linestyle=":",
            linewidth=2.5,
            label=f"{baseline_label} (streaming)",
        )
        # Add error band for streaming if available
        if include_error_bars and error_column and error_column in df_streaming.columns:
            streaming_error = df_streaming[error_column].values[0]
            if streaming_error > 0:
                ax.axhspan(
                    streaming_value - streaming_error,
                    streaming_value + streaming_error,
                    alpha=0.15,
                    color="#A23B72",
                )

    # Format axes
    formatter = FuncFormatter(_format_as_int)

    if x_log_scale:
        ax.set_xscale("log", base=2)
    # Set fixed x-axis ticks
    x_ticks = [1, 4, 16, 64, 256, 1024]
    ax.set_xticks(x_ticks)
    ax.xaxis.set_major_formatter(formatter)

    if y_log_scale:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(formatter)
    else:
        # For samples per second, start y-axis at 0
        if "per_second" in y_column:
            ax.set_ylim(bottom=0)

    # Add labels and title with improved font sizes
    ax.set_xlabel("Block Size", fontsize=fontsize_labels, fontweight='medium')
    ax.set_ylabel(y_label, fontsize=fontsize_labels, fontweight='medium')
    ax.set_title(plot_title, fontsize=fontsize_title, fontweight='bold', pad=15)

    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Add legend with improved styling
    if "per_second" in y_column:
        legend = ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=fontsize_legend,
            framealpha=0.95,
            edgecolor='#cccccc',
        )
    else:
        legend = ax.legend(
            loc="best",
            fontsize=fontsize_legend,
            framealpha=0.95,
            edgecolor='#cccccc',
        )
    legend.get_frame().set_linewidth(1.0)

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_tradeoff(
    csv_path: str,
    title: str = "Throughput vs Minibatch Entropy Tradeoff",
    x_column: str = "avg_batch_entropy",
    y_column: str = "samples_per_second",
    x_label: str = "Minibatch Entropy (bits)",
    y_label: str = "Samples Per Second",
    fig_size: tuple = (10, 6),
    fontsize_title: int = 18,
    fontsize_labels: int = 16,
    fontsize_ticks: int = 14,
    fontsize_legend: int = 10,
    x_scale: str = "linear",
    x_power: float = 0.5,
    y_log_scale: bool = True,
    color_map: str = "plasma",
    save_path: str = None,
    show_plot: bool = True,
    include_streaming: bool = True,
    annotate_block_size: bool = False,
):
    """
    Create a tradeoff plot showing throughput vs minibatch entropy.

    Displays the tradeoff between throughput (y-axis) and minibatch entropy (x-axis)
    for different block_size and fetch_factor combinations. Lines are colored by
    fetch_factor, with points connected in order of increasing block_size.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with benchmark results (must contain both throughput
        and minibatch entropy columns, typically AnnData experiments).
    title : str, default="Throughput vs Minibatch Entropy Tradeoff"
        Title for the plot.
    x_column : str, default="avg_batch_entropy"
        Column to plot on X axis (entropy metric).
    y_column : str, default="samples_per_second"
        Column to plot on Y axis (throughput metric).
    x_label : str, default="Minibatch Entropy (bits)"
        Label for X axis.
    y_label : str, default="Samples Per Second"
        Label for Y axis.
    fig_size : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    fontsize_title : int, default=18
        Font size for title.
    fontsize_labels : int, default=16
        Font size for axis labels.
    fontsize_ticks : int, default=14
        Font size for tick labels.
    fontsize_legend : int, default=10
        Font size for legend.
    x_scale : str, default="linear"
        Scale for X axis. Options: "linear", "log", or "power".
        For power scale, use x_power to set the exponent (e.g., 0.5 for sqrt).
    x_power : float, default=0.5
        Power exponent for x-axis when x_scale="power". Use 0.5 for sqrt scale.
    y_log_scale : bool, default=True
        Use log scale for Y axis (throughput).
    color_map : str, default="plasma"
        Matplotlib colormap for fetch factor lines.
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot.
    include_streaming : bool, default=True
        Whether to include streaming baseline if present in data.
    annotate_block_size : bool, default=False
        Whether to annotate points with block_size values.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_tradeoff(
    ...     'experiments/block_size_fetch_factor_eval_anncollection_random.csv',
    ...     title='Throughput vs Minibatch Entropy Tradeoff',
    ...     save_path='figures/tradeoff_anndata.pdf'
    ... )
    """
    # Apply Nature-style formatting
    _setup_publication_style()

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for scDataset results (random mode)
    df_scdataset = df[(df["mode"] == "random") & (df["loader"] == "scDataset")].copy()

    # Filter for baseline results (AnnLoader random)
    df_baseline = df[(df["mode"] == "random") & (df["loader"] == "AnnLoader")].copy()

    # Filter for streaming baseline (if present and requested)
    df_streaming = pd.DataFrame()
    if include_streaming:
        df_streaming = df[(df["mode"] == "stream") & (df["loader"] == "AnnLoader")].copy()

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Get unique fetch factors and create a color map with log-scale sampling
    fetch_factors = sorted(df_scdataset["fetch_factor"].dropna().unique())
    colors = _get_log_scale_colors(fetch_factors, color_map)

    # Plot a line for each fetch factor (points connected by increasing block_size)
    for i, ff in enumerate(fetch_factors):
        df_ff = df_scdataset[df_scdataset["fetch_factor"] == ff]
        # Sort by block_size to connect points in order
        df_ff = df_ff.sort_values("block_size")

        ff_label = int(ff) if float(ff).is_integer() else ff
        ax.plot(
            df_ff[x_column],
            df_ff[y_column],
            marker="o",
            label=f"fetch_factor = {ff_label}",
            linewidth=2,
            markersize=8,
            color=colors[i],
        )

        # Optionally annotate points with block_size values
        if annotate_block_size:
            for _, row in df_ff.iterrows():
                bs_label = int(row["block_size"]) if float(row["block_size"]).is_integer() else row["block_size"]
                ax.annotate(
                    str(bs_label),
                    (row[x_column], row[y_column]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    alpha=0.7,
                )

    # Add baseline as a reference point (random sampling)
    if not df_baseline.empty and x_column in df_baseline.columns:
        baseline_x = df_baseline[x_column].values[0]
        baseline_y = df_baseline[y_column].values[0]
        ax.scatter(
            [baseline_x],
            [baseline_y],
            marker="*",
            s=300,
            color="#2E86AB",  # Professional blue
            edgecolors="black",
            linewidths=1,
            label="Random Sampling",
            zorder=10,
        )

    # Add streaming baseline as a reference point (if present)
    if not df_streaming.empty and x_column in df_streaming.columns:
        streaming_x = df_streaming[x_column].values[0]
        streaming_y = df_streaming[y_column].values[0]
        ax.scatter(
            [streaming_x],
            [streaming_y],
            marker="D",
            s=150,
            color="#A23B72",  # Professional magenta
            edgecolors="black",
            linewidths=1,
            label="Streaming",
            zorder=10,
        )

    # Format axes
    formatter = FuncFormatter(_format_as_int)

    if x_scale == "log":
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(formatter)
    elif x_scale == "power":
        ax.set_xscale("power", power=x_power)
        # Add padding to avoid points overlapping with y-axis
        x_min, x_max = ax.get_xlim()
        x_padding = (x_max - x_min) * 0.05
        ax.set_xlim(left=max(0, x_min - x_padding))

    if y_log_scale:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(formatter)

    # Add labels and title with improved font sizes
    ax.set_xlabel(x_label, fontsize=fontsize_labels, fontweight='medium')
    ax.set_ylabel(y_label, fontsize=fontsize_labels, fontweight='medium')
    ax.set_title(title, fontsize=fontsize_title, fontweight='bold', pad=15)

    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Add legend with improved styling
    legend = ax.legend(
        loc="best",
        fontsize=fontsize_legend,
        framealpha=0.95,
        edgecolor='#cccccc',
    )
    legend.get_frame().set_linewidth(1.0)

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_tradeoff_scatter(
    csv_path: str,
    title: str = "Throughput vs Minibatch Entropy Tradeoff",
    x_column: str = "avg_batch_entropy",
    y_column: str = "samples_per_second",
    x_label: str = "Minibatch Entropy (bits)",
    y_label: str = "Samples Per Second",
    fig_size: tuple = (10, 6),
    fontsize_title: int = 18,
    fontsize_labels: int = 16,
    fontsize_ticks: int = 14,
    fontsize_legend: int = 10,
    x_scale: str = "linear",
    x_power: float = 0.5,
    y_log_scale: bool = True,
    color_map: str = "plasma",
    size_range: tuple = (50, 400),
    save_path: str = None,
    show_plot: bool = True,
    include_streaming: bool = True,
):
    """
    Create a scatter tradeoff plot showing throughput vs minibatch entropy.

    Displays the tradeoff between throughput (y-axis) and minibatch entropy (x-axis)
    using a scatter plot where:
    - Color encodes fetch_factor
    - Marker size encodes block_size (larger markers = larger block_size)

    Parameters
    ----------
    csv_path : str
        Path to CSV file with benchmark results (must contain both throughput
        and minibatch entropy columns, typically AnnData experiments).
    title : str, default="Throughput vs Minibatch Entropy Tradeoff"
        Title for the plot.
    x_column : str, default="avg_batch_entropy"
        Column to plot on X axis (entropy metric).
    y_column : str, default="samples_per_second"
        Column to plot on Y axis (throughput metric).
    x_label : str, default="Minibatch Entropy (bits)"
        Label for X axis.
    y_label : str, default="Samples Per Second"
        Label for Y axis.
    fig_size : tuple, default=(10, 6)
        Figure size (width, height) in inches.
    fontsize_title : int, default=18
        Font size for title.
    fontsize_labels : int, default=16
        Font size for axis labels.
    fontsize_ticks : int, default=14
        Font size for tick labels.
    fontsize_legend : int, default=10
        Font size for legend.
    x_scale : str, default="linear"
        Scale for X axis. Options: "linear", "log", or "power".
    x_power : float, default=0.5
        Power exponent for x-axis when x_scale="power". Use 0.5 for sqrt scale.
    y_log_scale : bool, default=True
        Use log scale for Y axis (throughput).
    color_map : str, default="plasma"
        Matplotlib colormap for fetch factor colors.
    size_range : tuple, default=(50, 400)
        Range of marker sizes (min, max) in pixels² for block_size encoding.
    save_path : str, optional
        Path to save figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot.
    include_streaming : bool, default=True
        Whether to include streaming baseline if present in data.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib figure object.

    Examples
    --------
    >>> fig = plot_tradeoff_scatter(
    ...     'experiments/block_size_fetch_factor_eval_anncollection_random.csv',
    ...     title='Throughput vs Minibatch Entropy Tradeoff',
    ...     save_path='figures/tradeoff_scatter_anndata.pdf'
    ... )
    """
    # Apply Nature-style formatting
    _setup_publication_style()

    # Read the CSV file
    df = pd.read_csv(csv_path)

    # Filter for scDataset results (random mode)
    df_scdataset = df[(df["mode"] == "random") & (df["loader"] == "scDataset")].copy()

    # Filter for baseline results (AnnLoader random)
    df_baseline = df[(df["mode"] == "random") & (df["loader"] == "AnnLoader")].copy()

    # Filter for streaming baseline (if present and requested)
    df_streaming = pd.DataFrame()
    if include_streaming:
        df_streaming = df[(df["mode"] == "stream") & (df["loader"] == "AnnLoader")].copy()

    # Set up the figure
    fig, ax = plt.subplots(figsize=fig_size)
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

    # Get unique fetch factors and block sizes
    fetch_factors = sorted(df_scdataset["fetch_factor"].dropna().unique())
    block_sizes = sorted(df_scdataset["block_size"].dropna().unique())

    # Create color mapping for fetch_factor (using log scale)
    colors = _get_log_scale_colors(fetch_factors, color_map)
    ff_to_color = dict(zip(fetch_factors, colors))

    # Create size mapping for block_size (log scale)
    size_min, size_max = size_range
    log_block_sizes = np.log2(np.maximum(block_sizes, 1))
    log_min, log_max = log_block_sizes.min(), log_block_sizes.max()
    if log_max == log_min:
        normalized_sizes = np.full(len(block_sizes), (size_min + size_max) / 2)
    else:
        normalized_sizes = (log_block_sizes - log_min) / (log_max - log_min)
        normalized_sizes = size_min + normalized_sizes * (size_max - size_min)
    bs_to_size = dict(zip(block_sizes, normalized_sizes))

    # Plot scatter points
    for _, row in df_scdataset.iterrows():
        ff = row["fetch_factor"]
        bs = row["block_size"]
        ax.scatter(
            row[x_column],
            row[y_column],
            s=bs_to_size[bs],
            c=[ff_to_color[ff]],
            edgecolors="white",
            linewidths=0.5,
            alpha=0.7,
            zorder=5,
        )

    # Create legend for fetch_factor (color)
    legend_elements_color = []
    for ff in fetch_factors:
        ff_label = int(ff) if float(ff).is_integer() else ff
        legend_elements_color.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=ff_to_color[ff],
                markersize=8,
                label=f"ff = {ff_label}",
            )
        )

    # Create legend for block_size (size) - show a subset
    legend_block_sizes = [block_sizes[0], block_sizes[len(block_sizes) // 2], block_sizes[-1]]
    legend_elements_size = []
    for bs in legend_block_sizes:
        bs_label = int(bs) if float(bs).is_integer() else bs
        # Convert scatter size to marker size (sqrt since scatter uses area)
        marker_size = np.sqrt(bs_to_size[bs]) / 2
        legend_elements_size.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor="gray",
                markersize=marker_size,
                label=f"bs = {bs_label}",
            )
        )

    # Add baseline as a reference point (random sampling)
    if not df_baseline.empty and x_column in df_baseline.columns:
        baseline_x = df_baseline[x_column].values[0]
        baseline_y = df_baseline[y_column].values[0]
        ax.scatter(
            [baseline_x],
            [baseline_y],
            marker="*",
            s=350,
            color="#2E86AB",
            edgecolors="black",
            linewidths=1,
            label="Random Sampling",
            zorder=10,
        )
        legend_elements_color.append(
            plt.Line2D(
                [0], [0],
                marker="*",
                color="w",
                markerfacecolor="#2E86AB",
                markeredgecolor="black",
                markersize=12,
                label="Random Sampling",
            )
        )

    # Add streaming baseline as a reference point (if present)
    if not df_streaming.empty and x_column in df_streaming.columns:
        streaming_x = df_streaming[x_column].values[0]
        streaming_y = df_streaming[y_column].values[0]
        ax.scatter(
            [streaming_x],
            [streaming_y],
            marker="D",
            s=180,
            color="#A23B72",
            edgecolors="black",
            linewidths=1,
            label="Streaming",
            zorder=10,
        )
        legend_elements_color.append(
            plt.Line2D(
                [0], [0],
                marker="D",
                color="w",
                markerfacecolor="#A23B72",
                markeredgecolor="black",
                markersize=10,
                label="Streaming",
            )
        )

    # Format axes
    formatter = FuncFormatter(_format_as_int)

    if x_scale == "log":
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(formatter)
    elif x_scale == "power":
        ax.set_xscale("power", power=x_power)
        # Add padding to avoid points overlapping with y-axis
        x_min, x_max = ax.get_xlim()
        x_padding = (x_max - x_min) * 0.05
        ax.set_xlim(left=max(0, x_min - x_padding))

    if y_log_scale:
        ax.set_yscale("log", base=2)
        ax.yaxis.set_major_formatter(formatter)

    # Add labels and title with improved font sizes
    ax.set_xlabel(x_label, fontsize=fontsize_labels, fontweight='medium')
    ax.set_ylabel(y_label, fontsize=fontsize_labels, fontweight='medium')
    ax.set_title(title, fontsize=fontsize_title, fontweight='bold', pad=15)

    # Style tick labels
    ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    # Add two legends: one for color (fetch_factor), one for size (block_size)
    # Place legends outside the plot area
    legend1 = ax.legend(
        handles=legend_elements_color,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=fontsize_legend,
        framealpha=0.95,
        edgecolor='#cccccc',
        title="fetch_factor",
        title_fontsize=fontsize_legend,
    )
    legend1.get_frame().set_linewidth(1.0)
    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=legend_elements_size,
        loc="lower left",
        bbox_to_anchor=(1.02, 0.0),
        fontsize=fontsize_legend,
        framealpha=0.95,
        edgecolor='#cccccc',
        title="block_size",
        title_fontsize=fontsize_legend,
    )
    legend2.get_frame().set_linewidth(1.0)

    # Add subtle grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_streaming_comparison(
    csv_path: str,
    title: str = "Streaming Throughput Comparison",
    save_path: str | None = None,
    show_plot: bool = True,
    figsize: tuple = (10, 6),
    fontsize_title: int = 16,
    fontsize_labels: int = 14,
    fontsize_ticks: int = 12,
    fontsize_legend: int = 11,
    annloader_color: str = "#7f7f7f",  # Gray for AnnLoader baseline
    color_map: str = "plasma",
):
    """
    Create a bar plot comparing AnnLoader streaming vs scDataset streaming.

    Shows throughput (samples/sec) for AnnLoader streaming baseline and
    scDataset streaming with multiple fetch_factor values. Uses the same
    color coding as other plots (plasma colormap for fetch_factor values).

    Parameters
    ----------
    csv_path : str
        Path to the CSV file with benchmark results.
    title : str, default="Streaming Throughput Comparison"
        Plot title.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot.
    figsize : tuple, default=(10, 6)
        Figure size in inches.
    fontsize_title : int, default=16
        Font size for plot title.
    fontsize_labels : int, default=14
        Font size for axis labels.
    fontsize_ticks : int, default=12
        Font size for tick labels.
    fontsize_legend : int, default=11
        Font size for legend.
    annloader_color : str, default="#7f7f7f"
        Color for AnnLoader baseline bar (gray by default).
    color_map : str, default="plasma"
        Colormap to use for scDataset fetch_factor bars.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> plot_streaming_comparison(
    ...     "experiments/block_size_fetch_factor_eval_anncollection_random.csv",
    ...     title="Streaming Throughput: AnnLoader vs scDataset",
    ...     save_path="figures/streaming_comparison.pdf"
    ... )
    """
    _setup_publication_style()

    # Load data
    df = pd.read_csv(csv_path)

    # Filter streaming data
    stream_df = df[df["mode"] == "stream"].copy()

    # Get AnnLoader streaming throughput
    annloader_data = stream_df[stream_df["loader"] == "AnnLoader"]
    if len(annloader_data) > 0:
        annloader_throughput = annloader_data["samples_per_second"].values[0]
    else:
        annloader_throughput = None

    # Get scDataset streaming throughput by fetch_factor
    scdataset_data = stream_df[stream_df["loader"] == "scDataset"].copy()
    scdataset_data = scdataset_data.sort_values("fetch_factor")

    # Get colors for fetch_factor values using the same log-scale color mapping
    fetch_factors = scdataset_data["fetch_factor"].values
    ff_colors = _get_log_scale_colors(fetch_factors, cmap_name=color_map, cmap_range=(0.0, 0.85))
    ff_color_map = dict(zip(sorted(fetch_factors), ff_colors))

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare bar positions
    # AnnLoader bar first, then scDataset bars for each fetch_factor
    labels = []
    throughputs = []
    colors = []

    # Add AnnLoader bar
    if annloader_throughput is not None:
        labels.append("AnnLoader\nStreaming")
        throughputs.append(annloader_throughput)
        colors.append(annloader_color)

    # Add scDataset bars with fetch_factor-based coloring
    for _, row in scdataset_data.iterrows():
        ff = int(row["fetch_factor"])
        labels.append(f"scDataset\nfetch_factor={ff}")
        throughputs.append(row["samples_per_second"])
        colors.append(ff_color_map[row["fetch_factor"]])

    # Create bar plot
    x_positions = np.arange(len(labels))
    bars = ax.bar(x_positions, throughputs, color=colors, edgecolor="white", linewidth=1.5)

    # Add value labels on top of bars
    for bar, value in zip(bars, throughputs):
        height = bar.get_height()
        ax.annotate(
            f"{value:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize_ticks - 1,
            color="#333333",
        )

    # Customize axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=fontsize_ticks - 3)
    ax.set_ylabel("Throughput (samples/sec)", fontsize=fontsize_labels)
    ax.set_title(title, fontsize=fontsize_title, fontweight="bold", pad=15)

    # Y-axis formatting
    ax.yaxis.set_major_formatter(FuncFormatter(_format_as_int))
    ax.tick_params(axis="y", labelsize=fontsize_ticks)

    # Add subtle grid (y-axis only for bar plots)
    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5, axis="y")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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

    Creates throughput and minibatch entropy plots for all available
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
            "title_suffix": "AnnData",
            "has_entropy": True,
        },
        {
            "csv": f"{experiments_dir}/block_size_fetch_factor_eval_huggingface_random.csv",
            "name": "HuggingFace",
            "title_suffix": "HuggingFace",
            "has_entropy": False,
        },
        {
            "csv": f"{experiments_dir}/block_size_fetch_factor_eval_bionemo_random.csv",
            "name": "BioNeMo",
            "title_suffix": "BioNeMo-SCDL",
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
            title=f"scDataset Throughput with {exp['title_suffix']}",
            save_path=f"{figures_dir}/throughput_{exp['name'].lower()}.pdf",
            show_plot=False,
        )

        # Batch entropy plot (only for AnnData which has plate info)
        if exp["has_entropy"]:
            plot_batch_entropy(
                exp["csv"],
                title=f"scDataset Minibatch Entropy",
                save_path=f"{figures_dir}/batch_entropy_{exp['name'].lower()}.pdf",
                show_plot=False,
            )

            # Tradeoff plot (only for AnnData which has both metrics)
            plot_tradeoff(
                exp["csv"],
                title="Throughput vs Minibatch Entropy Tradeoff",
                save_path=f"{figures_dir}/tradeoff_{exp['name'].lower()}.pdf",
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
