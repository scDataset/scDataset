"""
Configuration loading utilities for training experiments.

This module provides functionality to load and merge configuration from YAML files
and command-line arguments.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Default config path relative to this module
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Parameters
    ----------
    config_path : str, optional
        Path to the YAML config file. If None, loads the default config.

    Returns
    -------
    dict
        Configuration dictionary

    Raises
    ------
    FileNotFoundError
        If the config file does not exist
    """
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config or {}


def get_strategy_config(config: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific strategy.

    Parameters
    ----------
    config : dict
        Full configuration dictionary
    strategy_name : str
        Name of the strategy

    Returns
    -------
    dict
        Strategy-specific configuration
    """
    strategies = config.get("strategies", {})
    return strategies.get(strategy_name, {})


def merge_config_with_args(
    config: Dict[str, Any],
    batch_size: Optional[int] = None,
    num_epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    fetch_factor: Optional[int] = None,
    num_workers: Optional[int] = None,
    min_count_baseline: Optional[int] = None,
    block_size: Optional[int] = None,
    save_dir: Optional[str] = None,
    log_interval: Optional[int] = None,
    train_plates: Optional[list] = None,
    test_plates: Optional[list] = None,
    max_train_cells: Optional[int] = None,
    max_test_cells: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Merge configuration with command-line arguments.

    Command-line arguments take precedence over config file values.

    Parameters
    ----------
    config : dict
        Configuration dictionary from YAML file
    batch_size : int, optional
        Override batch size
    num_epochs : int, optional
        Override number of epochs
    learning_rate : float, optional
        Override learning rate
    fetch_factor : int, optional
        Override fetch factor
    num_workers : int, optional
        Override number of workers
    min_count_baseline : int, optional
        Override minimum count baseline
    block_size : int, optional
        Override block size
    save_dir : str, optional
        Override save directory
    log_interval : int, optional
        Override log interval
    train_plates : list, optional
        Override training plates
    test_plates : list, optional
        Override test plates
    max_train_cells : int, optional
        Override max training cells (pilot mode)
    max_test_cells : int, optional
        Override max test cells (pilot mode)

    Returns
    -------
    dict
        Merged configuration with CLI overrides applied
    """
    # Start with config values
    training = config.get("training", {})
    scdataset = config.get("scdataset", {})
    weights = config.get("weights", {})
    output = config.get("output", {})
    data = config.get("data", {})

    # Apply CLI overrides (None means use config value)
    merged = {
        "batch_size": batch_size
        if batch_size is not None
        else training.get("batch_size", 64),
        "num_epochs": num_epochs
        if num_epochs is not None
        else training.get("num_epochs", 1),
        "learning_rate": learning_rate
        if learning_rate is not None
        else training.get("learning_rate", 0.001),
        "fetch_factor": fetch_factor
        if fetch_factor is not None
        else scdataset.get("fetch_factor", 256),
        "num_workers": num_workers
        if num_workers is not None
        else scdataset.get("num_workers", 8),
        "min_count_baseline": min_count_baseline
        if min_count_baseline is not None
        else weights.get("min_count_baseline", 1000),
        "block_size": block_size,  # This is strategy-specific, no global default
        "save_dir": save_dir
        if save_dir is not None
        else output.get("save_dir", "./results"),
        "log_interval": log_interval
        if log_interval is not None
        else output.get("log_interval", 100),
        # Pilot mode options
        "train_plates": train_plates
        if train_plates is not None
        else data.get("train_plates"),
        "test_plates": test_plates
        if test_plates is not None
        else data.get("test_plates"),
        "max_train_cells": max_train_cells
        if max_train_cells is not None
        else data.get("max_train_cells"),
        "max_test_cells": max_test_cells
        if max_test_cells is not None
        else data.get("max_test_cells"),
    }

    return merged


def get_block_size_for_strategy(
    config: Dict[str, Any], strategy_name: str, override: Optional[int] = None
) -> Optional[int]:
    """
    Get block_size for a specific strategy.

    Parameters
    ----------
    config : dict
        Full configuration dictionary
    strategy_name : str
        Name of the strategy
    override : int, optional
        CLI override for block_size. If provided, takes precedence.

    Returns
    -------
    int or None
        Block size for the strategy, or None to use defaults
    """
    if override is not None:
        return override

    strategy_config = get_strategy_config(config, strategy_name)
    return strategy_config.get("block_size")


def get_data_paths(config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Get data directory paths from config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary

    Returns
    -------
    dict
        Dictionary with 'h5ad_dir' and 'label_dir' keys
    """
    data = config.get("data", {})
    return {
        "h5ad_dir": data.get("h5ad_dir"),
        "label_dir": data.get("label_dir"),
    }
