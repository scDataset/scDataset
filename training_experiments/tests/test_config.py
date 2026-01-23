"""
Tests for configuration loading utilities.
"""

import os
import tempfile

import pytest
import yaml

from training_experiments.utils.config import (
    get_block_size_for_strategy,
    get_data_paths,
    get_strategy_config,
    load_config,
    merge_config_with_args,
)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_default_config(self):
        """Test loading the default configuration file."""
        config = load_config()

        assert config is not None
        assert "training" in config
        assert "scdataset" in config
        assert "strategies" in config

    def test_load_custom_config(self):
        """Test loading a custom configuration file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"training": {"batch_size": 128}}, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)
            assert config["training"]["batch_size"] == 128
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_config_raises(self):
        """Test that loading a nonexistent config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestGetStrategyConfig:
    """Tests for get_strategy_config function."""

    def test_get_existing_strategy(self):
        """Test getting config for an existing strategy."""
        config = {
            "strategies": {
                "block_shuffling": {"block_size": 4, "enabled": True},
            }
        }

        strategy_config = get_strategy_config(config, "block_shuffling")

        assert strategy_config["block_size"] == 4
        assert strategy_config["enabled"] is True

    def test_get_nonexistent_strategy(self):
        """Test getting config for a nonexistent strategy."""
        config = {"strategies": {}}

        strategy_config = get_strategy_config(config, "nonexistent")

        assert strategy_config == {}


class TestMergeConfigWithArgs:
    """Tests for merge_config_with_args function."""

    def test_config_values_used_when_no_args(self):
        """Test that config values are used when no CLI args provided."""
        config = {
            "training": {"batch_size": 128, "num_epochs": 5, "learning_rate": 0.01},
            "scdataset": {"fetch_factor": 8, "num_workers": 4},
            "weights": {"min_count_baseline": 500},
            "output": {"save_dir": "/custom/path", "log_interval": 50},
        }

        merged = merge_config_with_args(config)

        assert merged["batch_size"] == 128
        assert merged["num_epochs"] == 5
        assert merged["learning_rate"] == 0.01
        assert merged["fetch_factor"] == 8
        assert merged["num_workers"] == 4
        assert merged["min_count_baseline"] == 500
        assert merged["save_dir"] == "/custom/path"
        assert merged["log_interval"] == 50

    def test_cli_args_override_config(self):
        """Test that CLI arguments override config values."""
        config = {
            "training": {"batch_size": 128, "num_epochs": 5},
            "scdataset": {"fetch_factor": 8, "num_workers": 4},
        }

        merged = merge_config_with_args(
            config,
            batch_size=256,
            num_epochs=10,
        )

        # CLI values override
        assert merged["batch_size"] == 256
        assert merged["num_epochs"] == 10
        # Config values used for others
        assert merged["fetch_factor"] == 8
        assert merged["num_workers"] == 4

    def test_defaults_used_when_missing(self):
        """Test that defaults are used when config is empty."""
        config = {}

        merged = merge_config_with_args(config)

        assert merged["batch_size"] == 64
        assert merged["num_epochs"] == 1
        assert merged["learning_rate"] == 0.001
        assert merged["fetch_factor"] == 256
        assert merged["num_workers"] == 8  # Default from config.py
        assert merged["min_count_baseline"] == 1000


class TestGetBlockSizeForStrategy:
    """Tests for get_block_size_for_strategy function."""

    def test_override_takes_precedence(self):
        """Test that override value takes precedence over config."""
        config = {
            "strategies": {
                "block_shuffling": {"block_size": 4},
            }
        }

        block_size = get_block_size_for_strategy(config, "block_shuffling", override=8)

        assert block_size == 8

    def test_config_value_used_when_no_override(self):
        """Test that config value is used when no override provided."""
        config = {
            "strategies": {
                "block_shuffling": {"block_size": 4},
            }
        }

        block_size = get_block_size_for_strategy(config, "block_shuffling")

        assert block_size == 4

    def test_returns_none_when_not_configured(self):
        """Test that None is returned when strategy not configured."""
        config = {"strategies": {}}

        block_size = get_block_size_for_strategy(config, "block_shuffling")

        assert block_size is None


class TestGetDataPaths:
    """Tests for get_data_paths function."""

    def test_gets_configured_paths(self):
        """Test getting configured data paths."""
        config = {
            "data": {
                "h5ad_dir": "/path/to/h5ad",
                "label_dir": "/path/to/labels",
            }
        }

        paths = get_data_paths(config)

        assert paths["h5ad_dir"] == "/path/to/h5ad"
        assert paths["label_dir"] == "/path/to/labels"

    def test_returns_none_for_missing_paths(self):
        """Test that None is returned for missing paths."""
        config = {}

        paths = get_data_paths(config)

        assert paths["h5ad_dir"] is None
        assert paths["label_dir"] is None
