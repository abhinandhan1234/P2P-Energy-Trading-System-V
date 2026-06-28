"""Unit tests for config_loader.py."""

from __future__ import annotations

# standard library
from pathlib import Path

# third party
import pytest
import yaml

# local
from p2p_energy_trading.training.config_loader import (
    load_training_config,
    validate_config,
)


@pytest.fixture
def base_config_dict() -> dict:
    """Provide a minimal valid configuration dictionary."""
    return {
        "environment": {
            "env_name": "p2p_energy_trading",
            "episode_length": 168,
            "grid_buy_rate": 8.15,
            "grid_sell_rate": 3.56,
            "profile_data_dir": "data/processed",
            "seed": 42,
        },
        "ppo": {
            "train_batch_size_per_learner": 7056,
            "rollout_fragment_length": 168,
            "sgd_minibatch_size": 504,
            "num_sgd_iter": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_param: ": 0.2,
            "entropy_coeff": 0.01,
            "lr": 3.0e-4,
            "vf_loss_coeff": 0.5,
            "actor": {"hidden_layers": [128, 128]},
            "critic": {"hidden_layers": [256, 256]},
        },
        "curriculum": {
            "stages": [
                {"name": "debug", "episode_length": 24},
            ]
        },
        "hardware": {
            "num_env_runners": 1,
        },
    }


def write_yaml_config(tmp_path: Path, data: dict) -> Path:
    """Helper to write configuration dictionary to a YAML file."""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return config_file


def test_valid_config_loading(tmp_path: Path, base_config_dict: dict) -> None:
    """Verify loading of a fully valid YAML configuration."""
    config_file = write_yaml_config(tmp_path, base_config_dict)
    config = load_training_config(str(config_file))

    assert config["environment"]["env_name"] == "p2p_energy_trading"
    assert config["environment"]["episode_length"] == 168
    assert config["ppo"]["lr"] == 3.0e-4
    assert Path(config["environment"]["profile_data_dir"]).is_absolute()


def test_config_validation_checks(base_config_dict: dict) -> None:
    """Verify validation of ranges, types and structural sections."""
    # Invalid top level
    bad_config = {"ppo": {}}
    with pytest.raises(ValueError, match="Missing required configuration section"):
        validate_config(bad_config)

    # Invalid learning rate
    base_config_dict["ppo"]["lr"] = -0.01
    with pytest.raises(ValueError, match="ppo.lr must be a positive number"):
        validate_config(base_config_dict)

    # Invalid gamma
    base_config_dict["ppo"]["lr"] = 3.0e-4
    base_config_dict["ppo"]["gamma"] = 1.05
    with pytest.raises(ValueError, match="ppo.gamma must be in range"):
        validate_config(base_config_dict)

    # Invalid gae_lambda
    base_config_dict["ppo"]["gamma"] = 0.99
    base_config_dict["ppo"]["gae_lambda"] = 0.0
    with pytest.raises(ValueError, match="ppo.gae_lambda must be in range"):
        validate_config(base_config_dict)

    # Invalid arbitrage check
    base_config_dict["ppo"]["gae_lambda"] = 0.95
    base_config_dict["environment"]["grid_buy_rate"] = 5.0
    base_config_dict["environment"]["grid_sell_rate"] = 5.5
    with pytest.raises(ValueError, match="must be strictly greater than"):
        validate_config(base_config_dict)


def test_minibatch_divisibility(base_config_dict: dict) -> None:
    """Verify that batch size not divisible by sgd_minibatch_size raises ValueError."""
    base_config_dict["ppo"]["train_batch_size_per_learner"] = 7000
    base_config_dict["ppo"]["sgd_minibatch_size"] = 512
    with pytest.raises(ValueError, match="must be evenly divisible"):
        validate_config(base_config_dict)


def test_cli_overrides_application(tmp_path: Path, base_config_dict: dict) -> None:
    """Verify that flat dot-notated CLI overrides merge correctly."""
    config_file = write_yaml_config(tmp_path, base_config_dict)

    overrides = {
        "environment.seed": 100,
        "ppo.lr": 1.0e-4,
        "hardware.num_env_runners": 4,
    }

    config = load_training_config(str(config_file), overrides)

    assert config["environment"]["seed"] == 100
    assert config["ppo"]["lr"] == 1.0e-4
    assert config["hardware"]["num_env_runners"] == 4
