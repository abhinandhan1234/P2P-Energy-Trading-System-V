"""Configuration Loader for the P2P Energy Trading training pipeline.

Loads, merges, and validates configuration parameters from YAML files and CLI overrides.
Ensures all file paths are resolved to absolute paths relative to the project root,
and validates parameter bounds and consistency constraints
(e.g. batch size divisibility).

Design reference: docs/module_8_training_pipeline.md §2
"""

from __future__ import annotations

# standard library
import logging
import os
from pathlib import Path
from typing import Any

# third party
import yaml

logger = logging.getLogger(__name__)

# Find the project root directory dynamically (4 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_absolute_path(relative_path: str) -> str:
    """Resolve a relative path string to an absolute path string
    relative to project root.

    Args:
        relative_path: A file path string.

    Returns:
        The absolute path string.
    """
    path = Path(relative_path)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def load_training_config(
    config_path: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the YAML training configuration and apply overrides.

    Args:
        config_path: Path to the YAML configuration file.
        overrides: Dictionary of flat dot-separated override keys
            (e.g. 'environment.seed').

    Returns:
        The loaded and validated configuration dictionary.

    Raises:
        FileNotFoundError: If the config file cannot be found.
        ValueError: If config parameters fail validation checks.
    """
    absolute_config_path = resolve_absolute_path(config_path)
    if not os.path.exists(absolute_config_path):
        raise FileNotFoundError(
            f"Configuration file not found at: '{absolute_config_path}'"
        )

    with open(absolute_config_path, encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML configuration: {e}") from e

    # Ensure structure exists
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary.")

    # Apply overrides (flat dot-notated format: 'environment.seed')
    if overrides:
        for dot_key, value in overrides.items():
            if value is None:
                continue
            parts = dot_key.split(".")
            target = config
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = value

    # Resolve all relative paths to absolute paths
    if "environment" in config and "profile_data_dir" in config["environment"]:
        config["environment"]["profile_data_dir"] = resolve_absolute_path(
            config["environment"]["profile_data_dir"]
        )

    if "checkpoint" in config:
        if "checkpoint_dir" in config["checkpoint"]:
            config["checkpoint"]["checkpoint_dir"] = resolve_absolute_path(
                config["checkpoint"]["checkpoint_dir"]
            )
        if "best_model_dir" in config["checkpoint"]:
            config["checkpoint"]["best_model_dir"] = resolve_absolute_path(
                config["checkpoint"]["best_model_dir"]
            )

    if "logging" in config:
        for key in ["tensorboard_dir", "metrics_dir", "evaluation_dir", "log_file"]:
            if key in config["logging"]:
                config["logging"][key] = resolve_absolute_path(config["logging"][key])

    # Validate parameters
    validate_config(config)

    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate values, types, and cross-consistency constraints of the configuration.

    Args:
        config: Loaded configuration dictionary.

    Raises:
        ValueError: If a parameter violates constraints.
        TypeError: If a parameter type is invalid.
    """
    # 1. Check top-level sections
    required_sections = ["environment", "ppo", "curriculum", "hardware"]
    for sec in required_sections:
        if sec not in config or not isinstance(config[sec], dict):
            raise ValueError(f"Missing required configuration section: '{sec}'")

    # 2. Validate environment settings
    env = config["environment"]
    if "episode_length" in env:
        el = env["episode_length"]
        if not isinstance(el, int):
            raise TypeError("environment.episode_length must be an integer.")
        if not (24 <= el <= 8760):
            raise ValueError(
                f"environment.episode_length ({el}) must be in range [24, 8760]."
            )

    buy_rate = env.get("grid_buy_rate")
    sell_rate = env.get("grid_sell_rate")
    if buy_rate is not None:
        if not isinstance(buy_rate, (int, float)) or buy_rate <= 0:
            raise ValueError("environment.grid_buy_rate must be a positive number.")
    if sell_rate is not None:
        if not isinstance(sell_rate, (int, float)) or sell_rate <= 0:
            raise ValueError("environment.grid_sell_rate must be a positive number.")
    if buy_rate is not None and sell_rate is not None:
        if buy_rate <= sell_rate:
            raise ValueError(
                f"grid_buy_rate ({buy_rate}) must be strictly greater than "
                f"grid_sell_rate ({sell_rate}) to prevent arbitrage."
            )

    # 3. Validate PPO settings
    ppo = config["ppo"]
    lr = ppo.get("lr")
    if lr is not None:
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError("ppo.lr must be a positive number.")

    gamma = ppo.get("gamma")
    if gamma is not None:
        if not isinstance(gamma, (int, float)) or not (0.0 < gamma <= 1.0):
            raise ValueError("ppo.gamma must be in range (0.0, 1.0].")

    gae_lambda = ppo.get("gae_lambda")
    if gae_lambda is not None:
        if not isinstance(gae_lambda, (int, float)) or not (0.0 < gae_lambda <= 1.0):
            raise ValueError("ppo.gae_lambda must be in range (0.0, 1.0].")

    clip_param = ppo.get("clip_param")
    if clip_param is not None:
        if not isinstance(clip_param, (int, float)) or not (0.0 < clip_param < 1.0):
            raise ValueError("ppo.clip_param must be in range (0.0, 1.0).")

    entropy_coeff = ppo.get("entropy_coeff")
    if entropy_coeff is not None:
        if not isinstance(entropy_coeff, (int, float)) or entropy_coeff < 0:
            raise ValueError("ppo.entropy_coeff must be a non-negative number.")

    vf_loss_coeff = ppo.get("vf_loss_coeff")
    if vf_loss_coeff is not None:
        if not isinstance(vf_loss_coeff, (int, float)) or vf_loss_coeff <= 0:
            raise ValueError("ppo.vf_loss_coeff must be a positive number.")

    # 4. Check batch size and minibatch divisibility
    batch_size = ppo.get("train_batch_size_per_learner")
    minibatch_size = ppo.get("sgd_minibatch_size")
    if batch_size is not None and minibatch_size is not None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                "ppo.train_batch_size_per_learner must be a positive integer."
            )
        if not isinstance(minibatch_size, int) or minibatch_size <= 0:
            raise ValueError("ppo.sgd_minibatch_size must be a positive integer.")
        if batch_size % minibatch_size != 0:
            raise ValueError(
                f"train_batch_size_per_learner ({batch_size}) must be evenly divisible "
                f"by sgd_minibatch_size ({minibatch_size})."
            )

    # 5. Validate network architectures
    for net in ["actor", "critic"]:
        if net in ppo:
            layers = ppo[net].get("hidden_layers")
            if layers is not None:
                if not isinstance(layers, list) or len(layers) == 0:
                    raise ValueError(
                        f"ppo.{net}.hidden_layers must be a non-empty list."
                    )
                for layer_dim in layers:
                    if not isinstance(layer_dim, int) or layer_dim <= 0:
                        raise ValueError(
                            f"ppo.{net}.hidden_layers dimensions must be"
                            " positive integers."
                        )
