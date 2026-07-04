"""Idempotent environment registration helper for Ray RLlib."""

from __future__ import annotations

# standard library
import logging

logger = logging.getLogger(__name__)

try:
    # third party
    from ray.tune.registry import ENV_CREATOR, _global_registry, register_env

    RAY_AVAILABLE = True
except ImportError:
    register_env = None  # type: ignore
    _global_registry = None
    ENV_CREATOR = None
    RAY_AVAILABLE = False


def register_p2p_environment() -> None:
    """Register the custom environment with Ray RLlib in an idempotent manner."""
    if not RAY_AVAILABLE:
        logger.warning("Ray/RLlib is not installed, skipping environment registration.")
        return

    # local
    from p2p_energy_trading.constants import ENV_NAME
    from p2p_energy_trading.environment.env import P2PEnergyTradingEnv

    if _global_registry and _global_registry.contains(ENV_CREATOR, ENV_NAME):
        logger.debug("Environment '%s' is already registered with RLlib.", ENV_NAME)
        return

    logger.info("Registering custom environment '%s' with RLlib...", ENV_NAME)
    register_env(ENV_NAME, lambda cfg: P2PEnergyTradingEnv(cfg))
