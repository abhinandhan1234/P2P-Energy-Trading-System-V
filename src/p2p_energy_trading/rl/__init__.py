"""RLlib MAPPO Integration package for the P2P Energy Trading System.

This package exposes the public API stability components required for RLlib integration.
Design reference: docs/module_7_mappo_integration.md
"""

from __future__ import annotations

# local
from p2p_energy_trading.rl.callbacks import P2PCallbacks
from p2p_energy_trading.rl.centralized_critic import CentralizedCriticRLModule
from p2p_energy_trading.rl.policy_config import build_ppo_config

__all__ = [
    "CentralizedCriticRLModule",
    "build_ppo_config",
    "P2PCallbacks",
]
