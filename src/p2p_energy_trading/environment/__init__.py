"""Environment Package — Module 6.

Sub-packages and classes implementing the custom RLlib MultiAgentEnv.
Design reference: docs/module_6_multiagent_environment.md
"""

from __future__ import annotations

# third party
from gymnasium.envs.registration import register

# local
from p2p_energy_trading.constants import ENV_NAME
from p2p_energy_trading.environment.env import P2PEnergyTradingEnv

# Register the environment with the Gymnasium registry
register(
    id=ENV_NAME,
    entry_point="p2p_energy_trading.environment.env:P2PEnergyTradingEnv",
)

__all__ = ["P2PEnergyTradingEnv"]
