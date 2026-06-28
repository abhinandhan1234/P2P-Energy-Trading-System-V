"""Reward System — Module 5.

Multi-component per-agent reward computation for MAPPO training.

Provides stateless, pure-function reward components for the economic,
market-participation, grid-safety, and battery sub-objectives.

The primary entry-point for Module 6 is ``compute_all_rewards``.
Individual component functions are exported for testing and diagnostics.

Reference: docs/module_5_reward_system.md
"""

from __future__ import annotations

# local
from p2p_energy_trading.modules.reward.aggregator import (
    compute_agent_reward,
    compute_all_rewards,
)
from p2p_energy_trading.modules.reward.battery import (
    compute_cycling_penalty,
    compute_soc_penalty,
    compute_storage_reward,
)
from p2p_energy_trading.modules.reward.economic import compute_economic_reward
from p2p_energy_trading.modules.reward.grid_safety import (
    compute_thermal_penalty,
    compute_transformer_penalty,
    compute_voltage_penalty,
)
from p2p_energy_trading.modules.reward.trading import (
    compute_import_reduction_reward,
    compute_p2p_reward,
    compute_self_consumption_reward,
)

__all__ = [
    # Aggregator (primary public API)
    "compute_agent_reward",
    "compute_all_rewards",
    # Economic
    "compute_economic_reward",
    # Trading
    "compute_p2p_reward",
    "compute_self_consumption_reward",
    "compute_import_reduction_reward",
    # Grid safety
    "compute_voltage_penalty",
    "compute_thermal_penalty",
    "compute_transformer_penalty",
    # Battery
    "compute_soc_penalty",
    "compute_cycling_penalty",
    "compute_storage_reward",
]
