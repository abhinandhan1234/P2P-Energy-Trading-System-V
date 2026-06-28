"""Observation Builder — Module 4.

Assembles 23-dim actor observations and 243-dim centralised critic state.

Reference: docs/module_4_observation_builder.md
"""

from __future__ import annotations

# local
from p2p_energy_trading.modules.observation.builder import build_observations

__all__ = [
    "build_observations",
]
