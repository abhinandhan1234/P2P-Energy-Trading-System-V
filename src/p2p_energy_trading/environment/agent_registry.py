"""Agent Registry for the P2P Energy Trading System.

Defines agent identifiers, metadata, and policy mappings.
Design reference: docs/module_6_multiagent_environment.md
"""

from __future__ import annotations

# standard library
from typing import Any

# local
from p2p_energy_trading.constants import (
    AGENT_TO_BUS,
    AGENT_TO_POLICY,
    ALL_AGENT_IDS,
    COLLEGE_AGENT_ID,
    CONSUMER_AGENT_IDS,
    SOLAR_AGENT_IDS,
)


class AgentRegistry:
    """Registry tracking agent mappings, types, and policy IDs.

    This class provides a lookup interface for agent metadata and policy mapping.
    """

    def __init__(self) -> None:
        """Initialize the static agent registry metadata mappings."""
        self._agents_meta: dict[str, dict[str, Any]] = {}
        self._policy_mapping = AGENT_TO_POLICY

        # Populate static agent metadata
        for aid in ALL_AGENT_IDS:
            is_college = aid == COLLEGE_AGENT_ID
            is_solar = aid in SOLAR_AGENT_IDS
            _ = aid in CONSUMER_AGENT_IDS

            if is_college:
                agent_type = "COLLEGE"
            elif is_solar:
                agent_type = "SOLAR"
            else:
                agent_type = "CONSUMER"

            self._agents_meta[aid] = {
                "agent_id": aid,
                "agent_type": agent_type,
                "bus_index": AGENT_TO_BUS[aid],
                "policy_id": self._policy_mapping[aid],
                "has_battery": is_college,
                "has_solar": is_college or is_solar,
            }

    def get_agent_metadata(self, agent_id: str) -> dict[str, Any]:
        """Retrieve static metadata dictionary for a specific agent.

        Args:
            agent_id: Unique string identifier of the agent.

        Returns:
            Dict containing type, bus, policy, and capability flags.

        Raises:
            KeyError: If agent_id is not registered.
        """
        if agent_id not in self._agents_meta:
            raise KeyError(
                f"Agent ID '{agent_id}' is not registered in the environment."
            )
        return self._agents_meta[agent_id]

    def policy_mapping_fn(self, agent_id: str) -> str:
        """Route an agent ID to its corresponding policy ID.

        Args:
            agent_id: Unique string identifier of the agent.

        Returns:
            Mapped policy ID string.

        Raises:
            KeyError: If agent_id is not registered.
        """
        if agent_id not in self._policy_mapping:
            raise KeyError(f"Agent ID '{agent_id}' has no registered policy mapping.")
        return self._policy_mapping[agent_id]

    def get_all_agent_ids(self) -> list[str]:
        """Return the list of all 21 registered agent IDs.

        Returns:
            List of agent ID strings.
        """
        return ALL_AGENT_IDS
