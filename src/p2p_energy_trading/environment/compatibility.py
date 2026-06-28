"""Compatibility layer for RLlib and Gymnasium.

Provides fallback for Ray/RLlib MultiAgentEnv if Ray is not installed,
allowing local testing without Ray dependencies.
"""

from __future__ import annotations

# standard library
import logging

# third party
import gymnasium

logger = logging.getLogger(__name__)

try:
    # third party
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ImportError:
    logger.warning(
        "Ray/RLlib is not installed in this Python environment. "
        "Falling back to a Gymnasium-based MultiAgentEnv stub for testing compatibility."
    )

    class MultiAgentEnv(gymnasium.Env):  # type: ignore
        """Mock stub representing Ray's MultiAgentEnv for testing when Ray is absent."""

        def __init__(self) -> None:
            super().__init__()
            # Expose agent fields based on what's defined on the subclass or defaults
            agent_ids = list(getattr(self, "_agent_ids", []))
            self.possible_agents: list[str] = getattr(
                self, "possible_agents", agent_ids
            )
            self.agents: list[str] = getattr(self, "agents", agent_ids)
            self.observation_space: gymnasium.spaces.Space | None = getattr(
                self, "observation_space", None
            )
            self.action_space: gymnasium.spaces.Space | None = getattr(
                self, "action_space", None
            )

            self.observation_spaces: dict[str, gymnasium.spaces.Space] = getattr(
                self, "observation_spaces", {}
            )
            self.action_spaces: dict[str, gymnasium.spaces.Space] = getattr(
                self, "action_spaces", {}
            )
