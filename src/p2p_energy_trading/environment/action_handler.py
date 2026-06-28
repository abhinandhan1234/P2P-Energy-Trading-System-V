"""Action Handler for the P2P Energy Trading Environment.

Handles completeness checks, NaN/Inf detection, clipping, and default actions injection.
Design reference: docs/module_6_multiagent_environment.md
"""

from __future__ import annotations

# standard library
import logging

# third party
import numpy as np

# local
from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    DEFAULT_SAFE_ACTION,
    ActionDict,
)

logger = logging.getLogger(__name__)


class ActionHandler:
    """Action validator and preprocessor for agent actions."""

    def __init__(self) -> None:
        """Initialize the ActionHandler."""
        self._default_action = np.asarray(DEFAULT_SAFE_ACTION, dtype=np.float32)

    def validate_and_scale(
        self,
        action_dict: ActionDict,
    ) -> tuple[ActionDict, int]:
        """Validate and pre-clip raw action fractions to [0.0, 1.0].

        Performs:
        1. Inject default actions for missing agents.
        2. Replace NaN action vectors with default safe action and log warnings.
        3. Replace Inf action vectors or values with clipped bounds.
        4. Clip raw values to [0.0, 1.0] range.

        Args:
            action_dict: Dict mapping agent ID to 3-dim raw action vectors.

        Returns:
            Tuple containing:
            - ActionDict: Cleaned and pre-clipped action vectors.
            - int: Count of NaN action vectors replaced in this step.
        """
        cleaned_actions: ActionDict = {}
        nan_count = 0

        for aid in ALL_AGENT_IDS:
            if aid not in action_dict:
                # Missing agent -> Inject default safe action
                logger.warning(
                    "Missing action vector for agent '%s' at current timestep. "
                    "Injecting default safe action %s.",
                    aid,
                    DEFAULT_SAFE_ACTION,
                )
                cleaned_actions[aid] = self._default_action.copy()
                continue

            raw_action = action_dict[aid]

            # Enforce numeric array type
            if not isinstance(raw_action, np.ndarray):
                try:
                    raw_action = np.asarray(raw_action)
                except (TypeError, ValueError):
                    logger.error(
                        "Action vector for agent '%s' could not be converted to numpy array. "
                        "Forcing safe default values.",
                        aid,
                    )
                    cleaned_actions[aid] = self._default_action.copy()
                    continue

            if not np.issubdtype(raw_action.dtype, np.number):
                logger.error(
                    "Action vector for agent '%s' has invalid non-numeric dtype %s. "
                    "Forcing safe default values.",
                    aid,
                    raw_action.dtype,
                )
                cleaned_actions[aid] = self._default_action.copy()
                continue

            raw_action = raw_action.astype(np.float32)
            if raw_action.shape != (3,):
                logger.error(
                    "Action vector for agent '%s' has invalid shape %s. "
                    "Forcing shape (3,) with safe default values.",
                    aid,
                    raw_action.shape,
                )
                cleaned_actions[aid] = self._default_action.copy()
                continue

            # NaN detection
            if np.isnan(raw_action).any():
                logger.warning(
                    "NaN values detected in action vector for agent '%s'. "
                    "Replacing with safe default action %s.",
                    aid,
                    DEFAULT_SAFE_ACTION,
                )
                cleaned_actions[aid] = self._default_action.copy()
                nan_count += 1
                continue

            # Inf detection & replacement
            if np.isinf(raw_action).any():
                logger.warning(
                    "Infinite values detected in action vector for agent '%s'. "
                    "Clipping values to boundary.",
                    aid,
                )
                raw_action = np.clip(raw_action, 0.0, 1.0)

            # Pre-clipping actions to valid [0.0, 1.0] bounds
            cleaned_actions[aid] = np.clip(raw_action, 0.0, 1.0).astype(np.float32)

        return cleaned_actions, nan_count
