"""Baseline policies for the P2P Energy Trading evaluation framework.

Defines a unified BaselinePolicy interface, and concrete implementations for
Grid-Only, Random, and Rule-Based Heuristic control.

Design reference: docs/module_9_evaluation_framework.md §3
"""

from __future__ import annotations

# standard library
from abc import ABC, abstractmethod
from typing import Any

# third party
import numpy as np

# local
from p2p_energy_trading.constants import COLLEGE_AGENT_ID


class BaselinePolicy(ABC):
    """Abstract base class for all evaluation baseline policies."""

    @abstractmethod
    def compute_actions(
        self, obs: dict[str, Any] | np.ndarray, agent_id: str
    ) -> np.ndarray:
        """Compute actions for the given agent.

        Args:
            obs: The agent observation (either a Dict containing 'obs' or the raw array).
            agent_id: The ID string of the agent.

        Returns:
            A 3-dimensional numpy array [buy_fraction, sell_fraction, battery_dispatch].
        """
        pass


class GridOnlyBaseline(BaselinePolicy):
    """Grid-Only Baseline Policy.

    All transactions occur strictly through the utility grid. P2P trading fractions
    are fixed to 0, and the battery remains idle.
    """

    def compute_actions(
        self, obs: dict[str, Any] | np.ndarray, agent_id: str
    ) -> np.ndarray:
        """Compute grid-only actions.

        Returns:
            Fixed actions array [0.0, 0.0, 0.5].
        """
        return np.array([0.0, 0.0, 0.5], dtype=np.float32)


class RandomBaseline(BaselinePolicy):
    """Random Baseline Policy.

    Actions are sampled uniformly from Uniform(0, 1). To comply with physical constraints
    and prevent crashes, the battery discharging action is self-corrected to not exceed
    the current local demand.
    """

    def __init__(self, seed: int | None = None, peak_demand: float = 361.0) -> None:
        """Initialize RandomBaseline.

        Args:
            seed: Optional seed for the random number generator.
            peak_demand: The peak demand value used to unnormalize observations.
        """
        self.rng = np.random.default_rng(seed)
        self.peak_demand = peak_demand

    def compute_actions(
        self, obs: dict[str, Any] | np.ndarray, agent_id: str
    ) -> np.ndarray:
        """Compute random actions, self-correcting battery discharge to avoid crashes.

        Returns:
            A random actions array of shape (3,).
        """
        actions = self.rng.random(3, dtype=np.float32)

        # Self-correction: Battery discharge must not exceed college demand
        if agent_id == COLLEGE_AGENT_ID and actions[2] < 0.5:
            if isinstance(obs, dict):
                obs_vector = obs["obs"]
            else:
                obs_vector = obs

            # Extract normalized demand (index 1) and calculate unnormalized demand
            normalised_demand = float(obs_vector[1])
            demand_kw = normalised_demand * self.peak_demand

            # Map random discharge action to power, then clip to demand
            desired_power_kw = (0.5 - float(actions[2])) * 500.0
            desired_power_kw = min(desired_power_kw, demand_kw)

            # Convert back to scaled action
            actions[2] = np.clip(0.5 - desired_power_kw / 500.0, 0.0, 0.5)

        return actions


class HeuristicBaseline(BaselinePolicy):
    """Rule-Based Heuristic Baseline Policy.

    Implements hand-crafted domain rules:
    - College: Buy full deficit, sell full surplus P2P. Charge battery during solar
      peak and discharge battery during evening load peaks (capped by actual demand).
    - Solar: Buy full deficit, sell full surplus P2P. Battery is idle.
    - Consumer: Buy full deficit P2P. Battery is idle.
    """

    def __init__(self, peak_demand: float = 361.0) -> None:
        """Initialize HeuristicBaseline.

        Args:
            peak_demand: The peak demand value used to unnormalize observations.
        """
        self.peak_demand = peak_demand

    def compute_actions(
        self, obs: dict[str, Any] | np.ndarray, agent_id: str
    ) -> np.ndarray:
        """Compute rule-based heuristic actions from normalized observation features.

        Observation features mapping (first 5 indices of local observation):
        - Index 2: State of Charge (SoC).
        - Index 3: Normalized solar surplus.
        - Index 4: Normalized demand deficit.

        Returns:
            Computed actions array [buy_fraction, sell_fraction, battery_dispatch].
        """
        if isinstance(obs, dict):
            obs_vector = obs.get("obs")
            if obs_vector is None:
                raise ValueError("Observation dictionary is missing the 'obs' key.")
        else:
            obs_vector = obs

        # Extract features
        soc = float(obs_vector[2])
        surplus_norm = float(obs_vector[3])
        deficit_norm = float(obs_vector[4])

        buy_fraction = 1.0 if deficit_norm > 0 else 0.0
        sell_fraction = 1.0 if surplus_norm > 0 else 0.0
        battery_dispatch = 0.5

        if agent_id == COLLEGE_AGENT_ID:
            # Simple solar-peak-charge / demand-peak-discharge rule
            if surplus_norm > 0 and soc < 0.90:
                battery_dispatch = 1.0  # Charge
            elif deficit_norm > 0 and soc > 0.15:
                # Retrieve normalized demand (index 1) and calculate unnormalized demand
                normalised_demand = float(obs_vector[1])
                demand_kw = normalised_demand * self.peak_demand

                # Cap battery discharge to actual demand
                desired_power_kw = min(250.0, demand_kw)
                battery_dispatch = np.clip(0.5 - desired_power_kw / 500.0, 0.0, 0.5)
            else:
                battery_dispatch = 0.5  # Idle
        else:
            # Solar and Consumer agents do not have batteries (battery idle)
            battery_dispatch = 0.5
            if agent_id.startswith("consumer_"):
                sell_fraction = 0.0

        return np.array(
            [buy_fraction, sell_fraction, battery_dispatch], dtype=np.float32
        )
