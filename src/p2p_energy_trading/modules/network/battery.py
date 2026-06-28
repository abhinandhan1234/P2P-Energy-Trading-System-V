"""Stateful battery storage model for the College building.

This module simulates the 500 kWh battery storage system at Bus 7 (College),
enforcing physical power ratings, SoC limits, round-trip efficiency losses,
and minimum dispatch thresholds.

Design reference: docs/module_2_pandapower_network.md
"""

from __future__ import annotations

# standard library
import logging
import math

# local
from p2p_energy_trading.constants import (
    BATTERY_CAPACITY_KWH,
    BATTERY_EFFICIENCY,
    BATTERY_INITIAL_SOC_EVAL,
    BATTERY_MIN_DISPATCH_KW,
    BATTERY_POWER_KW,
    BATTERY_SOC_MAX,
    BATTERY_SOC_MIN,
)

logger = logging.getLogger(__name__)


class BatteryModel:
    """Stateful battery model for the college building (Bus 7).

    Capacity: 500 kWh | Power: 250 kW | Efficiency: 90% (round-trip)
    SoC limits: [10%, 95%]
    """

    def __init__(self, initial_soc: float = BATTERY_INITIAL_SOC_EVAL) -> None:
        """Initialize the battery model with a starting State of Charge.

        Args:
            initial_soc: Initial State of Charge as a fraction of capacity (0.0 to 1.0).
        """
        self._capacity_kwh = BATTERY_CAPACITY_KWH
        self._power_kw = BATTERY_POWER_KW
        self._efficiency = BATTERY_EFFICIENCY
        self._soc_min = BATTERY_SOC_MIN
        self._soc_max = BATTERY_SOC_MAX

        # Symmetrical efficiency for charge and discharge: eta = sqrt(0.90)
        self._eta_charge = math.sqrt(self._efficiency)
        self._eta_discharge = math.sqrt(self._efficiency)

        self._soc = max(self._soc_min, min(self._soc_max, float(initial_soc)))
        self._last_dispatch_kw = 0.0
        self._prev_dispatch_kw = 0.0

    @property
    def soc(self) -> float:
        """Current State of Charge (SoC) in [0.0, 1.0]."""
        return self._soc

    @property
    def last_dispatch_kw(self) -> float:
        """Last dispatch power in kW (positive = discharging, negative = charging)."""
        return self._last_dispatch_kw

    @property
    def prev_dispatch_kw(self) -> float:
        """Previous dispatch power in kW (positive = discharging, negative = charging)."""
        return self._prev_dispatch_kw

    def predict_dispatch(
        self,
        action_charge_fraction: float,
        dt: float = 1.0,
    ) -> float:
        """Predict the battery dispatch power for a given action and current SoC.

        This method is a pure function that does not mutate the battery State of Charge
        or any other state.

        Args:
            action_charge_fraction: Agent's battery action in [0, 1].
            dt: Timestep duration in hours (default 1.0).

        Returns:
            Predicted dispatch power in kW (positive = discharging, negative = charging).
        """
        # 1. Convert action to desired power: action=0.5 -> 0, action=0 -> 250 (discharge), action=1 -> -250 (charge)
        desired_power_kw = (0.5 - action_charge_fraction) * 2.0 * self._power_kw

        # 2. Clip by power rating (±250 kW)
        desired_power_kw = max(-self._power_kw, min(self._power_kw, desired_power_kw))

        # 3. Clip by SoC-energy constraints
        if desired_power_kw < 0:  # Charging (absorbing from grid)
            charge_power_demand = -desired_power_kw
            # Remaining capacity in kWh that can be filled
            kwh_to_fill = (self._soc_max - self._soc) * self._capacity_kwh
            # Available power we can draw given efficiency and timestep
            max_charge_power = kwh_to_fill / (self._eta_charge * dt)
            actual_charge_power = min(charge_power_demand, max_charge_power)
            actual_power = -actual_charge_power

        elif desired_power_kw > 0:  # Discharging (injecting into grid)
            discharge_power_demand = desired_power_kw
            # Available energy in kWh we can discharge
            kwh_to_drain = (self._soc - self._soc_min) * self._capacity_kwh
            # Max power output to grid considering efficiency and timestep
            max_discharge_power = kwh_to_drain * self._eta_discharge / dt
            actual_discharge_power = min(discharge_power_demand, max_discharge_power)
            actual_power = actual_discharge_power

        else:
            actual_power = 0.0

        # 4. Apply minimum dispatch threshold (25 kW) to prevent micro-cycling
        if abs(actual_power) < BATTERY_MIN_DISPATCH_KW:
            actual_power = 0.0

        return actual_power

    def step(self, action_charge_fraction: float, dt: float = 1.0) -> float:
        """Execute one battery dispatch step.

        Args:
            action_charge_fraction: Agent's battery action in [0, 1].
                0.0 = full discharge at 250 kW (inject to bus)
                0.5 = idle (no dispatch)
                1.0 = full charge at 250 kW (absorb from bus)
            dt: Timestep duration in hours (default 1.0).

        Returns:
            Actual dispatch power in kW.
                Positive = discharging (injecting to bus).
                Negative = charging (absorbing from bus).
        """
        # Save previous dispatch state for cycling detection
        self._prev_dispatch_kw = self._last_dispatch_kw

        # Calculate actual power using predict_dispatch helper
        actual_power = self.predict_dispatch(action_charge_fraction, dt)

        # 5. Update State of Charge (SoC) based on actual power
        if actual_power < 0:  # Charging
            self._soc += (-actual_power * self._eta_charge * dt) / self._capacity_kwh
        elif actual_power > 0:  # Discharging
            self._soc -= (
                actual_power / (self._eta_discharge * dt)
            ) / self._capacity_kwh

        # Ensure floating-point clipping within safe bounds
        self._soc = max(self._soc_min, min(self._soc_max, self._soc))
        self._last_dispatch_kw = actual_power

        return actual_power

    def reset(self, initial_soc: float | None = None) -> None:
        """Reset the State of Charge for the start of an episode.

        Args:
            initial_soc: Starting SoC fraction. If None, defaults to BATTERY_INITIAL_SOC_EVAL.
        """
        if initial_soc is not None:
            self._soc = float(initial_soc)
        else:
            self._soc = BATTERY_INITIAL_SOC_EVAL

        self._soc = max(self._soc_min, min(self._soc_max, self._soc))
        self._last_dispatch_kw = 0.0
        self._prev_dispatch_kw = 0.0
        logger.info(f"Battery reset to SoC = {self._soc:.3f}")

    def get_state(self) -> dict[str, float]:
        """Return the current battery state for observations.

        Returns:
            Dict containing 'soc', 'dispatch_kw', 'at_min', and 'at_max'.
        """
        # Float flags for neural network observations (1.0 = True, 0.0 = False)
        # Tolerance margin to account for floating point comparisons
        tol = 1e-6
        at_min = 1.0 if (self._soc - self._soc_min) <= tol else 0.0
        at_max = 1.0 if (self._soc_max - self._soc) <= tol else 0.0

        return {
            "soc": self._soc,
            "dispatch_kw": self._last_dispatch_kw,
            "at_min": at_min,
            "at_max": at_max,
        }
