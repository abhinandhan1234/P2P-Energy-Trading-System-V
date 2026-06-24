"""Unit tests for the stateful battery model (Module 2).

Design reference: docs/module_2_pandapower_network.md
"""

from __future__ import annotations

import math
import pytest

from p2p_energy_trading.constants import (
    BATTERY_CAPACITY_KWH,
    BATTERY_POWER_KW,
    BATTERY_SOC_MIN,
    BATTERY_SOC_MAX,
    BATTERY_INITIAL_SOC_EVAL,
    BATTERY_MIN_DISPATCH_KW,
)
from p2p_energy_trading.modules.network.battery import BatteryModel


class TestBatteryInitialState:
    """Verify that battery model initializes correctly."""

    def test_default_init(self):
        """Default initialization should use evaluation default SoC."""
        battery = BatteryModel()
        assert battery.soc == pytest.approx(BATTERY_INITIAL_SOC_EVAL)
        assert battery.last_dispatch_kw == 0.0
        assert battery.prev_dispatch_kw == 0.0

    def test_custom_init(self):
        """Custom initial SoC should set the current SoC."""
        battery = BatteryModel(initial_soc=0.45)
        assert battery.soc == pytest.approx(0.45)

    def test_init_clipping(self):
        """Initial SoC outside limits should be clipped."""
        battery_low = BatteryModel(initial_soc=0.05)
        assert battery_low.soc == pytest.approx(BATTERY_SOC_MIN)

        battery_high = BatteryModel(initial_soc=0.99)
        assert battery_high.soc == pytest.approx(BATTERY_SOC_MAX)


class TestBatteryDischarge:
    """Test battery discharging behavior (action < 0.5)."""

    def test_nominal_discharge(self):
        """Action < 0.5 should result in positive dispatch power and decrease SoC."""
        battery = BatteryModel(initial_soc=0.80)
        action = 0.2  # 0.2 is discharge
        # Desired: (0.5 - 0.2) * 2 * 250 = 150 kW
        dispatch = battery.step(action, dt=1.0)
        assert dispatch == pytest.approx(150.0)
        assert battery.soc < 0.80
        # Expected SoC change: 150 / (sqrt(0.90) * 500)
        expected_soc_loss = 150.0 / (math.sqrt(0.90) * BATTERY_CAPACITY_KWH)
        assert battery.soc == pytest.approx(0.80 - expected_soc_loss)


class TestBatteryCharge:
    """Test battery charging behavior (action > 0.5)."""

    def test_nominal_charge(self):
        """Action > 0.5 should result in negative dispatch power and increase SoC."""
        battery = BatteryModel(initial_soc=0.30)
        action = 0.8  # 0.8 is charge
        # Desired: (0.5 - 0.8) * 2 * 250 = -150 kW
        dispatch = battery.step(action, dt=1.0)
        assert dispatch == pytest.approx(-150.0)
        assert battery.soc > 0.30
        # Expected SoC change: (150 * sqrt(0.90)) / 500
        expected_soc_gain = (150.0 * math.sqrt(0.90)) / BATTERY_CAPACITY_KWH
        assert battery.soc == pytest.approx(0.30 + expected_soc_gain)


class TestBatteryIdle:
    """Test battery idle behavior (action == 0.5)."""

    def test_idle(self):
        """Action = 0.5 should result in 0 dispatch and no SoC change."""
        battery = BatteryModel(initial_soc=0.50)
        dispatch = battery.step(0.5, dt=1.0)
        assert dispatch == 0.0
        assert battery.soc == pytest.approx(0.50)


class TestBatterySoCLimits:
    """Verify that battery cannot exceed SoC limits."""

    def test_discharge_limit(self):
        """Battery must not discharge below BATTERY_SOC_MIN."""
        # Start at 0.16 (only 0.06 * 500 = 30 kWh available to discharge)
        battery = BatteryModel(initial_soc=0.16)
        # Request full discharge (250 kW)
        dispatch = battery.step(0.0, dt=1.0)
        # Max discharge possible: 30 kWh * eta_discharge / 1.0 h = 28.46 kW (above 25 kW min dispatch)
        max_possible = 0.06 * BATTERY_CAPACITY_KWH * math.sqrt(0.90)
        assert dispatch == pytest.approx(max_possible)
        assert battery.soc == pytest.approx(BATTERY_SOC_MIN)

        # A subsequent discharge request when at min should yield 0 dispatch
        dispatch_blocked = battery.step(0.0, dt=1.0)
        assert dispatch_blocked == 0.0
        assert battery.soc == pytest.approx(BATTERY_SOC_MIN)

    def test_charge_limit(self):
        """Battery must not charge above BATTERY_SOC_MAX."""
        # Start at 0.89 (only 0.06 * 500 = 30 kWh capacity left)
        battery = BatteryModel(initial_soc=0.89)
        # Request full charge (-250 kW)
        dispatch = battery.step(1.0, dt=1.0)
        # Max charge possible: 30 kWh / (eta_charge * 1.0 h) = -31.62 kW (above 25 kW min dispatch)
        max_possible_charge = 0.06 * BATTERY_CAPACITY_KWH / math.sqrt(0.90)
        assert dispatch == pytest.approx(-max_possible_charge)
        assert battery.soc == pytest.approx(BATTERY_SOC_MAX)

        # A subsequent charge request when at max should yield 0 dispatch
        dispatch_blocked = battery.step(1.0, dt=1.0)
        assert dispatch_blocked == 0.0
        assert battery.soc == pytest.approx(BATTERY_SOC_MAX)


class TestBatteryPowerLimits:
    """Verify dispatch is clipped to power ratings."""

    def test_power_clipping(self):
        """Desired power outside ±250 kW should be clipped."""
        # Start at 0.80 for discharge clipping (sufficient headroom to discharge 250 kW)
        battery_discharge = BatteryModel(initial_soc=0.80)
        dispatch_low = battery_discharge.step(-10.0, dt=1.0)  # Clipped to full discharge
        assert dispatch_low == pytest.approx(BATTERY_POWER_KW)

        # Start at 0.30 for charge clipping (sufficient headroom to charge 250 kW)
        battery_charge = BatteryModel(initial_soc=0.30)
        dispatch_high = battery_charge.step(10.0, dt=1.0)  # Clipped to full charge
        assert dispatch_high == pytest.approx(-BATTERY_POWER_KW)


class TestBatteryMinimumDispatch:
    """Verify that dispatch values below the threshold round to zero."""

    def test_below_threshold(self):
        """Dispatch below BATTERY_MIN_DISPATCH_KW (25 kW) should be rounded to 0."""
        battery = BatteryModel(initial_soc=0.50)
        # Action that requests small power: e.g. desired_power = 10 kW
        # action = 0.5 - 10 / (2 * 250) = 0.48
        dispatch = battery.step(0.48, dt=1.0)
        assert dispatch == 0.0
        assert battery.soc == pytest.approx(0.50)

    def test_above_threshold(self):
        """Dispatch equal to or above 25 kW should not be rounded to 0."""
        battery = BatteryModel(initial_soc=0.50)
        # Action that requests 30 kW:
        # action = 0.5 - 30 / (2 * 250) = 0.44
        dispatch = battery.step(0.44, dt=1.0)
        assert dispatch == pytest.approx(30.0)
        assert battery.soc < 0.50


class TestBatteryReset:
    """Verify that reset restores starting state."""

    def test_reset_behavior(self):
        battery = BatteryModel(initial_soc=0.80)
        battery.step(0.0)  # Change state
        assert battery.soc != 0.80

        battery.reset()
        assert battery.soc == pytest.approx(BATTERY_INITIAL_SOC_EVAL)
        assert battery.last_dispatch_kw == 0.0

        battery.reset(initial_soc=0.35)
        assert battery.soc == pytest.approx(0.35)


class TestBatteryGetState:
    """Verify state dictionary representation."""

    def test_get_state_keys(self):
        battery = BatteryModel(initial_soc=0.50)
        state = battery.get_state()
        assert "soc" in state
        assert "dispatch_kw" in state
        assert "at_min" in state
        assert "at_max" in state

        assert state["soc"] == pytest.approx(0.50)
        assert state["dispatch_kw"] == 0.0
        assert state["at_min"] == 0.0
        assert state["at_max"] == 0.0

    def test_get_state_boundary_flags(self):
        # Test min boundary flag
        battery_min = BatteryModel(initial_soc=BATTERY_SOC_MIN)
        assert battery_min.get_state()["at_min"] == 1.0
        assert battery_min.get_state()["at_max"] == 0.0

        # Test max boundary flag
        battery_max = BatteryModel(initial_soc=BATTERY_SOC_MAX)
        assert battery_max.get_state()["at_min"] == 0.0
        assert battery_max.get_state()["at_max"] == 1.0


class TestBatteryCycling:
    """Verify that previous and current dispatch power are tracked for cycling detection."""

    def test_direction_switch_tracking(self):
        battery = BatteryModel(initial_soc=0.50)
        # Step 1: Charge (action > 0.5)
        dispatch_1 = battery.step(0.8)
        assert dispatch_1 < 0
        assert battery.last_dispatch_kw == dispatch_1
        assert battery.prev_dispatch_kw == 0.0

        # Step 2: Discharge (action < 0.5)
        dispatch_2 = battery.step(0.2)
        assert dispatch_2 > 0
        assert battery.last_dispatch_kw == dispatch_2
        assert battery.prev_dispatch_kw == dispatch_1
