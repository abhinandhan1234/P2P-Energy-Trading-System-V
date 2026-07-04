"""Unit tests for the P2P Market Clearing Engine (Module 3).

Design reference: docs/module_3_market_engine.md
"""

from __future__ import annotations

# standard library
import math

# third party
import numpy as np
import pytest

# local
from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    BATTERY_CAPACITY_KWH,
    BATTERY_EFFICIENCY,
    BATTERY_SOC_MIN,
    COLLEGE_AGENT_ID,
)
from p2p_energy_trading.exceptions import MarketClearingError
from p2p_energy_trading.modules.market import (
    clear_market_p2p,
    process_settlements,
)


@pytest.fixture
def base_market_inputs() -> tuple[
    dict[str, float], dict[str, float], dict[str, np.ndarray]
]:
    """Fixture providing zeroed inputs for all 21 agents to pass validation.

    Returns:
        tuple containing:
        - dict[str, float]: Demands mapped by agent ID.
        - dict[str, float]: Solar mapped by agent ID.
        - dict[str, np.ndarray]: Actions mapped by agent ID.
    """
    demands = {aid: 0.0 for aid in ALL_AGENT_IDS}
    solar = {aid: 0.0 for aid in ALL_AGENT_IDS}
    actions = {
        aid: np.array([0.0, 0.0, 0.5], dtype=np.float32) for aid in ALL_AGENT_IDS
    }
    return demands, solar, actions


class TestMarketClearingP2P:
    """Verify core P2P market clearing logic and pro-rata allocations."""

    def test_supply_greater_than_demand_prorata(self):
        """When supply > demand, sellers are curtailed pro-rata while buyers
        clear fully."""
        bids = {"consumer_01": 10.0, "consumer_02": 10.0}
        offers = {"solar_01": 15.0, "solar_02": 25.0}

        p2p_bought, p2p_sold, volume, curtailed = clear_market_p2p(bids, offers)

        # Total demand = 20.0, Total supply = 40.0. Cleared volume = 20.0
        assert volume == pytest.approx(20.0)
        assert curtailed is True

        # Buyers clear fully
        assert p2p_bought["consumer_01"] == pytest.approx(10.0)
        assert p2p_bought["consumer_02"] == pytest.approx(10.0)

        # Sellers curtailed pro-rata: solar_01 gets 15/40 * 20 = 7.5,
        # solar_02 gets 25/40 * 20 = 12.5
        assert p2p_sold["solar_01"] == pytest.approx(7.5)
        assert p2p_sold["solar_02"] == pytest.approx(12.5)

    def test_demand_greater_than_supply_prorata(self):
        """When demand > supply, buyers are curtailed pro-rata while sellers
        clear fully."""
        bids = {"consumer_01": 30.0, "consumer_02": 10.0}
        offers = {"solar_01": 10.0, "solar_02": 10.0}

        p2p_bought, p2p_sold, volume, curtailed = clear_market_p2p(bids, offers)

        # Total demand = 40.0, Total supply = 20.0. Cleared volume = 20.0
        assert volume == pytest.approx(20.0)
        assert curtailed is True

        # Sellers clear fully
        assert p2p_sold["solar_01"] == pytest.approx(10.0)
        assert p2p_sold["solar_02"] == pytest.approx(10.0)

        # Buyers curtailed pro-rata: consumer_01 gets 30/40 * 20 = 15.0,
        # consumer_02 gets 10/40 * 20 = 5.0
        assert p2p_bought["consumer_01"] == pytest.approx(15.0)
        assert p2p_bought["consumer_02"] == pytest.approx(5.0)

    def test_equal_supply_demand(self):
        """When supply equals demand, both sides clear fully without curtailment."""
        bids = {"consumer_01": 10.0}
        offers = {"solar_01": 10.0}

        p2p_bought, p2p_sold, volume, curtailed = clear_market_p2p(bids, offers)

        assert volume == pytest.approx(10.0)
        assert curtailed is False
        assert p2p_bought["consumer_01"] == pytest.approx(10.0)
        assert p2p_sold["solar_01"] == pytest.approx(10.0)

    def test_zero_supply(self):
        """When supply is zero, cleared volume is zero and curtailment is not
        applied."""
        bids = {"consumer_01": 10.0}
        offers: dict[str, float] = {}

        p2p_bought, p2p_sold, volume, curtailed = clear_market_p2p(bids, offers)

        assert volume == 0.0
        assert curtailed is False
        assert p2p_bought.get("consumer_01", 0.0) == 0.0

    def test_zero_demand(self):
        """When demand is zero, cleared volume is zero and curtailment is not
        applied."""
        bids: dict[str, float] = {}
        offers = {"solar_01": 10.0}

        p2p_bought, p2p_sold, volume, curtailed = clear_market_p2p(bids, offers)

        assert volume == 0.0
        assert curtailed is False
        assert p2p_sold.get("solar_01", 0.0) == 0.0

    def test_empty_market(self):
        """When both supply and demand are zero, cleared volume is zero and
        curtailment is False."""
        bids: dict[str, float] = {}
        offers: dict[str, float] = {}

        p2p_bought, p2p_sold, volume, curtailed = clear_market_p2p(bids, offers)

        assert volume == 0.0
        assert curtailed is False


class TestMarketSettlement:
    """Verify end-to-end settlement processing, pricing, and grid fallbacks."""

    def test_midpoint_price_calculation(self, base_market_inputs):
        """Uniform price must be exactly the midpoint between buy and sell
        grid rates."""
        demands, solar, actions = base_market_inputs
        grid_buy = 8.0
        grid_sell = 4.0
        expected_price = (8.0 + 4.0) / 2.0  # 6.0

        records, state = process_settlements(
            demands, solar, actions, 0.5, grid_buy, grid_sell
        )

        assert state.p2p_clearing_price == pytest.approx(expected_price)
        for record in records.values():
            assert record.p2p_price == pytest.approx(expected_price)

    def test_utility_import_fallback(self, base_market_inputs):
        """Residual deficit after P2P clearing must be imported from the grid
        at grid_buy_rate."""
        demands, solar, actions = base_market_inputs

        # consumer_01 demand: 10 kW, bids 100% to P2P
        demands["consumer_01"] = 10.0
        actions["consumer_01"] = np.array([1.0, 0.0, 0.5], dtype=np.float32)

        # solar_01 has solar: 4 kW, offers 100% to P2P
        solar["solar_01"] = 4.0
        actions["solar_01"] = np.array([0.0, 1.0, 0.5], dtype=np.float32)

        grid_buy = 8.15
        grid_sell = 3.56
        p2p_price = (grid_buy + grid_sell) / 2.0  # 5.855

        # Clearing: total supply = 4 kW, total demand = 10 kW.
        # Cleared volume = 4 kW.
        # consumer_01 gets 4 kW P2P bought. Remaining 6 kW imported from grid.
        records, state = process_settlements(
            demands, solar, actions, 0.5, grid_buy, grid_sell
        )

        rec_c = records["consumer_01"]
        assert rec_c.p2p_bought_kw == pytest.approx(4.0)
        assert rec_c.grid_bought_kw == pytest.approx(6.0)
        assert rec_c.p2p_cost == pytest.approx(4.0 * p2p_price)
        assert rec_c.grid_cost == pytest.approx(6.0 * grid_buy)
        assert rec_c.net_cost == pytest.approx(4.0 * p2p_price + 6.0 * grid_buy)
        assert state.total_bids == pytest.approx(10.0)
        assert state.total_offers == pytest.approx(4.0)

    def test_utility_export_fallback(self, base_market_inputs):
        """Residual surplus after P2P clearing must be exported to the grid
        at grid_sell_rate."""
        demands, solar, actions = base_market_inputs

        # consumer_01 demand: 4 kW, bids 100% to P2P
        demands["consumer_01"] = 4.0
        actions["consumer_01"] = np.array([1.0, 0.0, 0.5], dtype=np.float32)

        # solar_01 has solar: 10 kW, offers 100% to P2P
        solar["solar_01"] = 10.0
        actions["solar_01"] = np.array([0.0, 1.0, 0.5], dtype=np.float32)

        grid_buy = 8.15
        grid_sell = 3.56
        p2p_price = (grid_buy + grid_sell) / 2.0  # 5.855

        # Clearing: total supply = 10 kW, total demand = 4 kW.
        # Cleared volume = 4 kW.
        # solar_01 gets 4 kW P2P sold. Remaining 6 kW exported to grid.
        records, state = process_settlements(
            demands, solar, actions, 0.5, grid_buy, grid_sell
        )

        rec_s = records["solar_01"]
        assert rec_s.p2p_sold_kw == pytest.approx(4.0)
        assert rec_s.grid_sold_kw == pytest.approx(6.0)
        assert rec_s.p2p_revenue == pytest.approx(4.0 * p2p_price)
        assert rec_s.grid_revenue == pytest.approx(6.0 * grid_sell)
        assert rec_s.net_cost == pytest.approx(-(4.0 * p2p_price + 6.0 * grid_sell))
        assert state.total_bids == pytest.approx(4.0)
        assert state.total_offers == pytest.approx(10.0)

    def test_college_battery_discharge_behaviour(self, base_market_inputs):
        """Verify that discharging the college battery reduces its P2P demand
        and matches constraints."""
        demands, solar, actions = base_market_inputs

        # College demand: 200 kW, battery discharging action: 0.1 (discharge)
        demands[COLLEGE_AGENT_ID] = 200.0
        actions[COLLEGE_AGENT_ID] = np.array([1.0, 0.0, 0.1], dtype=np.float32)

        # Desired dispatch power: (0.5 - 0.1) * 2 * 250 = 200 kW (discharge)
        # SOC = 0.5. Discharge energy limit: (0.5 - 0.1) * 500 = 200 kWh.
        # Max discharge power over dt=1.0: 200 * sqrt(0.9) = 189.7366 kW
        # actual power = min(200, 189.7366) = 189.7366 kW
        expected_dispatch = min(
            200.0,
            (0.5 - BATTERY_SOC_MIN)
            * BATTERY_CAPACITY_KWH
            * math.sqrt(BATTERY_EFFICIENCY),
        )

        records, state = process_settlements(demands, solar, actions, 0.5, 8.15, 3.56)

        # Net power deficit before market: 200.0 - 0.0 - 189.7366 = 10.2634 kW
        # Since buy_fraction = 1.0, college P2P bid should be 10.2634 kW
        rec = records[COLLEGE_AGENT_ID]
        assert rec.p2p_bought_kw + rec.grid_bought_kw == pytest.approx(
            200.0 - expected_dispatch
        )

    def test_college_battery_charge_behaviour(self, base_market_inputs):
        """Verify that charging the college battery increases its P2P demand
        and matches constraints."""
        demands, solar, actions = base_market_inputs

        # College demand: 0 kW, solar: 0 kW, battery charging action: 0.9 (charge)
        demands[COLLEGE_AGENT_ID] = 0.0
        actions[COLLEGE_AGENT_ID] = np.array([1.0, 0.0, 0.9], dtype=np.float32)

        # Desired dispatch power: (0.5 - 0.9) * 2 * 250 = -200 kW (charge)
        # SOC = 0.5. Charge energy limit: (0.95 - 0.5) * 500 = 225 kWh.
        # Max charge power over dt=1.0: 225 / sqrt(0.9) = 237.17 kW
        # actual power = max(-200, -237.17) = -200.0 kW (charging 200 kW)
        expected_dispatch = -200.0

        records, state = process_settlements(demands, solar, actions, 0.5, 8.15, 3.56)

        # Net power deficit before market: 0.0 - 0.0 - (-200) = 200.0 kW
        # Since buy_fraction = 1.0, college P2P bid should be 200.0 kW
        rec = records[COLLEGE_AGENT_ID]
        assert rec.p2p_bought_kw + rec.grid_bought_kw == pytest.approx(
            -expected_dispatch
        )

    def test_energy_balance_conservation(self, base_market_inputs):
        """Verify that process_settlements enforces energy balance and raises
        errors on violations."""
        demands, solar, actions = base_market_inputs

        # Basic balanced market
        demands["consumer_01"] = 10.0
        actions["consumer_01"] = np.array([1.0, 0.0, 0.5], dtype=np.float32)
        solar["solar_01"] = 10.0
        actions["solar_01"] = np.array([0.0, 1.0, 0.5], dtype=np.float32)

        # Should pass without error
        records, state = process_settlements(demands, solar, actions, 0.5, 8.15, 3.56)
        assert state.total_p2p_volume == pytest.approx(10.0)

    def test_constraint_flag_placeholders(self, base_market_inputs):
        """Verify that voltage_violation and thermal_violation are initialized
        as False."""
        demands, solar, actions = base_market_inputs
        records, state = process_settlements(demands, solar, actions, 0.5, 8.15, 3.56)
        assert state.voltage_violation is False
        assert state.thermal_violation is False

    def test_frozen_dataclass_immutability(self, base_market_inputs):
        """Verify SettlementRecord and MarketState are read-only (frozen)."""
        # standard library
        import dataclasses

        demands, solar, actions = base_market_inputs
        records, state = process_settlements(demands, solar, actions, 0.5, 8.15, 3.56)

        rec = records[COLLEGE_AGENT_ID]
        with pytest.raises(dataclasses.FrozenInstanceError):
            # Try to mutate frozen dataclass
            rec.p2p_sold_kw = 100.0  # type: ignore

        with pytest.raises(dataclasses.FrozenInstanceError):
            # Try to mutate frozen dataclass
            state.total_p2p_volume = 100.0  # type: ignore

    def test_market_clearing_error_validation(self, base_market_inputs):
        """Verify validation errors are raised for invalid inputs or action lengths."""
        demands, solar, actions = base_market_inputs

        # 1. Negative demand
        bad_demands = demands.copy()
        bad_demands[COLLEGE_AGENT_ID] = -10.0
        with pytest.raises(MarketClearingError, match="Negative demand or solar"):
            process_settlements(bad_demands, solar, actions, 0.5, 8.15, 3.56)

        # 2. Negative solar
        bad_solar = solar.copy()
        bad_solar[COLLEGE_AGENT_ID] = -10.0
        with pytest.raises(MarketClearingError, match="Negative demand or solar"):
            process_settlements(demands, bad_solar, actions, 0.5, 8.15, 3.56)

        # 3. Action length mismatch
        bad_actions = actions.copy()
        bad_actions[COLLEGE_AGENT_ID] = np.array([1.0, 0.0], dtype=np.float32)
        with pytest.raises(MarketClearingError, match="must have length 3"):
            process_settlements(demands, solar, bad_actions, 0.5, 8.15, 3.56)

        # 4. Missing action for agent
        missing_actions = actions.copy()
        del missing_actions[COLLEGE_AGENT_ID]
        with pytest.raises(MarketClearingError, match="Missing action vector"):
            process_settlements(demands, solar, missing_actions, 0.5, 8.15, 3.56)

        # 5. College discharging too much (causing negative solar_used_locally)
        # College demand is 10 kW. Battery discharges 100 kW.
        # Desired dispatch = 100 kW.
        # Since demand = 10 kW and discharge = 100 kW, solar_used = 10 - 100 = -90 kW.
        # This is < -0.01 and should trigger MarketClearingError.
        excess_discharge_actions = actions.copy()
        # Battery dispatch power is (0.5 - a2) * 2 * 250
        # If a2 = 0.3, desired power = 0.2 * 500 = 100 kW discharge
        excess_discharge_actions[COLLEGE_AGENT_ID] = np.array(
            [0.0, 1.0, 0.3], dtype=np.float32
        )
        college_demands = demands.copy()
        college_demands[COLLEGE_AGENT_ID] = 10.0

        with pytest.raises(
            MarketClearingError, match="Calculated local solar consumption is negative"
        ):
            process_settlements(
                college_demands, solar, excess_discharge_actions, 0.5, 8.15, 3.56
            )
