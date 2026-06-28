"""Unit tests for the Observation Builder (Module 4).

Design reference: docs/module_4_observation_builder.md
"""

from __future__ import annotations

# standard library
import math
from typing import Any

# third party
import numpy as np
import pandas as pd
import pytest

# local
from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    COLLEGE_AGENT_ID,
    CONSUMER_AGENT_IDS,
    MAX_GRID_RATE,
)
from p2p_energy_trading.modules.market.models import MarketState
from p2p_energy_trading.modules.network.powerflow import PowerFlowResult
from p2p_energy_trading.modules.observation.builder import build_observations
from p2p_energy_trading.modules.observation.normalisation import (
    normalise_grid_price,
    normalise_loading,
)


@pytest.fixture
def sample_metadata() -> dict[str, Any]:
    """Fixture providing peak demand/solar capacities for portfolio buildings."""
    buildings = {}
    for aid in ALL_AGENT_IDS:
        if aid == COLLEGE_AGENT_ID:
            b_type = "college"
            peak_d = 150.0
            peak_s = 100.0
        elif aid.startswith("solar_"):
            b_type = "solar"
            peak_d = 50.0
            peak_s = 80.0
        else:
            b_type = "consumer"
            peak_d = 40.0
            peak_s = 0.0
        buildings[aid] = {
            "building_id": aid,
            "building_type": b_type,
            "profile_stats": {
                "demand_kw": {"peak": peak_d},
                "solar_generation_kw": {"peak": peak_s},
            },
        }
    return {"buildings": buildings}


@pytest.fixture
def sample_inputs(
    sample_metadata,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, np.ndarray],
    PowerFlowResult,
    MarketState,
    pd.Timestamp,
    float,
    float,
]:
    """Fixture providing valid input arguments for build_observations."""
    demands = {aid: 10.0 for aid in ALL_AGENT_IDS}
    solar = {aid: 5.0 for aid in ALL_AGENT_IDS}
    solar[COLLEGE_AGENT_ID] = 8.0
    for aid in CONSUMER_AGENT_IDS:
        solar[aid] = 0.0

    battery = {"soc": 0.5, "dispatch_kw": 0.0}
    last_actions = {
        aid: np.array([0.1, 0.2, 0.5], dtype=np.float32) for aid in ALL_AGENT_IDS
    }
    grid_result = PowerFlowResult(
        converged=True,
        bus_vm_pu={i: 1.01 for i in range(33)},
        line_loading_pct={i: 20.0 for i in range(32)},
        trafo_loading_pct={0: 45.0},
        p_grid_kw=500.0,
    )
    market_state = MarketState(
        p2p_clearing_price=5.855,
        total_p2p_volume=100.0,
        p2p_utilisation_ratio=1.0,
        grid_import_total=500.0,
        grid_export_total=0.0,
        voltage_violation=False,
        thermal_violation=False,
        curtailment_applied=False,
        total_bids=120.0,
        total_offers=100.0,
    )
    timestamp = pd.Timestamp("2026-06-24 14:00:00")
    grid_buy = 8.15
    grid_sell = 3.56
    return (
        demands,
        solar,
        battery,
        last_actions,
        grid_result,
        market_state,
        timestamp,
        grid_buy,
        grid_sell,
    )


class TestObservationBuilder:
    """Verify that build_observations behaves exactly according to the spec."""

    def test_shapes_and_structure(self, sample_inputs, sample_metadata):
        """Verify actor shape is (23,), critic state shape is (243,), and RLlib Dict structure is correct."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        assert len(obs_dict) == 21
        for aid in ALL_AGENT_IDS:
            assert aid in obs_dict
            assert "obs" in obs_dict[aid]
            assert "state" in obs_dict[aid]
            assert obs_dict[aid]["obs"].shape == (23,)
            assert obs_dict[aid]["state"].shape == (243,)

    def test_float32_dtype_enforcement(self, sample_inputs, sample_metadata):
        """Verify that observation arrays strictly enforce dtype=np.float32."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        for aid in ALL_AGENT_IDS:
            assert obs_dict[aid]["obs"].dtype == np.float32
            assert obs_dict[aid]["state"].dtype == np.float32

    def test_agent_ordering_in_critic(self, sample_inputs, sample_metadata):
        """Verify that local observations of all 21 agents are concatenated in ALL_AGENT_IDS sequence."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        # Concatenated local features at the start of critic state
        critic_state = obs_dict[COLLEGE_AGENT_ID]["state"]
        for idx, aid in enumerate(ALL_AGENT_IDS):
            start = idx * 11
            end = start + 11
            agent_local_in_critic = critic_state[start:end]
            agent_actor_local = obs_dict[aid]["obs"][0:11]
            assert np.array_equal(agent_local_in_critic, agent_actor_local)

    def test_voltage_normalisation(self, sample_inputs, sample_metadata):
        """Verify voltage is normalized as (Vpu - 0.95) / 0.10 and clipped to [0, 1]."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        # Case 1: Voltage = 1.01 p.u. -> normalized to (1.01 - 0.95) / 0.1 = 0.6
        grid_result.bus_vm_pu = {i: 1.01 for i in range(33)}
        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )
        assert obs_dict[COLLEGE_AGENT_ID]["obs"][11] == pytest.approx(0.6)

        # Case 2: Out of bounds high -> 1.07 -> clipped to 1.0
        grid_result.bus_vm_pu = {i: 1.07 for i in range(33)}
        obs_dict2 = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )
        assert obs_dict2[COLLEGE_AGENT_ID]["obs"][11] == pytest.approx(1.0)

        # Case 3: Out of bounds low -> 0.93 -> clipped to 0.0
        grid_result.bus_vm_pu = {i: 0.93 for i in range(33)}
        obs_dict3 = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )
        assert obs_dict3[COLLEGE_AGENT_ID]["obs"][11] == pytest.approx(0.0)

    def test_price_normalisation(self, sample_inputs, sample_metadata):
        """Verify price normalization relative to grid buy and sell bounds."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        # Price = 5.855, bounds = 8.15 / 3.56 -> (5.855 - 3.56) / (8.15 - 3.56) = 0.5
        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        assert obs_dict[COLLEGE_AGENT_ID]["obs"][15] == pytest.approx(0.5)

    def test_grid_flow_normalisation(self, sample_inputs, sample_metadata):
        """Verify grid active power flow is normalized relative to grid limit."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        # Import of 500 kW, limit 2000 kW -> 500 / 2000 = 0.25
        grid_result.p_grid_kw = 500.0
        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )
        assert obs_dict[COLLEGE_AGENT_ID]["obs"][13] == pytest.approx(0.25)

        # Export of 1000 kW -> -1000 / 2000 = -0.50
        grid_result.p_grid_kw = -1000.0
        obs_dict2 = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )
        assert obs_dict2[COLLEGE_AGENT_ID]["obs"][13] == pytest.approx(-0.50)

    def test_cyclical_time_encoding(self, sample_inputs, sample_metadata):
        """Verify hour and day sin/cos cyclical encoding matches specifications."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        # Hour 14 (sin = sin(2pi*14/24) = -0.5, cos = cos(2pi*14/24) = -0.866)
        # Day of week for 2026-06-24 is Wednesday (day=2, 0-indexed: Mon=0, Tue=1, Wed=2)
        # sin(2pi*2/7) = 0.9749, cos = -0.2225
        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        obs = obs_dict[COLLEGE_AGENT_ID]["obs"]
        assert obs[19] == pytest.approx(math.sin(2 * math.pi * 14 / 24.0))
        assert obs[20] == pytest.approx(math.cos(2 * math.pi * 14 / 24.0))
        assert obs[21] == pytest.approx(math.sin(2 * math.pi * 2 / 7.0))
        assert obs[22] == pytest.approx(math.cos(2 * math.pi * 2 / 7.0))

    def test_zero_peak_demand_handling(self, sample_inputs, sample_metadata):
        """Verify that a zero peak demand in metadata doesn't cause division by zero or NaN."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        # Modify metadata for solar_01 to have 0 peak demand
        sample_metadata["buildings"]["solar_01"]["profile_stats"]["demand_kw"][
            "peak"
        ] = 0.0

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        assert (
            obs_dict["solar_01"]["obs"][1] == 0.0
        )  # normalised demand should fallback to 0.0
        assert not np.isnan(obs_dict["solar_01"]["obs"]).any()

    def test_zero_peak_solar_handling(self, sample_inputs, sample_metadata):
        """Verify that pure consumers have zero peak solar in metadata and normalized solar features are 0.0."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        # Solar feature at index 0 and surplus at index 3 must be 0 for consumers
        for consumer_id in CONSUMER_AGENT_IDS:
            assert obs_dict[consumer_id]["obs"][0] == 0.0
            assert obs_dict[consumer_id]["obs"][3] == 0.0
            assert not np.isnan(obs_dict[consumer_id]["obs"]).any()

    def test_grid_result_none_bypass_mode(self, sample_inputs, sample_metadata):
        """Verify default values are correctly injected when grid_result=None (bypass mode)."""
        (
            demands,
            solar,
            battery,
            last_actions,
            _,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            None,  # bypass mode
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        obs = obs_dict[COLLEGE_AGENT_ID]["obs"]
        # local_bus_voltage = 1.0 pu -> normalized to (1.0 - 0.95) / 0.10 = 0.5
        assert obs[11] == pytest.approx(0.5)
        # local_line_loading = 0.0
        assert obs[12] == pytest.approx(0.0)
        # grid_net_power = 0.0
        assert obs[13] == pytest.approx(0.0)
        # transformer_loading = 0.0
        assert obs[14] == pytest.approx(0.0)

        # Critic state values
        critic_state = obs_dict[COLLEGE_AGENT_ID]["state"]
        # min_bus_voltage at 237 -> 0.5, max_line_loading at 238 -> 0.0, transformer_loading at 239 -> 0.0
        assert critic_state[237] == pytest.approx(0.5)
        assert critic_state[238] == pytest.approx(0.0)
        assert critic_state[239] == pytest.approx(0.0)

    def test_critic_state_identical_across_agents(self, sample_inputs, sample_metadata):
        """Verify that centralized critic state is exactly identical (element-wise equal) across all agents."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        first_state = obs_dict[COLLEGE_AGENT_ID]["state"]
        for aid in ALL_AGENT_IDS:
            assert np.array_equal(obs_dict[aid]["state"], first_state)

    def test_observation_values_finite(self, sample_inputs, sample_metadata):
        """Verify that all observations are finite (no NaN or Inf)."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        for aid in ALL_AGENT_IDS:
            assert np.isfinite(obs_dict[aid]["obs"]).all()
            assert np.isfinite(obs_dict[aid]["state"]).all()

    def test_loading_clipping(self):
        """Verify normalise_loading clips values greater than 100% to 1.0."""
        assert normalise_loading(150.0) == 1.0

    def test_grid_price_clipping(self):
        """Verify normalise_grid_price clips values greater than MAX_GRID_RATE to 1.0."""
        assert normalise_grid_price(MAX_GRID_RATE * 2) == 1.0

    def test_action_clipping(self, sample_inputs, sample_metadata):
        """Verify build_observations clips last_action to [0, 1]."""
        (
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
        ) = sample_inputs

        # Inject out-of-bounds actions
        last_actions[COLLEGE_AGENT_ID] = np.array([-0.5, 1.5, 2.0], dtype=np.float32)

        obs_dict = build_observations(
            demands,
            solar,
            battery,
            last_actions,
            grid_result,
            market_state,
            timestamp,
            grid_buy,
            grid_sell,
            sample_metadata,
        )

        # own_last_action is indices 5, 6, 7 in the uniform actor observation vector
        action_features = obs_dict[COLLEGE_AGENT_ID]["obs"][5:8]
        assert np.array_equal(
            action_features, np.array([0.0, 1.0, 1.0], dtype=np.float32)
        )
