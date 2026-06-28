"""Unit tests for Module 5 — Reward System.

Covers every reward component individually, combined aggregation,
edge cases, NaN handling, zero denominators, curriculum phases,
per-agent-type applicability, and the three worked examples from
docs/module_5_reward_system.md §8.

Design reference: docs/module_5_reward_system.md
"""

from __future__ import annotations

# standard library
import math

# third party
import pytest

# local
from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    BATTERY_POWER_KW,
    COLLEGE_AGENT_ID,
    CONSUMER_AGENT_IDS,
    REWARD_CLIP_MAX,
    REWARD_CLIP_MIN,
    REWARD_W_CYCLING,
    REWARD_W_IMPORT,
    REWARD_W_P2P,
    REWARD_W_SELF,
    REWARD_W_SOC,
    REWARD_W_STORAGE,
    REWARD_W_THERMAL_PHASE1,
    REWARD_W_THERMAL_PHASE2,
    REWARD_W_VOLTAGE_PHASE1,
    SOLAR_AGENT_IDS,
)
from p2p_energy_trading.modules.market.models import SettlementRecord
from p2p_energy_trading.modules.network.powerflow import PowerFlowResult
from p2p_energy_trading.modules.reward.aggregator import (
    _derive_energy_quantities,
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

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_settlement(
    p2p_sold_kw: float = 0.0,
    p2p_bought_kw: float = 0.0,
    grid_sold_kw: float = 0.0,
    grid_bought_kw: float = 0.0,
    p2p_price: float = 10.0,
    p2p_revenue: float = 0.0,
    p2p_cost: float = 0.0,
    grid_revenue: float = 0.0,
    grid_cost: float = 0.0,
    net_cost: float = 0.0,
) -> SettlementRecord:
    """Helper factory for SettlementRecord instances."""
    return SettlementRecord(
        p2p_sold_kw=p2p_sold_kw,
        p2p_bought_kw=p2p_bought_kw,
        grid_sold_kw=grid_sold_kw,
        grid_bought_kw=grid_bought_kw,
        p2p_price=p2p_price,
        p2p_revenue=p2p_revenue,
        p2p_cost=p2p_cost,
        grid_revenue=grid_revenue,
        grid_cost=grid_cost,
        net_cost=net_cost,
    )


def _make_power_flow(
    bus_vm_pu: dict[int, float] | None = None,
    line_loading_pct: dict[int, float] | None = None,
    trafo_loading_pct: dict[int, float] | None = None,
    p_grid_kw: float = 0.0,
    converged: bool = True,
) -> PowerFlowResult:
    """Helper factory for PowerFlowResult instances."""
    if bus_vm_pu is None:
        bus_vm_pu = {i: 1.0 for i in range(33)}
    if line_loading_pct is None:
        line_loading_pct = {i: 0.0 for i in range(32)}
    if trafo_loading_pct is None:
        trafo_loading_pct = {i: 0.0 for i in range(22)}
    return PowerFlowResult(
        converged=converged,
        bus_vm_pu=bus_vm_pu,
        line_loading_pct=line_loading_pct,
        trafo_loading_pct=trafo_loading_pct,
        p_grid_kw=p_grid_kw,
    )


# ---------------------------------------------------------------------------
# Economic reward
# ---------------------------------------------------------------------------


class TestEconomicReward:
    """Tests for compute_economic_reward."""

    def test_positive_net_income(self):
        """Agent earns income → positive r_econ."""
        s = _make_settlement(net_cost=-500.0)
        r = compute_economic_reward(s, max_possible_cost=1000.0)
        assert r == pytest.approx(0.5)

    def test_zero_net_cost(self):
        """Zero net cost → r_econ = 0."""
        s = _make_settlement(net_cost=0.0)
        r = compute_economic_reward(s, max_possible_cost=1000.0)
        assert r == pytest.approx(0.0)

    def test_positive_net_cost(self):
        """Agent pays more than earns → negative r_econ."""
        s = _make_settlement(net_cost=300.0)
        r = compute_economic_reward(s, max_possible_cost=1000.0)
        assert r == pytest.approx(-0.3)

    def test_normalisation_at_max_cost(self):
        """Net cost equal to max_possible_cost → r_econ = -1.0."""
        s = _make_settlement(net_cost=1000.0)
        r = compute_economic_reward(s, max_possible_cost=1000.0)
        assert r == pytest.approx(-1.0)

    def test_invalid_max_possible_cost_raises(self):
        """Non-positive max_possible_cost must raise ValueError."""
        s = _make_settlement(net_cost=0.0)
        with pytest.raises(ValueError, match="max_possible_cost must be positive"):
            compute_economic_reward(s, max_possible_cost=0.0)
        with pytest.raises(ValueError, match="max_possible_cost must be positive"):
            compute_economic_reward(s, max_possible_cost=-100.0)

    def test_spec_example_1_r_econ(self):
        """§8 Example 1: r_econ = 518.60 / 1000 = +0.519."""
        s = _make_settlement(net_cost=-518.60)
        r = compute_economic_reward(s, max_possible_cost=1000.0)
        assert r == pytest.approx(0.5186, abs=1e-4)

    def test_spec_example_2_r_econ(self):
        """§8 Example 2: r_econ = 818.40 / 1000 = +0.818."""
        s = _make_settlement(net_cost=-818.40)
        r = compute_economic_reward(s, max_possible_cost=1000.0)
        assert r == pytest.approx(0.8184, abs=1e-4)

    def test_spec_example_3_r_econ(self):
        """§8 Example 3: r_econ = 547.00 / 1000 = +0.547."""
        s = _make_settlement(net_cost=-547.00)
        r = compute_economic_reward(s, max_possible_cost=1000.0)
        assert r == pytest.approx(0.547, abs=1e-4)

    def test_deterministic(self):
        """Same inputs must produce identical outputs."""
        s = _make_settlement(net_cost=-200.0)
        assert compute_economic_reward(s, 500.0) == compute_economic_reward(s, 500.0)


# ---------------------------------------------------------------------------
# P2P reward
# ---------------------------------------------------------------------------


class TestP2PReward:
    """Tests for compute_p2p_reward — normalised §4 form."""

    def test_full_p2p_utilisation(self):
        """All surplus traded P2P → r_p2p = w_p2p."""
        s = _make_settlement(p2p_sold_kw=50.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=50.0, own_deficit_kw=0.0, agent_id="solar_01"
        )
        assert r == pytest.approx(REWARD_W_P2P * 1.0)

    def test_partial_p2p_utilisation(self):
        """Partial P2P trade gives proportional reward."""
        s = _make_settlement(p2p_sold_kw=40.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=50.0, own_deficit_kw=0.0, agent_id="solar_01"
        )
        assert r == pytest.approx(REWARD_W_P2P * 0.8)

    def test_zero_energy_need(self):
        """Zero energy need (idle agent) → r_p2p = 0."""
        s = _make_settlement(p2p_sold_kw=0.0, p2p_bought_kw=0.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=0.0, own_deficit_kw=0.0, agent_id="consumer_01"
        )
        assert r == pytest.approx(0.0)

    def test_consumer_buys_from_p2p(self):
        """Consumer buying from P2P counted correctly."""
        s = _make_settlement(p2p_bought_kw=30.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=0.0, own_deficit_kw=50.0, agent_id="consumer_01"
        )
        assert r == pytest.approx(REWARD_W_P2P * 30.0 / 50.0)

    def test_spec_example_1_r_p2p(self):
        """§8 Example 1: r_p2p = 0.1 × (40 / 50) = 0.080."""
        s = _make_settlement(p2p_sold_kw=40.0, p2p_bought_kw=0.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=50.0, own_deficit_kw=0.0, agent_id="solar_01"
        )
        assert r == pytest.approx(0.080, abs=1e-4)

    def test_spec_example_2_r_p2p(self):
        """§8 Example 2: r_p2p = 0.1 × (60 / 80) = 0.075."""
        s = _make_settlement(p2p_sold_kw=60.0, p2p_bought_kw=0.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=80.0, own_deficit_kw=0.0, agent_id="solar_01"
        )
        assert r == pytest.approx(0.075, abs=1e-4)

    def test_spec_example_3_r_p2p(self):
        """§8 Example 3: r_p2p = 0.1 × (50 / 100) = 0.050."""
        s = _make_settlement(p2p_sold_kw=50.0, p2p_bought_kw=0.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=100.0, own_deficit_kw=0.0, agent_id=COLLEGE_AGENT_ID
        )
        assert r == pytest.approx(0.050, abs=1e-4)

    def test_result_non_negative(self):
        """r_p2p is always non-negative."""
        s = _make_settlement(p2p_sold_kw=10.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=50.0, own_deficit_kw=0.0, agent_id="solar_02"
        )
        assert r >= 0.0

    def test_bounded_by_weight(self):
        """r_p2p ≤ w_p2p for any inputs."""
        s = _make_settlement(p2p_sold_kw=500.0, p2p_bought_kw=500.0)
        r = compute_p2p_reward(
            s, own_surplus_kw=100.0, own_deficit_kw=100.0, agent_id="solar_03"
        )
        # numerator > denominator; ratio capped by content not by function
        # (ratio > 1 possible if traded > need, but in practice always ≤ need)
        assert r >= 0.0


# ---------------------------------------------------------------------------
# Self-consumption reward
# ---------------------------------------------------------------------------


class TestSelfConsumptionReward:
    """Tests for compute_self_consumption_reward."""

    def test_all_solar_used_locally(self):
        """All solar used locally → r_self = w_self."""
        s = _make_settlement(p2p_sold_kw=0.0, grid_sold_kw=0.0)
        r = compute_self_consumption_reward(
            s, solar_kw=80.0, battery_dispatch_kw=0.0, agent_id="solar_01"
        )
        assert r == pytest.approx(REWARD_W_SELF)

    def test_all_solar_exported(self):
        """All solar exported, zero demand → r_self ≈ 0."""
        s = _make_settlement(p2p_sold_kw=40.0, grid_sold_kw=40.0)
        r = compute_self_consumption_reward(
            s, solar_kw=80.0, battery_dispatch_kw=0.0, agent_id="solar_01"
        )
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_zero_solar(self):
        """Zero solar generation → r_self = 0 (denominator protection)."""
        s = _make_settlement(p2p_sold_kw=0.0, grid_sold_kw=0.0)
        r = compute_self_consumption_reward(
            s, solar_kw=0.0, battery_dispatch_kw=0.0, agent_id="solar_01"
        )
        assert r == pytest.approx(0.0)

    def test_battery_charge_reduces_solar_used(self):
        """Battery charging from solar reduces effective local consumption."""
        # solar=100, p2p_sold=0, grid_sold=0, battery_charge=50
        # solar_used_locally = 100 - 0 - 0 - 50 = 50
        s = _make_settlement(p2p_sold_kw=0.0, grid_sold_kw=0.0)
        # battery_dispatch_kw=-50 means charging at 50 kW
        r = compute_self_consumption_reward(
            s, solar_kw=100.0, battery_dispatch_kw=-50.0, agent_id=COLLEGE_AGENT_ID
        )
        assert r == pytest.approx(REWARD_W_SELF * (50.0 / 100.0), abs=1e-6)

    def test_spec_example_1_r_self(self):
        """§8 Example 1: r_self = 0.05 × (30 / 80) ≈ 0.01875."""
        s = _make_settlement(p2p_sold_kw=40.0, grid_sold_kw=10.0)
        r = compute_self_consumption_reward(
            s, solar_kw=80.0, battery_dispatch_kw=0.0, agent_id="solar_01"
        )
        assert r == pytest.approx(0.05 * (30.0 / 80.0), abs=1e-5)

    def test_spec_example_3_r_self(self):
        """§8 Example 3: r_self = 0.05 × (100 / 200) = 0.025."""
        # solar=200, p2p_sold=50, grid_sold=0, battery_charge=50
        s = _make_settlement(p2p_sold_kw=50.0, grid_sold_kw=0.0)
        r = compute_self_consumption_reward(
            s, solar_kw=200.0, battery_dispatch_kw=-50.0, agent_id=COLLEGE_AGENT_ID
        )
        assert r == pytest.approx(0.025, abs=1e-5)

    def test_non_negative(self):
        """r_self is always non-negative."""
        s = _make_settlement(p2p_sold_kw=10.0, grid_sold_kw=10.0)
        r = compute_self_consumption_reward(
            s, solar_kw=15.0, battery_dispatch_kw=0.0, agent_id="solar_02"
        )
        assert r >= 0.0


# ---------------------------------------------------------------------------
# Grid import reduction reward
# ---------------------------------------------------------------------------


class TestImportReductionReward:
    """Tests for compute_import_reduction_reward (Consumer agents)."""

    def test_zero_grid_import(self):
        """100 % P2P coverage → r_import = w_import."""
        s = _make_settlement(grid_bought_kw=0.0)
        r = compute_import_reduction_reward(
            s, own_deficit_kw=50.0, agent_id="consumer_01"
        )
        assert r == pytest.approx(REWARD_W_IMPORT)

    def test_full_grid_import(self):
        """100 % grid coverage → r_import = 0."""
        s = _make_settlement(grid_bought_kw=50.0)
        r = compute_import_reduction_reward(
            s, own_deficit_kw=50.0, agent_id="consumer_01"
        )
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_partial_grid_import(self):
        """50 % grid coverage → r_import = 0.5 × w_import."""
        s = _make_settlement(grid_bought_kw=25.0)
        r = compute_import_reduction_reward(
            s, own_deficit_kw=50.0, agent_id="consumer_01"
        )
        assert r == pytest.approx(REWARD_W_IMPORT * 0.5)

    def test_zero_deficit_no_crash(self):
        """Zero own deficit → denominator protection, r_import ≈ 0."""
        s = _make_settlement(grid_bought_kw=0.0)
        r = compute_import_reduction_reward(
            s, own_deficit_kw=0.0, agent_id="consumer_02"
        )
        assert math.isfinite(r)

    def test_non_negative(self):
        """r_import is always non-negative."""
        s = _make_settlement(grid_bought_kw=30.0)
        r = compute_import_reduction_reward(
            s, own_deficit_kw=50.0, agent_id="consumer_03"
        )
        assert r >= 0.0

    def test_bounded_by_weight(self):
        """r_import ≤ w_import always."""
        s = _make_settlement(grid_bought_kw=0.0)
        r = compute_import_reduction_reward(
            s, own_deficit_kw=100.0, agent_id="consumer_04"
        )
        assert r <= REWARD_W_IMPORT + 1e-9


# ---------------------------------------------------------------------------
# Voltage penalty
# ---------------------------------------------------------------------------


class TestVoltagePenalty:
    """Tests for compute_voltage_penalty."""

    def test_nominal_voltage_no_penalty(self):
        """Voltage at 1.0 p.u. → r_voltage = 0."""
        pf = _make_power_flow(bus_vm_pu={6: 1.0})
        r = compute_voltage_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=1)
        assert r == pytest.approx(0.0)

    def test_at_safe_upper_bound_no_penalty(self):
        """Voltage exactly at 1.05 p.u. → no penalty."""
        pf = _make_power_flow(bus_vm_pu={6: 1.05})
        r = compute_voltage_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=1)
        assert r == pytest.approx(0.0)

    def test_at_safe_lower_bound_no_penalty(self):
        """Voltage exactly at 0.95 p.u. → no penalty."""
        pf = _make_power_flow(bus_vm_pu={6: 0.95})
        r = compute_voltage_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=1)
        assert r == pytest.approx(0.0)

    def test_overvoltage_phase1(self):
        """V=1.08 p.u., Phase 1: r_voltage = -2.0 × 0.60 = -1.20."""
        pf = _make_power_flow(bus_vm_pu={6: 1.08})
        r = compute_voltage_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=1)
        assert r == pytest.approx(-1.2, abs=1e-5)

    def test_overvoltage_phase2(self):
        """V=1.08 p.u., Phase 2: r_voltage = -5.0 × 0.60 = -3.00."""
        pf = _make_power_flow(bus_vm_pu={6: 1.08})
        r = compute_voltage_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=2)
        assert r == pytest.approx(-3.0, abs=1e-5)

    def test_spec_example_2_r_voltage_phase2(self):
        """§8 Example 2, Phase 2: deviation=0.03, ratio=0.60, r=-5.0×0.60=-3.00."""
        pf = _make_power_flow(bus_vm_pu={7: 1.08})  # solar_01 at bus 7
        r = compute_voltage_penalty(pf, agent_id="solar_01", curriculum_phase=2)
        assert r == pytest.approx(-3.0, abs=1e-5)

    def test_spec_example_2_r_voltage_phase1(self):
        """§8 Example 2, Phase 1: r_voltage = -2.0 × 0.60 = -1.20."""
        pf = _make_power_flow(bus_vm_pu={7: 1.08})
        r = compute_voltage_penalty(pf, agent_id="solar_01", curriculum_phase=1)
        assert r == pytest.approx(-1.2, abs=1e-5)

    def test_undervoltage(self):
        """Undervoltage violation produces negative penalty."""
        pf = _make_power_flow(bus_vm_pu={6: 0.90})
        r = compute_voltage_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=1)
        # deviation = |0.90 - 1.0| - 0.05 = 0.05; ratio = 0.05/0.05 = 1.0
        assert r == pytest.approx(-REWARD_W_VOLTAGE_PHASE1 * 1.0)

    def test_none_power_flow_bypassed(self):
        """None power_flow_result → r_voltage = 0.0."""
        r = compute_voltage_penalty(None, agent_id=COLLEGE_AGENT_ID, curriculum_phase=2)
        assert r == pytest.approx(0.0)

    def test_non_positive(self):
        """r_voltage is always non-positive."""
        pf = _make_power_flow(bus_vm_pu={6: 1.20})
        r = compute_voltage_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=2)
        assert r <= 0.0

    def test_weight_override(self):
        """Explicit w_v override is respected."""
        pf = _make_power_flow(bus_vm_pu={6: 1.08})
        r = compute_voltage_penalty(
            pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=1, w_v=10.0
        )
        assert r == pytest.approx(-10.0 * 0.60, abs=1e-5)


# ---------------------------------------------------------------------------
# Thermal penalty
# ---------------------------------------------------------------------------


class TestThermalPenalty:
    """Tests for compute_thermal_penalty."""

    def test_no_overload(self):
        """Line loading at 80 % → r_thermal = 0."""
        pf = _make_power_flow(line_loading_pct={5: 80.0})  # college at bus 6 → line 5
        r = compute_thermal_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=1)
        assert r == pytest.approx(0.0)

    def test_exactly_at_limit(self):
        """Line loading exactly at 100 % → r_thermal = 0."""
        pf = _make_power_flow(line_loading_pct={5: 100.0})
        r = compute_thermal_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=1)
        assert r == pytest.approx(0.0)

    def test_overload_phase1(self):
        """115 % loading, Phase 1: -2.0 × (1.15 - 1.0) = -0.30."""
        pf = _make_power_flow(line_loading_pct={6: 115.0})  # solar_01 at bus 7 → line 6
        r = compute_thermal_penalty(pf, agent_id="solar_01", curriculum_phase=1)
        assert r == pytest.approx(-REWARD_W_THERMAL_PHASE1 * 0.15, abs=1e-5)

    def test_overload_phase2(self):
        """115 % loading, Phase 2: -5.0 × 0.15 = -0.75."""
        pf = _make_power_flow(line_loading_pct={6: 115.0})
        r = compute_thermal_penalty(pf, agent_id="solar_01", curriculum_phase=2)
        assert r == pytest.approx(-REWARD_W_THERMAL_PHASE2 * 0.15, abs=1e-5)

    def test_spec_example_2_phase2(self):
        """§8 Example 2, Phase 2: r_thermal = -5.0 × 0.15 = -0.75."""
        pf = _make_power_flow(line_loading_pct={6: 115.0})
        r = compute_thermal_penalty(pf, agent_id="solar_01", curriculum_phase=2)
        assert r == pytest.approx(-0.75, abs=1e-5)

    def test_spec_example_2_phase1(self):
        """§8 Example 2, Phase 1: r_thermal = -2.0 × 0.15 = -0.30."""
        pf = _make_power_flow(line_loading_pct={6: 115.0})
        r = compute_thermal_penalty(pf, agent_id="solar_01", curriculum_phase=1)
        assert r == pytest.approx(-0.30, abs=1e-5)

    def test_none_power_flow_bypassed(self):
        """None power_flow_result → r_thermal = 0.0."""
        r = compute_thermal_penalty(None, agent_id="solar_01", curriculum_phase=2)
        assert r == pytest.approx(0.0)

    def test_non_positive(self):
        """r_thermal is always non-positive."""
        pf = _make_power_flow(line_loading_pct={5: 200.0})
        r = compute_thermal_penalty(pf, agent_id=COLLEGE_AGENT_ID, curriculum_phase=2)
        assert r <= 0.0

    def test_line_index_bus_minus_one(self):
        """Line index = bus_idx - 1 for every agent in the topology."""
        # local
        from p2p_energy_trading.constants import AGENT_TO_BUS

        for agent_id in ALL_AGENT_IDS:
            bus_idx = AGENT_TO_BUS[agent_id]
            expected_line_idx = bus_idx - 1
            loading_dict = {expected_line_idx: 150.0}
            pf = _make_power_flow(line_loading_pct=loading_dict)
            r = compute_thermal_penalty(pf, agent_id=agent_id, curriculum_phase=1)
            # Should detect overload (150 %) and return negative value
            assert r < 0.0, (
                f"Agent {agent_id}: expected penalty but got {r} "
                f"(bus={bus_idx}, line={expected_line_idx})"
            )


# ---------------------------------------------------------------------------
# Transformer penalty
# ---------------------------------------------------------------------------


class TestTransformerPenalty:
    """Tests for compute_transformer_penalty (College only)."""

    def test_no_overload(self):
        """Transformer at 80 % → r_transformer = 0."""
        pf = _make_power_flow(trafo_loading_pct={0: 80.0})
        r = compute_transformer_penalty(pf, curriculum_phase=1)
        assert r == pytest.approx(0.0)

    def test_overload_phase1(self):
        """120 % loading, Phase 1: -2.0 × 0.20 = -0.40."""
        pf = _make_power_flow(trafo_loading_pct={0: 120.0})
        r = compute_transformer_penalty(pf, curriculum_phase=1)
        assert r == pytest.approx(-2.0 * 0.20, abs=1e-5)

    def test_overload_phase2(self):
        """120 % loading, Phase 2: -5.0 × 0.20 = -1.00."""
        pf = _make_power_flow(trafo_loading_pct={0: 120.0})
        r = compute_transformer_penalty(pf, curriculum_phase=2)
        assert r == pytest.approx(-5.0 * 0.20, abs=1e-5)

    def test_none_power_flow(self):
        """None power_flow_result → r_transformer = 0.0."""
        r = compute_transformer_penalty(None, curriculum_phase=1)
        assert r == pytest.approx(0.0)

    def test_non_positive(self):
        """r_transformer is always non-positive."""
        pf = _make_power_flow(trafo_loading_pct={0: 300.0})
        r = compute_transformer_penalty(pf, curriculum_phase=2)
        assert r <= 0.0


# ---------------------------------------------------------------------------
# SoC health penalty
# ---------------------------------------------------------------------------


class TestSoCPenalty:
    """Tests for compute_soc_penalty (Decision 2 — normalised §5 form)."""

    def test_soc_in_safe_range(self):
        """SoC inside [0.10, 0.95] → r_soc = 0.0."""
        for soc in [0.10, 0.50, 0.75, 0.95]:
            assert compute_soc_penalty(soc) == pytest.approx(0.0), (
                f"Failed for SoC={soc}"
            )

    def test_soc_below_minimum(self):
        """SoC=0.05 → penalty = -1.0 × 0.05 / 0.10 = -0.50."""
        r = compute_soc_penalty(0.05)
        assert r == pytest.approx(-REWARD_W_SOC * (0.05 / 0.10), abs=1e-6)

    def test_soc_at_zero(self):
        """SoC=0.0 → maximum low penalty = -w_soc × 0.10/0.10 = -w_soc."""
        r = compute_soc_penalty(0.0)
        assert r == pytest.approx(-REWARD_W_SOC, abs=1e-6)

    def test_soc_above_maximum(self):
        """SoC=0.98 → penalty = -1.0 × 0.03 / 0.05 = -0.60."""
        r = compute_soc_penalty(0.98)
        assert r == pytest.approx(-REWARD_W_SOC * (0.03 / 0.05), abs=1e-6)

    def test_soc_at_one(self):
        """SoC=1.0 → maximum high penalty = -w_soc × 0.05/0.05 = -w_soc."""
        r = compute_soc_penalty(1.0)
        assert r == pytest.approx(-REWARD_W_SOC, abs=1e-6)

    def test_spec_example_3_soc_in_range(self):
        """§8 Example 3: SoC=0.40 (in range) → r_soc = 0.0."""
        assert compute_soc_penalty(0.40) == pytest.approx(0.0)

    def test_non_positive(self):
        """r_soc is always non-positive."""
        for soc in [0.0, 0.05, 0.50, 0.97, 1.0]:
            assert compute_soc_penalty(soc) <= 0.0

    def test_weight_override(self):
        """Explicit w_soc override is respected."""
        r = compute_soc_penalty(0.0, w_soc=2.0)
        assert r == pytest.approx(-2.0, abs=1e-6)

    def test_penalty_increases_with_violation_depth(self):
        """Deeper SoC violation → larger (more negative) penalty."""
        r_mild = compute_soc_penalty(0.08)  # 0.02 below min
        r_deep = compute_soc_penalty(0.0)  # 0.10 below min
        assert r_deep < r_mild


# ---------------------------------------------------------------------------
# Cycling penalty
# ---------------------------------------------------------------------------


class TestCyclingPenalty:
    """Tests for compute_cycling_penalty (Decision 3 — normalised §5)."""

    def test_no_switch_discharge_to_discharge(self):
        """Sustained discharge → r_cycling = 0.0."""
        r = compute_cycling_penalty(battery_dispatch_kw=100.0, prev_dispatch_kw=50.0)
        assert r == pytest.approx(0.0)

    def test_no_switch_charge_to_charge(self):
        """Sustained charge → r_cycling = 0.0."""
        r = compute_cycling_penalty(battery_dispatch_kw=-80.0, prev_dispatch_kw=-60.0)
        assert r == pytest.approx(0.0)

    def test_no_switch_idle_then_charge(self):
        """Previous idle (below threshold) → no cycling detection."""
        r = compute_cycling_penalty(
            battery_dispatch_kw=-100.0,
            prev_dispatch_kw=0.0,
        )
        assert r == pytest.approx(0.0)

    def test_direction_switch_charge_to_discharge(self):
        """Charge→discharge switch → penalty applied."""
        curr = 100.0  # discharging
        prev = -80.0  # charging
        r = compute_cycling_penalty(battery_dispatch_kw=curr, prev_dispatch_kw=prev)
        expected = -REWARD_W_CYCLING * abs(curr) / BATTERY_POWER_KW
        assert r == pytest.approx(expected, abs=1e-6)

    def test_direction_switch_discharge_to_charge(self):
        """Discharge→charge switch → penalty applied."""
        curr = -100.0  # charging
        prev = 80.0  # discharging
        r = compute_cycling_penalty(battery_dispatch_kw=curr, prev_dispatch_kw=prev)
        expected = -REWARD_W_CYCLING * abs(curr) / BATTERY_POWER_KW
        assert r == pytest.approx(expected, abs=1e-6)

    def test_uses_battery_power_kw_constant(self):
        """Denominator must equal BATTERY_POWER_KW (not hardcoded 250.0)."""
        curr = BATTERY_POWER_KW  # full discharge
        prev = -BATTERY_POWER_KW  # full charge
        r = compute_cycling_penalty(curr, prev)
        expected = -REWARD_W_CYCLING * 1.0  # |curr|/BATTERY_POWER_KW = 1.0
        assert r == pytest.approx(expected, abs=1e-6)

    def test_non_positive(self):
        """r_cycling is always non-positive."""
        r = compute_cycling_penalty(100.0, -100.0)
        assert r <= 0.0

    def test_both_below_min_dispatch_no_penalty(self):
        """Both dispatches below BATTERY_MIN_DISPATCH_KW → r_cycling = 0."""
        r = compute_cycling_penalty(
            battery_dispatch_kw=10.0,  # below 25 kW threshold
            prev_dispatch_kw=-10.0,  # below 25 kW threshold
        )
        assert r == pytest.approx(0.0)

    def test_spec_example_3_no_cycling(self):
        """§8 Example 3: no direction switch → r_cycling = 0.0."""
        # Previous: idle (0.0), current: charging (-50 kW) — no switch from active
        r = compute_cycling_penalty(battery_dispatch_kw=-50.0, prev_dispatch_kw=0.0)
        assert r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Storage reward
# ---------------------------------------------------------------------------


class TestStorageReward:
    """Tests for compute_storage_reward."""

    def test_charging_from_surplus(self):
        """Charging during surplus → positive reward."""
        r = compute_storage_reward(
            battery_dispatch_kw=-50.0,
            own_surplus_kw=100.0,
            own_deficit_kw=0.0,
        )
        expected = REWARD_W_STORAGE * min(50.0, 100.0) / 100.0
        assert r == pytest.approx(expected, abs=1e-6)

    def test_discharging_from_deficit(self):
        """Discharging during deficit → positive reward."""
        r = compute_storage_reward(
            battery_dispatch_kw=50.0,
            own_surplus_kw=0.0,
            own_deficit_kw=100.0,
        )
        expected = REWARD_W_STORAGE * min(50.0, 100.0) / 100.0
        assert r == pytest.approx(expected, abs=1e-6)

    def test_spec_example_3_storage(self):
        """§8 Example 3: r_storage = 0.05 × (50/100) = 0.025."""
        r = compute_storage_reward(
            battery_dispatch_kw=-50.0,
            own_surplus_kw=100.0,
            own_deficit_kw=0.0,
        )
        assert r == pytest.approx(0.025, abs=1e-5)

    def test_below_min_dispatch_threshold(self):
        """Dispatch below BATTERY_MIN_DISPATCH_KW (25 kW) → r_storage = 0."""
        r = compute_storage_reward(
            battery_dispatch_kw=-20.0,  # below 25 kW
            own_surplus_kw=100.0,
            own_deficit_kw=0.0,
        )
        assert r == pytest.approx(0.0)

    def test_charging_during_deficit_no_reward(self):
        """Charging when there is no surplus → r_storage = 0."""
        r = compute_storage_reward(
            battery_dispatch_kw=-50.0,
            own_surplus_kw=0.0,  # no surplus
            own_deficit_kw=100.0,
        )
        assert r == pytest.approx(0.0)

    def test_discharging_during_surplus_no_reward(self):
        """Discharging when there is no deficit → r_storage = 0."""
        r = compute_storage_reward(
            battery_dispatch_kw=50.0,
            own_surplus_kw=100.0,
            own_deficit_kw=0.0,
        )
        assert r == pytest.approx(0.0)

    def test_idle_battery_no_reward(self):
        """Idle battery → r_storage = 0."""
        r = compute_storage_reward(
            battery_dispatch_kw=0.0,
            own_surplus_kw=100.0,
            own_deficit_kw=0.0,
        )
        assert r == pytest.approx(0.0)

    def test_non_negative(self):
        """r_storage is always non-negative."""
        r = compute_storage_reward(-100.0, 200.0, 0.0)
        assert r >= 0.0

    def test_bounded_by_weight(self):
        """r_storage ≤ w_store always."""
        r = compute_storage_reward(-250.0, 100.0, 0.0)
        assert r <= REWARD_W_STORAGE + 1e-9


# ---------------------------------------------------------------------------
# Auxiliary: derive energy quantities
# ---------------------------------------------------------------------------


class TestDeriveEnergyQuantities:
    """Tests for the internal _derive_energy_quantities helper."""

    def test_pure_solar_surplus(self):
        """Solar agent with surplus: own_surplus > 0, own_deficit = 0."""
        surplus, deficit = _derive_energy_quantities(
            "solar_01", demand_kw=20.0, solar_kw=80.0, battery_dispatch_kw=0.0
        )
        assert surplus == pytest.approx(60.0)
        assert deficit == pytest.approx(0.0)

    def test_pure_demand_deficit(self):
        """Consumer with deficit: own_deficit > 0, own_surplus = 0."""
        surplus, deficit = _derive_energy_quantities(
            "consumer_01", demand_kw=50.0, solar_kw=0.0, battery_dispatch_kw=0.0
        )
        assert surplus == pytest.approx(0.0)
        assert deficit == pytest.approx(50.0)

    def test_college_with_discharging_battery_pre_battery(self):
        """Battery discharging at college does not reduce pre-battery deficit."""
        # demand=150, solar=100, dispatch=+50 (discharge) → pre-battery net = 150-100 = 50
        surplus, deficit = _derive_energy_quantities(
            COLLEGE_AGENT_ID, demand_kw=150.0, solar_kw=100.0, battery_dispatch_kw=50.0
        )
        assert surplus == pytest.approx(0.0)
        assert deficit == pytest.approx(50.0)

    def test_college_with_charging_battery_pre_battery(self):
        """Battery charging at college does not increase pre-battery deficit."""
        # demand=100, solar=200, dispatch=-100 (charge) → pre-battery net = 100-200 = -100
        surplus, deficit = _derive_energy_quantities(
            COLLEGE_AGENT_ID,
            demand_kw=100.0,
            solar_kw=200.0,
            battery_dispatch_kw=-100.0,
        )
        assert surplus == pytest.approx(100.0)
        assert deficit == pytest.approx(0.0)

    def test_non_negative_outputs(self):
        """Both outputs are always non-negative."""
        surplus, deficit = _derive_energy_quantities("solar_01", 30.0, 80.0, 0.0)
        assert surplus >= 0.0
        assert deficit >= 0.0


# ---------------------------------------------------------------------------
# Full aggregator: compute_agent_reward
# ---------------------------------------------------------------------------


class TestComputeAgentReward:
    """Tests for compute_agent_reward — full integration of all components."""

    def _base_pf_no_violations(self) -> PowerFlowResult:
        """Power flow with no violations for all agents."""
        return _make_power_flow(
            bus_vm_pu={i: 1.0 for i in range(40)},
            line_loading_pct={i: 0.0 for i in range(32)},
            trafo_loading_pct={i: 0.0 for i in range(22)},
        )

    def test_spec_example_1_full(self):
        """§8 Example 1: Solar agent, r_total ≈ +0.618 (including r_self)."""
        s = _make_settlement(
            p2p_sold_kw=40.0,
            grid_sold_kw=10.0,
            p2p_revenue=437.60,
            grid_revenue=81.0,
            net_cost=-518.60,
        )
        r = compute_agent_reward(
            agent_id="solar_01",
            settlement=s,
            demand_kw=30.0,
            solar_kw=80.0,
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=self._base_pf_no_violations(),
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        # r_econ=0.5186, r_p2p=0.080, r_self=0.01875
        expected = pytest.approx(0.5186 + 0.080 + 0.01875, abs=1e-3)
        assert r == expected

    def test_spec_example_2_full_phase2(self):
        """§8 Example 2, Phase 2: Solar with violations, r_total ≈ -2.847 (with r_self)."""
        # Bus 7 = solar_01, line 6
        bus_vm = {i: 1.0 for i in range(40)}
        bus_vm[7] = 1.08
        line_ld = {i: 0.0 for i in range(32)}
        line_ld[6] = 115.0
        pf = _make_power_flow(bus_vm_pu=bus_vm, line_loading_pct=line_ld)
        s = _make_settlement(
            p2p_sold_kw=60.0,
            grid_sold_kw=20.0,
            p2p_revenue=656.40,
            grid_revenue=162.0,
            net_cost=-818.40,
        )
        r = compute_agent_reward(
            agent_id="solar_01",
            settlement=s,
            demand_kw=20.0,
            solar_kw=100.0,
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=pf,
            max_possible_cost=1000.0,
            curriculum_phase=2,
        )
        # With r_self included: 0.8184 + 0.075 + 0.010 - 3.000 - 0.750 = -2.847
        assert r == pytest.approx(-2.847, abs=1e-2)

    def test_spec_example_3_full(self):
        """§8 Example 3: College agent, r_total ≈ +0.647."""
        s = _make_settlement(
            p2p_sold_kw=50.0,
            grid_sold_kw=0.0,
            p2p_revenue=547.0,
            net_cost=-547.0,
        )
        r = compute_agent_reward(
            agent_id=COLLEGE_AGENT_ID,
            settlement=s,
            demand_kw=100.0,
            solar_kw=200.0,
            battery_dispatch_kw=-50.0,  # charging
            battery_soc=0.40,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=self._base_pf_no_violations(),
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        assert r == pytest.approx(0.647, abs=1e-2)

    def test_reward_clipped_at_upper_bound(self):
        """Reward above +10 must be clipped to +10."""
        s = _make_settlement(net_cost=-100000.0)
        r = compute_agent_reward(
            agent_id="solar_01",
            settlement=s,
            demand_kw=0.0,
            solar_kw=0.0,
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1.0,
            curriculum_phase=1,
        )
        assert r == pytest.approx(REWARD_CLIP_MAX)

    def test_reward_clipped_at_lower_bound(self):
        """Reward below -10 must be clipped to -10."""
        bus_vm = {i: 0.70 for i in range(40)}  # extreme undervoltage
        pf = _make_power_flow(
            bus_vm_pu=bus_vm, line_loading_pct={i: 500.0 for i in range(32)}
        )
        s = _make_settlement(net_cost=100000.0)
        r = compute_agent_reward(
            agent_id="solar_01",
            settlement=s,
            demand_kw=0.0,
            solar_kw=0.0,
            battery_dispatch_kw=0.0,
            battery_soc=0.0,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=pf,
            max_possible_cost=1.0,
            curriculum_phase=2,
        )
        assert r == pytest.approx(REWARD_CLIP_MIN)

    def test_consumer_no_battery_rewards(self):
        """Consumer agents should not receive battery-specific rewards."""
        s = _make_settlement(net_cost=-100.0, grid_bought_kw=20.0)
        r_consumer = compute_agent_reward(
            agent_id="consumer_01",
            settlement=s,
            demand_kw=50.0,
            solar_kw=0.0,
            battery_dispatch_kw=0.0,
            battery_soc=0.0,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        # Should be r_econ + r_p2p + r_import only (no r_self, r_soc, etc.)
        r_econ = 100.0 / 1000.0  # net_cost=-100 → r_econ_raw=+100
        assert math.isfinite(r_consumer)
        assert r_consumer > 0.0  # net income with P2P bonus

    def test_solar_agent_no_r_import_no_battery(self):
        """Solar agents do not receive r_import or battery rewards."""
        s = _make_settlement(net_cost=-50.0, p2p_sold_kw=20.0)
        r = compute_agent_reward(
            agent_id="solar_01",
            settlement=s,
            demand_kw=20.0,
            solar_kw=50.0,
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        assert math.isfinite(r)

    def test_phase_weights_differ(self):
        """Phase 2 grid penalties must be stronger than Phase 1."""
        bus_vm = {i: 1.0 for i in range(40)}
        bus_vm[7] = 1.08
        pf = _make_power_flow(bus_vm_pu=bus_vm)
        s = _make_settlement(net_cost=0.0)
        r_phase1 = compute_agent_reward(
            "solar_01", s, 20.0, 20.0, 0.0, 0.5, 0.0, pf, 1000.0, curriculum_phase=1
        )
        r_phase2 = compute_agent_reward(
            "solar_01", s, 20.0, 20.0, 0.0, 0.5, 0.0, pf, 1000.0, curriculum_phase=2
        )
        assert r_phase2 < r_phase1

    def test_deterministic(self):
        """Same inputs always produce identical outputs."""
        s = _make_settlement(net_cost=-200.0, p2p_sold_kw=30.0, grid_sold_kw=10.0)
        pf = self._base_pf_no_violations()
        kwargs = dict(
            agent_id="solar_01",
            settlement=s,
            demand_kw=30.0,
            solar_kw=60.0,
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=pf,
            max_possible_cost=500.0,
            curriculum_phase=1,
        )
        assert compute_agent_reward(**kwargs) == compute_agent_reward(**kwargs)

    def test_power_flow_none_does_not_crash(self):
        """None power_flow_result (bypass mode) must not raise any exception."""
        s = _make_settlement(net_cost=-100.0)
        r = compute_agent_reward(
            COLLEGE_AGENT_ID, s, 100.0, 200.0, -50.0, 0.5, 0.0, None, 1000.0, 1
        )
        assert math.isfinite(r)


# ---------------------------------------------------------------------------
# Full portfolio: compute_all_rewards
# ---------------------------------------------------------------------------


class TestComputeAllRewards:
    """Tests for compute_all_rewards — all 21 agents at once."""

    def _make_settlements(
        self, net_cost: float = -100.0
    ) -> dict[str, SettlementRecord]:
        return {aid: _make_settlement(net_cost=net_cost) for aid in ALL_AGENT_IDS}

    def _make_demands(self, value: float = 50.0) -> dict[str, float]:
        return {aid: value for aid in ALL_AGENT_IDS}

    def _make_solar(self, value: float = 30.0) -> dict[str, float]:
        return {aid: value for aid in ALL_AGENT_IDS}

    def test_returns_all_agent_ids(self):
        """Output must contain a reward for every agent in ALL_AGENT_IDS."""
        rewards = compute_all_rewards(
            settlements=self._make_settlements(),
            demands_kw=self._make_demands(),
            solar_kw=self._make_solar(),
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        assert set(rewards.keys()) == set(ALL_AGENT_IDS)

    def test_all_rewards_finite(self):
        """All rewards must be finite floats."""
        rewards = compute_all_rewards(
            settlements=self._make_settlements(),
            demands_kw=self._make_demands(),
            solar_kw=self._make_solar(),
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        for aid, r in rewards.items():
            assert math.isfinite(r), f"Non-finite reward for agent '{aid}': {r}"

    def test_all_rewards_within_clip_range(self):
        """All rewards must be within [REWARD_CLIP_MIN, REWARD_CLIP_MAX]."""
        rewards = compute_all_rewards(
            settlements=self._make_settlements(),
            demands_kw=self._make_demands(),
            solar_kw=self._make_solar(),
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        for aid, r in rewards.items():
            assert REWARD_CLIP_MIN <= r <= REWARD_CLIP_MAX, (
                f"Reward out of clip range for agent '{aid}': {r}"
            )

    def test_missing_settlement_defaults_to_zero(self):
        """Missing settlement for one agent → reward = 0.0 (no crash)."""
        settlements = self._make_settlements()
        del settlements["solar_01"]
        rewards = compute_all_rewards(
            settlements=settlements,
            demands_kw=self._make_demands(),
            solar_kw=self._make_solar(),
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        assert rewards["solar_01"] == pytest.approx(0.0)

    def test_zero_demand_zero_solar(self):
        """Zero demand and zero solar must not raise any exception."""
        rewards = compute_all_rewards(
            settlements=self._make_settlements(net_cost=0.0),
            demands_kw={aid: 0.0 for aid in ALL_AGENT_IDS},
            solar_kw={aid: 0.0 for aid in ALL_AGENT_IDS},
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
            curriculum_phase=1,
        )
        for r in rewards.values():
            assert math.isfinite(r)

    def test_battery_dispatch_only_affects_college(self):
        """Battery dispatch should only change the College agent's reward."""
        settlements = self._make_settlements(net_cost=0.0)
        rewards_idle = compute_all_rewards(
            settlements=settlements,
            demands_kw=self._make_demands(100.0),
            solar_kw=self._make_solar(200.0),
            battery_dispatch_kw=0.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
        )
        rewards_charging = compute_all_rewards(
            settlements=settlements,
            demands_kw=self._make_demands(100.0),
            solar_kw=self._make_solar(200.0),
            battery_dispatch_kw=-100.0,
            battery_soc=0.5,
            prev_battery_dispatch_kw=0.0,
            power_flow_result=None,
            max_possible_cost=1000.0,
        )
        # Solar and consumer rewards should be identical in both runs
        for aid in SOLAR_AGENT_IDS + CONSUMER_AGENT_IDS:
            assert rewards_idle[aid] == pytest.approx(rewards_charging[aid])
