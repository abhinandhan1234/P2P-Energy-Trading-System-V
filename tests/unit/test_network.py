"""Unit tests for the PandaPower network builder and power flow solver (Module 2).

Design reference: docs/module_2_pandapower_network.md
"""

from __future__ import annotations

# third party
import pandapower as pp
import pytest

# local
from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    COLLEGE_AGENT_ID,
    CONSUMER_AGENT_IDS,
    GRID_IMPORT_EXPORT_LIMIT_KW,
    NUM_BUSES,
    SOLAR_AGENT_IDS,
    VOLTAGE_MAX_PU,
    VOLTAGE_MIN_PU,
)
from p2p_energy_trading.exceptions import PowerFlowError
from p2p_energy_trading.modules.network.constraints import (
    check_constraints,
    check_grid_import_limit,
)
from p2p_energy_trading.modules.network.network_builder import (
    build_network,
    get_agent_bus_index,
    get_load_index,
    get_sgen_index,
)
from p2p_energy_trading.modules.network.powerflow import (
    PowerFlowResult,
    run_power_flow,
    update_network_loads,
)


class TestNetworkBuilder:
    """Verify that build_network constructs the topology correctly."""

    def test_build_network_structure(self):
        """Check the number of elements created in the PandaPower network."""
        net = build_network()
        assert isinstance(net, pp.pandapowerNet)

        # Expected buses: 33 primary + 1 utility HV slack + 21 agent secondary
        # = 55 buses
        assert len(net.bus) == 55

        # Expected lines: exactly 32 primary feeder lines
        assert len(net.line) == 32

        # Expected transformers: 1 substation + 21 agent step-down = 22 transformers
        assert len(net.trafo) == 22

        # Expected loads: exactly 21 loads (one per agent)
        assert len(net.load) == 21

        # Expected sgens: 1 college + 15 solar = 16 sgens
        assert len(net.sgen) == 16

        # Expected external grid (slack generator connection) at the HV utility bus
        assert len(net.ext_grid) == 1

    def test_agent_bus_indexes(self):
        """Verify get_agent_bus_index works correctly."""
        for aid in ALL_AGENT_IDS:
            bus = get_agent_bus_index(aid)
            assert isinstance(bus, int)
            assert 0 <= bus < NUM_BUSES

        with pytest.raises(PowerFlowError):
            get_agent_bus_index("invalid_agent")

    def test_load_indexes(self):
        """Verify load indices are unique and valid."""
        load_idxs = set()
        for aid in ALL_AGENT_IDS:
            idx = get_load_index(aid)
            assert isinstance(idx, int)
            load_idxs.add(idx)

        assert len(load_idxs) == len(ALL_AGENT_IDS)

        with pytest.raises(PowerFlowError):
            get_load_index("invalid_agent")

    def test_sgen_indexes(self):
        """Verify sgen indices exist for college/solar, and raise error for
        consumers."""
        sgen_idxs = set()
        for aid in SOLAR_AGENT_IDS + [COLLEGE_AGENT_ID]:
            idx = get_sgen_index(aid)
            assert isinstance(idx, int)
            sgen_idxs.add(idx)

        assert len(sgen_idxs) == len(SOLAR_AGENT_IDS) + 1

        for aid in CONSUMER_AGENT_IDS:
            with pytest.raises(PowerFlowError):
                get_sgen_index(aid)


class TestPowerFlow:
    """Verify load updates and solver convergence."""

    def test_update_loads(self):
        """Verify update_network_loads updates p_mw values in the network."""
        net = build_network()

        # Build mock demands and solar
        demands = {aid: 50.0 for aid in ALL_AGENT_IDS}  # 50 kW each
        solars = {
            aid: 20.0 for aid in SOLAR_AGENT_IDS + [COLLEGE_AGENT_ID]
        }  # 20 kW each
        battery_dispatch = 100.0  # 100 kW discharge

        update_network_loads(net, demands, solars, battery_dispatch)

        # Check loads
        for aid in ALL_AGENT_IDS:
            idx = get_load_index(aid)
            assert net.load.at[idx, "p_mw"] == pytest.approx(0.05)  # 50 kW / 1000

        # Check sgens
        for aid in SOLAR_AGENT_IDS:
            idx = get_sgen_index(aid)
            assert net.sgen.at[idx, "p_mw"] == pytest.approx(0.02)  # 20 kW / 1000

        college_sgen_idx = get_sgen_index(COLLEGE_AGENT_ID)
        # College solar (20 kW) + battery discharge (100 kW) = 120 kW -> 0.12 MW
        assert net.sgen.at[college_sgen_idx, "p_mw"] == pytest.approx(0.12)

    def test_nominal_power_flow_convergence(self):
        """Run power flow with normal load values and ensure it converges."""
        net = build_network()
        demands = {aid: 10.0 for aid in ALL_AGENT_IDS}
        solars = {aid: 5.0 for aid in SOLAR_AGENT_IDS + [COLLEGE_AGENT_ID]}

        update_network_loads(net, demands, solars, battery_dispatch_kw=0.0)
        res = run_power_flow(net)

        assert isinstance(res, PowerFlowResult)
        assert res.converged is True
        assert isinstance(res.bus_vm_pu, dict)
        assert isinstance(res.line_loading_pct, dict)
        assert isinstance(res.trafo_loading_pct, dict)
        assert isinstance(res.p_grid_kw, float)

        # Voltage check (should be close to nominal/slack)
        for bus_idx in range(NUM_BUSES):
            assert 0.90 < res.bus_vm_pu[bus_idx] < 1.10

    def test_power_flow_divergence_handling(self):
        """Check that extreme load values trigger a PowerFlowError on solver failure."""
        net = build_network()
        # Inject an physically impossible load of 1,000,000 kW (1,000 MW) at
        # a single bus
        demands = {ALL_AGENT_IDS[0]: 1e6}
        update_network_loads(net, demands, {}, 0.0)

        with pytest.raises(PowerFlowError):
            run_power_flow(net, max_retries=2)


class TestConstraintChecks:
    """Verify constraint violation detection."""

    def test_check_constraints_nominal(self):
        """Nominal results should not trigger any violations."""
        # Create a mock result where all values are healthy
        bus_vm = {i: 1.01 for i in range(NUM_BUSES)}
        line_loading = {i: 40.0 for i in range(32)}
        trafo_loading = {i: 50.0 for i in range(22)}

        res = PowerFlowResult(
            converged=True,
            bus_vm_pu=bus_vm,
            line_loading_pct=line_loading,
            trafo_loading_pct=trafo_loading,
            p_grid_kw=100.0,
        )

        violations = check_constraints(res)
        assert violations.voltage_violation is False
        assert violations.thermal_violation is False
        assert violations.transformer_violation is False
        assert violations.voltage_min_pu == 1.01
        assert violations.voltage_max_pu == 1.01
        assert violations.line_loading_max_pct == 40.0
        assert violations.trafo_loading_max_pct == 50.0

    def test_voltage_violations(self):
        """Voltages outside [0.95, 1.05] should trigger voltage_violation."""
        # Under-voltage
        bus_vm_low = {i: 1.0 for i in range(NUM_BUSES)}
        bus_vm_low[10] = VOLTAGE_MIN_PU - 0.01  # 0.94 p.u.
        res_low = PowerFlowResult(
            converged=True,
            bus_vm_pu=bus_vm_low,
            line_loading_pct={},
            trafo_loading_pct={},
            p_grid_kw=0.0,
        )
        assert check_constraints(res_low).voltage_violation is True
        assert check_constraints(res_low).voltage_min_pu == pytest.approx(
            VOLTAGE_MIN_PU - 0.01
        )

        # Over-voltage
        bus_vm_high = {i: 1.0 for i in range(NUM_BUSES)}
        bus_vm_high[15] = VOLTAGE_MAX_PU + 0.01  # 1.06 p.u.
        res_high = PowerFlowResult(
            converged=True,
            bus_vm_pu=bus_vm_high,
            line_loading_pct={},
            trafo_loading_pct={},
            p_grid_kw=0.0,
        )
        assert check_constraints(res_high).voltage_violation is True
        assert check_constraints(res_high).voltage_max_pu == pytest.approx(
            VOLTAGE_MAX_PU + 0.01
        )

    def test_thermal_violations(self):
        """Line loading above 100% should trigger thermal_violation."""
        line_loading = {i: 50.0 for i in range(32)}
        line_loading[5] = 101.5  # Overload
        res = PowerFlowResult(
            converged=True,
            bus_vm_pu={i: 1.0 for i in range(NUM_BUSES)},
            line_loading_pct=line_loading,
            trafo_loading_pct={},
            p_grid_kw=0.0,
        )
        violations = check_constraints(res)
        assert violations.thermal_violation is True
        assert violations.line_loading_max_pct == 101.5

    def test_transformer_violations(self):
        """Transformer loading above 100% should trigger transformer_violation."""
        trafo_loading = {i: 80.0 for i in range(22)}
        trafo_loading[1] = 105.0  # Overload
        res = PowerFlowResult(
            converged=True,
            bus_vm_pu={i: 1.0 for i in range(NUM_BUSES)},
            line_loading_pct={},
            trafo_loading_pct=trafo_loading,
            p_grid_kw=0.0,
        )
        violations = check_constraints(res)
        assert violations.transformer_violation is True
        assert violations.trafo_loading_max_pct == 105.0


class TestGridImportLimit:
    """Verify grid active power import limit checks."""

    def test_within_limits(self):
        assert check_grid_import_limit(0.0) is False
        assert check_grid_import_limit(GRID_IMPORT_EXPORT_LIMIT_KW - 1.0) is False
        assert check_grid_import_limit(-GRID_IMPORT_EXPORT_LIMIT_KW + 1.0) is False

    def test_outside_limits(self):
        assert check_grid_import_limit(GRID_IMPORT_EXPORT_LIMIT_KW + 1.0) is True
        assert check_grid_import_limit(-GRID_IMPORT_EXPORT_LIMIT_KW - 1.0) is True
