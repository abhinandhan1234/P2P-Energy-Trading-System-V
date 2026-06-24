"""PandaPower Power Flow solver and load updater.

This module provides functions to update network demand and generation profiles,
run Newton-Raphson power flow calculations with robust convergence retries, and
package the resulting state.

Design reference: docs/module_2_pandapower_network.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
import pandapower as pp
from pandapower import LoadflowNotConverged

from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    COLLEGE_AGENT_ID,
    POWERFLOW_MAX_RETRIES,
    POWERFLOW_DEFAULT_TOLERANCE,
    POWERFLOW_RELAXED_TOLERANCE,
)
from p2p_energy_trading.exceptions import PowerFlowError
from p2p_energy_trading.modules.network.network_builder import get_load_index, get_sgen_index

logger = logging.getLogger(__name__)


@dataclass
class PowerFlowResult:
    """Results extracted from a converged power flow."""
    converged: bool
    bus_vm_pu: dict[int, float]        # bus index → voltage magnitude (p.u.)
    line_loading_pct: dict[int, float]   # line index → loading (%)
    trafo_loading_pct: dict[int, float]  # trafo index → loading (%)
    p_grid_kw: float                    # net grid import (+ = import, - = export)


def update_network_loads(
    net: pp.pandapowerNet,
    demands_kw: dict[str, float],
    solar_kw: dict[str, float],
    battery_dispatch_kw: float,
) -> None:
    """Inject per-timestep demand and generation into the network.

    Battery dispatch at college bus: positive = discharging (injection),
    negative = charging (absorption).

    Args:
        net: The PandaPower network instance.
        demands_kw: Dict mapping agent ID to demand in kW.
        solar_kw: Dict mapping agent ID to solar generation in kW.
        battery_dispatch_kw: College battery dispatch in kW.
    """
    for aid in ALL_AGENT_IDS:
        try:
            load_idx = get_load_index(aid)
            demand_kw = demands_kw.get(aid, 0.0)
            net.load.at[load_idx, "p_mw"] = demand_kw / 1000.0

            if aid == COLLEGE_AGENT_ID or aid.startswith("solar_"):
                sgen_idx = get_sgen_index(aid)
                if aid == COLLEGE_AGENT_ID:
                    # College sgen represents combined solar and battery dispatch
                    college_solar = solar_kw.get(aid, 0.0)
                    net.sgen.at[sgen_idx, "p_mw"] = (college_solar + battery_dispatch_kw) / 1000.0
                else:
                    solar_gen = solar_kw.get(aid, 0.0)
                    net.sgen.at[sgen_idx, "p_mw"] = solar_gen / 1000.0

        except PowerFlowError as e:
            logger.warning(f"Error updating loads for agent {aid}: {e}")


def run_power_flow(
    net: pp.pandapowerNet,
    max_retries: int = POWERFLOW_MAX_RETRIES,
    tolerance: float = POWERFLOW_DEFAULT_TOLERANCE,
) -> PowerFlowResult:
    """Run Newton-Raphson power flow with retry on convergence failure.

    Retries with relaxed tolerance if the first attempt fails.
    Raises PowerFlowError after all retries are exhausted.

    Args:
        net: The PandaPower network instance.
        max_retries: Maximum number of solver attempts.
        tolerance: Default convergence tolerance in MVA.

    Returns:
        PowerFlowResult containing converged status, voltages, loadings, and net grid exchange.

    Raises:
        PowerFlowError: If solver fails to converge on all attempts.
    """
    attempt = 1
    current_tolerance = tolerance

    while attempt <= max_retries:
        try:
            pp.runpp(
                net,
                tolerance_mva=current_tolerance,
                numba=False,
                init="flat" if attempt > 1 else "results",
            )
            if net.converged:
                # Extract results
                bus_vm_pu = net.res_bus.vm_pu.to_dict()
                line_loading_pct = net.res_line.loading_percent.to_dict()
                trafo_loading_pct = net.res_trafo.loading_percent.to_dict()

                # Net grid import active power (+ = import, - = export)
                # PandaPower res_ext_grid.p_mw is positive for import
                p_grid_mw = net.res_ext_grid.p_mw.values[0]
                p_grid_kw = float(p_grid_mw * 1000.0)

                return PowerFlowResult(
                    converged=True,
                    bus_vm_pu=bus_vm_pu,
                    line_loading_pct=line_loading_pct,
                    trafo_loading_pct=trafo_loading_pct,
                    p_grid_kw=p_grid_kw,
                )
        except LoadflowNotConverged:
            logger.warning(
                f"Power flow convergence failed on attempt {attempt}/{max_retries} "
                f"with tolerance {current_tolerance}."
            )

        # Relax tolerance and attempt strategy variation
        attempt += 1
        current_tolerance = POWERFLOW_RELAXED_TOLERANCE

    # If we get here, all retries failed
    raise PowerFlowError("PandaPower Newton-Raphson solver failed to converge after all retries.")
