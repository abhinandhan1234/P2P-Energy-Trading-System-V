"""Constraint checking for PandaPower power flow results.

This module evaluates voltages, line loadings, and transformer loadings against
defined limits to detect safety violations and calculate metrics.

Design reference: docs/module_2_pandapower_network.md
"""

from __future__ import annotations

# standard library
from dataclasses import dataclass

# local
from p2p_energy_trading.constants import (
    GRID_IMPORT_EXPORT_LIMIT_KW,
    LINE_LOADING_MAX_PERCENT,
    NUM_BUSES,
    TRANSFORMER_LOADING_MAX_PERCENT,
    VOLTAGE_MAX_PU,
    VOLTAGE_MIN_PU,
)
from p2p_energy_trading.modules.network.powerflow import PowerFlowResult


@dataclass
class ConstraintViolations:
    """Constraint violation flags and severity metrics."""

    voltage_violation: bool
    thermal_violation: bool
    transformer_violation: bool
    voltage_min_pu: float  # worst-case minimum primary voltage
    voltage_max_pu: float  # worst-case maximum primary voltage
    line_loading_max_pct: float  # worst-case line loading (%)
    trafo_loading_max_pct: float  # worst-case transformer loading (%)


def check_constraints(result: PowerFlowResult) -> ConstraintViolations:
    """Check all constraint violations from a power flow result.

    Voltage: 0.95 <= V <= 1.05 p.u. (VOLTAGE_MIN_PU / VOLTAGE_MAX_PU)
    Thermal: line loading <= 100% (LINE_LOADING_MAX_PERCENT)
    Transformer: loading <= 100% (TRANSFORMER_LOADING_MAX_PERCENT)

    Only primary buses (0 to NUM_BUSES-1) are considered for the voltage violation
    and worst-case voltage metrics, consistent with feeder constraints.

    Args:
        result: The PowerFlowResult dataclass from a converged run.

    Returns:
        ConstraintViolations dataclass containing violation flags and worst-case values.
    """
    # 1. Voltage check (primary buses only)
    primary_voltages = [
        result.bus_vm_pu[i] for i in range(NUM_BUSES) if i in result.bus_vm_pu
    ]

    if primary_voltages:
        voltage_min_pu = min(primary_voltages)
        voltage_max_pu = max(primary_voltages)
    else:
        # Fallbacks if dictionary is empty
        voltage_min_pu = 1.0
        voltage_max_pu = 1.0

    voltage_violation = (
        voltage_min_pu < VOLTAGE_MIN_PU or voltage_max_pu > VOLTAGE_MAX_PU
    )

    # 2. Thermal check (lines)
    line_loadings = list(result.line_loading_pct.values())
    line_loading_max_pct = max(line_loadings) if line_loadings else 0.0
    thermal_violation = line_loading_max_pct > LINE_LOADING_MAX_PERCENT

    # 3. Transformer check
    trafo_loadings = list(result.trafo_loading_pct.values())
    trafo_loading_max_pct = max(trafo_loadings) if trafo_loadings else 0.0
    transformer_violation = trafo_loading_max_pct > TRANSFORMER_LOADING_MAX_PERCENT

    return ConstraintViolations(
        voltage_violation=voltage_violation,
        thermal_violation=thermal_violation,
        transformer_violation=transformer_violation,
        voltage_min_pu=voltage_min_pu,
        voltage_max_pu=voltage_max_pu,
        line_loading_max_pct=line_loading_max_pct,
        trafo_loading_max_pct=trafo_loading_max_pct,
    )


def check_grid_import_limit(p_grid_kw: float) -> bool:
    """Return True if the grid import/export exceeds +/-2 MW.

    Args:
        p_grid_kw: Net active power imported from grid in kW.

    Returns:
        True if import/export limit is violated.
    """
    return abs(p_grid_kw) > GRID_IMPORT_EXPORT_LIMIT_KW
