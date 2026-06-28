"""Grid safety penalty components for the P2P Energy Trading reward system.

Implements three per-agent grid constraint penalties:

* ``r_voltage``    — bus voltage deviation penalty (all agents)
* ``r_thermal``    — line thermal overload penalty (all agents)
* ``r_transformer``— main transformer overload penalty (College agent only)

All penalties are zero when the grid is within safe operating limits and
scale linearly with the magnitude of any constraint violation.  Each
penalty is normalised so that the weight parameter ``w`` equals the
maximum achievable penalty magnitude (consistent with §7 of the reward
specification: "Grid penalties: already normalised by violation magnitude
denominators").

The agent-to-line index mapping uses ``AGENT_TO_BUS[agent_id] − 1``,
which is factually correct for the sequential IEEE 33-bus radial topology
implemented in ``modules/network/network_builder.py``.

Design reference: docs/module_5_reward_system.md §2, §3
"""

from __future__ import annotations

# standard library
import logging
import math

# local
from p2p_energy_trading.constants import (
    AGENT_TO_BUS,
    REWARD_W_THERMAL_PHASE1,
    REWARD_W_THERMAL_PHASE2,
    REWARD_W_TRANSFORMER_PHASE1,
    REWARD_W_TRANSFORMER_PHASE2,
    REWARD_W_VOLTAGE_PHASE1,
    REWARD_W_VOLTAGE_PHASE2,
)
from p2p_energy_trading.modules.network.powerflow import PowerFlowResult

logger = logging.getLogger(__name__)

# Voltage penalty denominator: a bus at exactly ±0.10 p.u. outside the safe
# band [0.95, 1.05] produces a normalised violation of 1.0 (§3 line 123).
_VOLTAGE_PENALTY_DENOMINATOR: float = 0.05

# Line loading threshold (fraction, not percent): penalties begin above 1.0.
_LOADING_OVERLOAD_THRESHOLD: float = 1.0


def _get_phase_voltage_weight(curriculum_phase: int) -> float:
    """Return the voltage penalty weight for the given curriculum phase."""
    return REWARD_W_VOLTAGE_PHASE1 if curriculum_phase == 1 else REWARD_W_VOLTAGE_PHASE2


def _get_phase_thermal_weight(curriculum_phase: int) -> float:
    """Return the thermal penalty weight for the given curriculum phase."""
    return REWARD_W_THERMAL_PHASE1 if curriculum_phase == 1 else REWARD_W_THERMAL_PHASE2


def _get_phase_transformer_weight(curriculum_phase: int) -> float:
    """Return the transformer penalty weight for the given curriculum phase."""
    return (
        REWARD_W_TRANSFORMER_PHASE1
        if curriculum_phase == 1
        else REWARD_W_TRANSFORMER_PHASE2
    )


def compute_voltage_penalty(
    power_flow_result: PowerFlowResult | None,
    agent_id: str,
    curriculum_phase: int = 1,
    w_v: float | None = None,
) -> float:
    """Compute the per-agent bus voltage deviation penalty.

    Returns zero when the bus voltage is within the safe band [0.95, 1.05]
    p.u.  When outside, the penalty scales linearly with deviation magnitude,
    normalised so that a bus at 0.90 p.u. or 1.10 p.u. yields exactly
    ``−w_v``.

    Formula (§2/§3):

        deviation = max(0, |V_bus − 1.0| − 0.05)
        r_voltage  = −w_v × (deviation / 0.05)

    Args:
        power_flow_result: Power flow result from Module 2.  If ``None``
            (bypass mode), returns 0.0.
        agent_id: Agent identifier string.
        curriculum_phase: Training phase (1 = exploration, 2 =
            constraint-aware).  Selects ``w_v``.
        w_v: Override voltage penalty weight.  If ``None``, the phase-
            appropriate constant from ``constants.py`` is used.

    Returns:
        Non-positive voltage penalty.  Zero when no violation.
    """
    if power_flow_result is None:
        return 0.0

    if w_v is None:
        w_v = _get_phase_voltage_weight(curriculum_phase)

    bus_idx = AGENT_TO_BUS.get(agent_id)
    if bus_idx is None:
        logger.warning(
            "Agent '%s' not in AGENT_TO_BUS mapping; r_voltage defaults to 0.0.",
            agent_id,
        )
        return 0.0

    v_pu = power_flow_result.bus_vm_pu.get(bus_idx, 1.0)

    if math.isnan(v_pu):
        logger.warning(
            "NaN bus voltage for agent '%s' (bus %d). r_voltage defaults to 0.0.",
            agent_id,
            bus_idx,
        )
        return 0.0

    deviation = max(0.0, abs(v_pu - 1.0) - 0.05)
    r_voltage = -w_v * (deviation / _VOLTAGE_PENALTY_DENOMINATOR)
    return float(r_voltage)


def compute_thermal_penalty(
    power_flow_result: PowerFlowResult | None,
    agent_id: str,
    curriculum_phase: int = 1,
    w_th: float | None = None,
) -> float:
    """Compute the per-agent line thermal overload penalty.

    Returns zero when line loading is at or below 100 % (loading fraction
    ≤ 1.0).  When overloaded, the penalty scales linearly with the excess
    fraction above 1.0.

    The connecting-line index for agent at primary bus ``B`` is
    ``B − 1``, which is correct for the sequential IEEE 33-bus radial
    topology (verified against ``network_builder.py`` lines 102–135).

    ``PowerFlowResult.line_loading_pct`` stores values in percent (e.g.
    115.0 for 115 %).  This function converts to a fraction before
    applying the formula.

    Formula (§2/§3):

        loading = line_loading_pct / 100.0
        r_thermal = −w_th × max(0, loading − 1.0)

    Args:
        power_flow_result: Power flow result from Module 2.  If ``None``
            (bypass mode), returns 0.0.
        agent_id: Agent identifier string.
        curriculum_phase: Training phase (1 or 2).  Selects ``w_th``.
        w_th: Override thermal penalty weight.  If ``None``, the phase-
            appropriate constant is used.

    Returns:
        Non-positive thermal penalty.  Zero when no overload.
    """
    if power_flow_result is None:
        return 0.0

    if w_th is None:
        w_th = _get_phase_thermal_weight(curriculum_phase)

    bus_idx = AGENT_TO_BUS.get(agent_id)
    if bus_idx is None:
        logger.warning(
            "Agent '%s' not in AGENT_TO_BUS mapping; r_thermal defaults to 0.0.",
            agent_id,
        )
        return 0.0

    # Connecting line index: for bus B in this radial topology, the line
    # whose to_bus == B has index B − 1.
    line_idx = bus_idx - 1
    loading_pct = power_flow_result.line_loading_pct.get(line_idx, 0.0)

    if math.isnan(loading_pct):
        logger.warning(
            "NaN line loading for agent '%s' (line %d). r_thermal defaults to 0.0.",
            agent_id,
            line_idx,
        )
        return 0.0

    loading_fraction = loading_pct / 100.0
    overload = max(0.0, loading_fraction - _LOADING_OVERLOAD_THRESHOLD)
    r_thermal = -w_th * overload
    return float(r_thermal)


def compute_transformer_penalty(
    power_flow_result: PowerFlowResult | None,
    curriculum_phase: int = 1,
    w_tr: float | None = None,
) -> float:
    """Compute the College main transformer overload penalty.

    Only applies to the College agent (Bus 7, 500 kVA transformer at
    PandaPower transformer index 0).  Solar and consumer buildings use
    individual distribution transformers whose overloading is captured by
    line loading penalties (§3 line 151).

    Formula (§2/§3):

        loading = trafo_loading_pct / 100.0
        r_transformer = −w_tr × max(0, loading − 1.0)

    Args:
        power_flow_result: Power flow result from Module 2.  If ``None``
            (bypass mode), returns 0.0.
        curriculum_phase: Training phase (1 or 2).  Selects ``w_tr``.
        w_tr: Override transformer penalty weight.  If ``None``, the
            phase-appropriate constant is used.

    Returns:
        Non-positive transformer penalty.  Zero when no overload.
    """
    if power_flow_result is None:
        return 0.0

    if w_tr is None:
        w_tr = _get_phase_transformer_weight(curriculum_phase)

    # Substation transformer is at index 0 in the PandaPower trafo table.
    # Index 1 onward are agent-level distribution transformers.
    trafo_loading_pct = power_flow_result.trafo_loading_pct.get(0, 0.0)

    if math.isnan(trafo_loading_pct):
        logger.warning(
            "NaN transformer loading (index 0). r_transformer defaults to 0.0."
        )
        return 0.0

    loading_fraction = trafo_loading_pct / 100.0
    overload = max(0.0, loading_fraction - _LOADING_OVERLOAD_THRESHOLD)
    r_transformer = -w_tr * overload
    return float(r_transformer)
