"""Battery reward components for the P2P Energy Trading reward system.

Implements three battery-specific shaping terms, applied exclusively to
the College agent:

* ``r_soc``     — SoC health penalty (out-of-bounds penalty, normalised)
* ``r_cycling`` — excessive charge/discharge switching penalty (normalised)
* ``r_storage`` — intelligent charge/discharge incentive

All components are either zero or negative (penalties) or zero or positive
(incentives), consistent with the reward hierarchy in §6 of the
specification.

Decision 2 (SoC): normalised by boundary distance, maximum penalty = -w_soc.
Decision 3 (Cycling): normalised by BATTERY_POWER_KW; never hardcoded.

Design reference: docs/module_5_reward_system.md §2, §5
"""

from __future__ import annotations

# standard library
import logging
import math

# local
from p2p_energy_trading.constants import (
    BATTERY_MIN_DISPATCH_KW,
    BATTERY_POWER_KW,
    BATTERY_SOC_MAX,
    BATTERY_SOC_MIN,
    EPSILON,
    REWARD_W_CYCLING,
    REWARD_W_SOC,
    REWARD_W_STORAGE,
)

logger = logging.getLogger(__name__)

# SoC penalty normalisers: distance from the corresponding bound.
# A battery fully at 0.0 yields −w_soc × 0.10 / 0.10 = −w_soc.
# A battery fully at 1.0 yields −w_soc × 0.05 / 0.05 = −w_soc.
_SOC_LOW_BOUNDARY: float = BATTERY_SOC_MIN  # 0.10
_SOC_HIGH_BOUNDARY: float = BATTERY_SOC_MAX  # 0.95
_SOC_LOW_RANGE: float = BATTERY_SOC_MIN  # 0.10 (range from 0.0 to min)
_SOC_HIGH_RANGE: float = 1.0 - BATTERY_SOC_MAX  # 0.05 (range from max to 1.0)


def compute_soc_penalty(
    battery_soc: float,
    w_soc: float = REWARD_W_SOC,
) -> float:
    """Compute the normalised battery SoC health penalty.

    Returns zero when SoC is within the safe operating band
    [``BATTERY_SOC_MIN``, ``BATTERY_SOC_MAX``] = [0.10, 0.95].  Outside
    this band, the penalty scales linearly with the violation depth,
    normalised so that the maximum achievable penalty is exactly ``−w_soc``.

    Formula (Decision 2 — §5 normalised form):

        if SoC < 0.10:
            r_soc = −w_soc × (0.10 − SoC) / 0.10
        elif SoC > 0.95:
            r_soc = −w_soc × (SoC − 0.95) / 0.05
        else:
            r_soc = 0.0

    Args:
        battery_soc: Current State of Charge in [0.0, 1.0].
        w_soc: SoC penalty weight (default ``REWARD_W_SOC = 1.0``).

    Returns:
        Non-positive SoC penalty in [−w_soc, 0.0].
    """
    if math.isnan(battery_soc):
        logger.warning("NaN battery SoC detected. r_soc defaults to 0.0.")
        return 0.0

    if battery_soc < _SOC_LOW_BOUNDARY:
        # Violation below minimum (0.10)
        violation_depth = _SOC_LOW_BOUNDARY - battery_soc
        r_soc = -w_soc * (violation_depth / _SOC_LOW_RANGE)
    elif battery_soc > _SOC_HIGH_BOUNDARY:
        # Violation above maximum (0.95)
        violation_depth = battery_soc - _SOC_HIGH_BOUNDARY
        r_soc = -w_soc * (violation_depth / _SOC_HIGH_RANGE)
    else:
        r_soc = 0.0

    return float(r_soc)


def compute_cycling_penalty(
    battery_dispatch_kw: float,
    prev_dispatch_kw: float,
    w_cyc: float = REWARD_W_CYCLING,
) -> float:
    """Compute the normalised excessive cycling penalty.

    Penalises rapid charge-to-discharge or discharge-to-charge reversals.
    A direction switch is detected when the current dispatch has the
    opposite sign to the previous dispatch and both are non-zero (above
    the minimum dispatch threshold ``BATTERY_MIN_DISPATCH_KW``).

    Formula (Decision 3 — §5 normalised form):

        if direction switch detected:
            r_cycling = −w_cyc × |dispatch_power_kw| / BATTERY_POWER_KW
        else:
            r_cycling = 0.0

    The denominator uses the ``BATTERY_POWER_KW`` constant (250 kW);
    the literal 250.0 is never hardcoded.

    Args:
        battery_dispatch_kw: Current dispatch in kW
            (positive = discharging, negative = charging).
        prev_dispatch_kw: Dispatch from the previous timestep in kW.
        w_cyc: Cycling penalty weight
            (default ``REWARD_W_CYCLING = 0.5``).

    Returns:
        Non-positive cycling penalty in [−w_cyc, 0.0].
    """
    if math.isnan(battery_dispatch_kw) or math.isnan(prev_dispatch_kw):
        logger.warning(
            "NaN battery dispatch values detected. r_cycling defaults to 0.0."
        )
        return 0.0

    # A direction switch requires both steps to have meaningful power flow
    # (above the minimum dispatch threshold) and opposite signs.
    prev_is_active = abs(prev_dispatch_kw) >= BATTERY_MIN_DISPATCH_KW
    curr_is_active = abs(battery_dispatch_kw) >= BATTERY_MIN_DISPATCH_KW

    if not (prev_is_active and curr_is_active):
        return 0.0

    # Signs differ: one positive (discharge) and one negative (charge)
    direction_switched = (prev_dispatch_kw * battery_dispatch_kw) < 0.0

    if not direction_switched:
        return 0.0

    r_cycling = -w_cyc * abs(battery_dispatch_kw) / BATTERY_POWER_KW
    return float(r_cycling)


def compute_storage_reward(
    battery_dispatch_kw: float,
    own_surplus_kw: float,
    own_deficit_kw: float,
    w_store: float = REWARD_W_STORAGE,
) -> float:
    """Compute the intelligent battery charge/discharge incentive.

    Rewards the College agent for charging during surplus periods and
    discharging during deficit periods, provided the dispatch magnitude
    exceeds the minimum dispatch threshold (``BATTERY_MIN_DISPATCH_KW``
    = 25 kW = 10 % of 250 kW rating).

    Formula (§5):

        min_dispatch = BATTERY_MIN_DISPATCH_KW  (25 kW)

        if own_surplus > 0 AND charging AND |dispatch| >= min_dispatch:
            r_storage = w_store × min(charge_kw, own_surplus) / max(own_surplus, ε)

        elif own_deficit > 0 AND discharging AND |dispatch| >= min_dispatch:
            r_storage = w_store × min(discharge_kw, own_deficit) / max(own_deficit, ε)

        else:
            r_storage = 0.0

    Args:
        battery_dispatch_kw: Actual dispatch in kW
            (positive = discharging, negative = charging).
        own_surplus_kw: Available surplus before market clearing (kW).
        own_deficit_kw: Available deficit before market clearing (kW).
        w_store: Storage incentive weight
            (default ``REWARD_W_STORAGE = 0.05``).

    Returns:
        Non-negative storage reward in [0, w_store].
    """
    if (
        math.isnan(battery_dispatch_kw)
        or math.isnan(own_surplus_kw)
        or math.isnan(own_deficit_kw)
    ):
        logger.warning("NaN value detected in r_storage inputs. Falling back to 0.0.")
        return 0.0

    dispatch_magnitude = abs(battery_dispatch_kw)

    # Below minimum dispatch threshold: no storage incentive
    if dispatch_magnitude < BATTERY_MIN_DISPATCH_KW:
        return 0.0

    is_charging = battery_dispatch_kw < 0.0
    is_discharging = battery_dispatch_kw > 0.0

    if is_charging and own_surplus_kw > 0.0:
        # Charging during solar surplus: reward proportional to surplus absorbed
        charge_kw = dispatch_magnitude
        effective_charge = min(charge_kw, own_surplus_kw)
        r_storage = w_store * (effective_charge / max(own_surplus_kw, EPSILON))
        return float(r_storage)

    if is_discharging and own_deficit_kw > 0.0:
        # Discharging during demand deficit: reward proportional to deficit served
        discharge_kw = dispatch_magnitude
        effective_discharge = min(discharge_kw, own_deficit_kw)
        r_storage = w_store * (effective_discharge / max(own_deficit_kw, EPSILON))
        return float(r_storage)

    return 0.0
