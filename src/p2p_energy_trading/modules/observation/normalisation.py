"""Normalization utilities for the P2P Energy Trading observation vectors.

Converts raw physical units (kW, kWh, p.u., Rs) into standard normalized ranges
suitable for RL policy network inputs.

Design reference: docs/module_4_observation_builder.md
"""

from __future__ import annotations

# standard library
import math

# third party
import numpy as np

# local
from p2p_energy_trading.constants import MAX_GRID_RATE


def normalise_energy(value: float, peak: float) -> float:
    """Normalize demand or solar generation against peak capacity.

    Args:
        value: Raw power value (kW).
        peak: Peak power capacity (kW).

    Returns:
        Normalized power value in [0.0, 1.0]. Returns 0.0 if peak is 0.
    """
    if peak <= 0.0:
        return 0.0
    val = float(value / peak)
    return float(np.clip(val, 0.0, 1.0))


def normalise_voltage(v_pu: float) -> float:
    """Normalize bus per-unit voltage magnitude.

    Uses min-max scaling between the safety bounds [0.95, 1.05] p.u.

    Args:
        v_pu: Per-unit voltage magnitude.

    Returns:
        Normalized voltage in [0.0, 1.0].
    """
    norm = (v_pu - 0.95) / 0.10
    return float(np.clip(norm, 0.0, 1.0))


def normalise_loading(loading_pct: float) -> float:
    """Normalize percentage thermal loading of lines or transformers.

    Args:
        loading_pct: loading in percentage (e.g. 85.0%).

    Returns:
        Normalized loading fraction.
    """
    return float(
        np.clip(
            loading_pct / 100.0,
            0.0,
            1.0,
        )
    )


def normalise_grid_flow(p_grid_kw: float, limit_kw: float) -> float:
    """Normalize net active grid flow against HESCOM import/export limit.

    Args:
        p_grid_kw: Net grid import (kW, positive = import, negative = export).
        limit_kw: Grid import/export limit (kW).

    Returns:
        Normalized grid flow in [-1.0, 1.0].
    """
    if limit_kw <= 0.0:
        return 0.0
    val = float(p_grid_kw / limit_kw)
    return float(np.clip(val, -1.0, 1.0))


def normalise_p2p_price(price: float, grid_buy: float, grid_sell: float) -> float:
    """Normalize uniform P2P price relative to grid buy and sell bounds.

    Args:
        price: Uniform P2P price (Rs/kWh).
        grid_buy: Grid buy rate (Rs/kWh).
        grid_sell: Grid sell rate (Rs/kWh).

    Returns:
        Normalized price in [0.0, 1.0]. Defaults to 0.5 if bounds are equal.
    """
    denom = grid_buy - grid_sell
    if denom <= 0.0:
        return 0.5
    val = float((price - grid_sell) / denom)
    return float(np.clip(val, 0.0, 1.0))


def normalise_grid_price(price: float) -> float:
    """Normalize HESCOM grid rates against the system-wide maximum price limit.

    Args:
        price: Grid rate (Rs/kWh).

    Returns:
        Normalized rate in [0.0, 1.0].
    """
    return float(
        np.clip(
            price / MAX_GRID_RATE,
            0.0,
            1.0,
        )
    )


def compute_demand_ratio(total_bids: float, total_offers: float) -> float:
    """Compute the normalized market demand ratio (bids / offers).

    Args:
        total_bids: Aggregated buy bids in market (kW).
        total_offers: Aggregated sell offers in market (kW).

    Returns:
        Normalized ratio in [0.0, 1.0] where 1.0 represents bid count >= 2x offers.
    """
    if total_offers <= 0.0:
        ratio = 2.0
    else:
        ratio = total_bids / total_offers
    return float(np.clip(ratio, 0.0, 2.0) / 2.0)


def cyclical_time_encoding(
    hour: int, day_of_week: int
) -> tuple[float, float, float, float]:
    """Encode hour of day and day of week into sin/cos cyclical features.

    Args:
        hour: Hour of the day (0-23).
        day_of_week: Day of the week (0-6).

    Returns:
        tuple containing:
        - hour_sin
        - hour_cos
        - day_sin
        - day_cos
    """
    hour_sin = math.sin(2.0 * math.pi * hour / 24.0)
    hour_cos = math.cos(2.0 * math.pi * hour / 24.0)
    day_sin = math.sin(2.0 * math.pi * day_of_week / 7.0)
    day_cos = math.cos(2.0 * math.pi * day_of_week / 7.0)
    return hour_sin, hour_cos, day_sin, day_cos
