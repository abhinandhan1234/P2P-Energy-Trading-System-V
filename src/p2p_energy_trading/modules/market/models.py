"""Market dataclass definitions for the P2P Energy Trading System.

This module defines the structured data containers for per-agent settlements
and global market clearing states.

Design reference: docs/module_3_market_engine.md
"""

from __future__ import annotations

# standard library
from dataclasses import dataclass


@dataclass(frozen=True)
class SettlementRecord:
    """Per-agent financial and energy allocation record for one timestep.

    All quantities are in kW and Rs.
    """

    p2p_sold_kw: float
    p2p_bought_kw: float
    grid_sold_kw: float
    grid_bought_kw: float
    p2p_price: float
    p2p_revenue: float
    p2p_cost: float
    grid_revenue: float
    grid_cost: float
    net_cost: float


@dataclass(frozen=True)
class MarketState:
    """Global market state and summary observations for one timestep.

    Tracks uniform clearing price, total trading volumes, grid exchange, and
    constraint indicators.
    """

    p2p_clearing_price: float
    total_p2p_volume: float
    p2p_utilisation_ratio: float
    grid_import_total: float
    grid_export_total: float
    voltage_violation: bool
    thermal_violation: bool
    curtailment_applied: bool
    total_bids: float
    total_offers: float
