"""Market Engine — Module 3.

Uniform clearing P2P energy market with pro-rata allocation.

Reference: docs/module_3_market_engine.md
"""

from __future__ import annotations

# local
from p2p_energy_trading.modules.market.clearing import clear_market_p2p
from p2p_energy_trading.modules.market.models import MarketState, SettlementRecord
from p2p_energy_trading.modules.market.settlement import process_settlements

__all__ = [
    "clear_market_p2p",
    "process_settlements",
    "SettlementRecord",
    "MarketState",
]
