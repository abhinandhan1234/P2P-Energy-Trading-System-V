"""P2P Market Clearing algorithm.

This module aggregates buy bids and sell offers, calculates the cleared volume,
and performs pro-rata curtailments when supply and demand are unbalanced.

Design reference: docs/module_3_market_engine.md
"""

from __future__ import annotations

# standard library
import logging

logger = logging.getLogger(__name__)


def clear_market_p2p(
    bids_kw: dict[str, float],
    offers_kw: dict[str, float],
) -> tuple[dict[str, float], dict[str, float], float, bool]:
    """Execute P2P market clearing.

    Args:
        bids_kw: Dict mapping agent ID to requested buy quantity (kW).
        offers_kw: Dict mapping agent ID to requested sell quantity (kW).

    Returns:
        tuple containing:
        - dict[str, float]: Cleared P2P buy allocations per agent (kW).
        - dict[str, float]: Cleared P2P sell allocations per agent (kW).
        - float: Total P2P cleared volume (kW).
        - bool: True if curtailment/pro-rata scaling was applied.
    """
    total_demand = sum(bids_kw.values())
    total_supply = sum(offers_kw.values())
    cleared_volume = min(total_supply, total_demand)

    p2p_bought = {aid: 0.0 for aid in bids_kw}
    p2p_sold = {aid: 0.0 for aid in offers_kw}
    curtailment_applied = False

    if cleared_volume <= 0.0:
        return p2p_bought, p2p_sold, 0.0, False

    # Pro-rata allocation logic
    if total_supply > total_demand:
        # Supply surplus -> Sellers are curtailed, buyers clear fully
        curtailment_applied = True
        for aid, bid in bids_kw.items():
            p2p_bought[aid] = bid

        for aid, offer in offers_kw.items():
            if total_supply > 0.0:
                p2p_sold[aid] = (offer / total_supply) * cleared_volume

    elif total_demand > total_supply:
        # Demand surplus -> Buyers are curtailed, sellers clear fully
        curtailment_applied = True
        for aid, offer in offers_kw.items():
            p2p_sold[aid] = offer

        for aid, bid in bids_kw.items():
            if total_demand > 0.0:
                p2p_bought[aid] = (bid / total_demand) * cleared_volume

    else:
        # Balanced market -> Everyone clears fully
        for aid, bid in bids_kw.items():
            p2p_bought[aid] = bid
        for aid, offer in offers_kw.items():
            p2p_sold[aid] = offer

    logger.info(
        f"Cleared P2P market: Volume={cleared_volume:.2f} kW, "
        f"Demand={total_demand:.2f} kW, Supply={total_supply:.2f} kW, "
        f"Curtailment={curtailment_applied}"
    )

    return p2p_bought, p2p_sold, cleared_volume, curtailment_applied
