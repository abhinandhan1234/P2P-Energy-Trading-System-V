"""Trading reward components for the P2P Energy Trading reward system.

Implements three market-participation shaping terms:

* ``r_p2p``   — normalised P2P utilisation bonus (all agents)
* ``r_self``  — self-consumption bonus (College and Solar agents only)
* ``r_import``— grid-import reduction bonus (Consumer agents only)

All components are normalised so that their values lie in [0, w_component]
by design, consistent with §7 of the reward specification.

Design reference: docs/module_5_reward_system.md §2, §4, §8
"""

from __future__ import annotations

# standard library
import logging
import math

# local
from p2p_energy_trading.constants import (
    EPSILON,
    REWARD_W_IMPORT,
    REWARD_W_P2P,
    REWARD_W_SELF,
)
from p2p_energy_trading.modules.market.models import SettlementRecord

logger = logging.getLogger(__name__)


def compute_p2p_reward(
    settlement: SettlementRecord,
    own_surplus_kw: float,
    own_deficit_kw: float,
    agent_id: str,
    w_p2p: float = REWARD_W_P2P,
) -> float:
    """Compute the normalised P2P participation bonus.

    Rewards agents for utilising the peer-to-peer market rather than the
    utility grid.  The traded volume is normalised by the agent's total
    energy need so that the result lies in [0, w_p2p] regardless of scale.

    Formula (Decision 1 — normalised form, §4/§8):

        p2p_traded_kw   = p2p_sold_kw + p2p_bought_kw
        total_energy_need = max(own_surplus_kw, own_deficit_kw)
        r_p2p           = w_p2p × (p2p_traded_kw / max(total_energy_need, ε))

    For consumer agents ``p2p_sold_kw`` is always 0, so the formula
    reduces to ``w_p2p × (p2p_bought_kw / max(own_deficit_kw, ε))``.

    Args:
        settlement: Per-agent settlement record from the Market Engine.
        own_surplus_kw: Available surplus before market clearing (kW).
            For non-college agents: ``max(0, solar_kw − demand_kw)``.
            For college: accounts for battery dispatch.
        own_deficit_kw: Available deficit before market clearing (kW).
            For non-college agents: ``max(0, demand_kw − solar_kw)``.
            For college: accounts for battery dispatch.
        agent_id: Agent identifier string.
        w_p2p: P2P participation weight (default ``REWARD_W_P2P = 0.1``).

    Returns:
        Non-negative P2P reward in [0, w_p2p].
    """
    p2p_traded_kw = settlement.p2p_sold_kw + settlement.p2p_bought_kw
    total_energy_need = max(own_surplus_kw, own_deficit_kw)

    if math.isnan(p2p_traded_kw) or math.isnan(total_energy_need):
        logger.warning(
            "NaN detected in r_p2p inputs for agent '%s'. Falling back to 0.0.",
            agent_id,
        )
        return 0.0

    r_p2p = w_p2p * (p2p_traded_kw / max(total_energy_need, EPSILON))
    return float(r_p2p)


def compute_self_consumption_reward(
    settlement: SettlementRecord,
    solar_kw: float,
    battery_dispatch_kw: float,
    agent_id: str,
    w_self: float = REWARD_W_SELF,
) -> float:
    """Compute the self-consumption bonus for College and Solar agents.

    Rewards agents for consuming their own generated solar power locally
    rather than exporting it.  The fraction of solar used on-site is
    normalised by the agent's total generation.

    Formula (§2/§4):

        solar_used_locally = solar_kw
                             − p2p_sold_kw
                             − grid_sold_kw
                             − max(0, −battery_dispatch_kw)   (college only)
        r_self = w_self × (solar_used_locally / max(solar_kw, ε))

    The ``solar_used_locally`` value is mathematically derived from the
    settlement record fields without modifying Module 3.

    Args:
        settlement: Per-agent settlement record.
        solar_kw: Raw solar generation at this timestep (kW).
        battery_dispatch_kw: Actual battery dispatch at this timestep (kW).
            Positive = discharging; negative = charging.
            Use 0.0 for non-college agents.
        agent_id: Agent identifier string.
        w_self: Self-consumption weight (default ``REWARD_W_SELF = 0.05``).

    Returns:
        Non-negative self-consumption reward in [0, w_self].
    """
    # Battery charge power absorbed from the bus (positive quantity)
    battery_charge_kw = max(0.0, -battery_dispatch_kw)

    # Local solar consumption: what is left after exports and battery charging
    solar_used_locally = (
        solar_kw - settlement.p2p_sold_kw - settlement.grid_sold_kw - battery_charge_kw
    )
    # Clamp floating-point artefacts; self-consumption cannot be negative
    solar_used_locally = max(0.0, solar_used_locally)

    if math.isnan(solar_used_locally) or math.isnan(solar_kw):
        logger.warning(
            "NaN detected in r_self inputs for agent '%s'. Falling back to 0.0.",
            agent_id,
        )
        return 0.0

    r_self = w_self * (solar_used_locally / max(solar_kw, EPSILON))
    return float(r_self)


def compute_import_reduction_reward(
    settlement: SettlementRecord,
    own_deficit_kw: float,
    agent_id: str,
    w_import: float = REWARD_W_IMPORT,
) -> float:
    """Compute the grid-import reduction bonus for Consumer agents.

    Rewards consumers for buying from the P2P market rather than the
    utility grid.  When a consumer buys 100 % from P2P the bonus equals
    ``w_import``; when buying 100 % from the grid the bonus is 0.

    Formula (§2):

        r_import = w_import × (1.0 − grid_bought_kw / max(own_deficit_kw, ε))

    Args:
        settlement: Per-agent settlement record.
        own_deficit_kw: Available deficit before market clearing (kW).
            Must be non-negative.
        agent_id: Agent identifier string.
        w_import: Import-reduction weight
            (default ``REWARD_W_IMPORT = 0.05``).

    Returns:
        Non-negative import-reduction reward in [0, w_import].
    """
    if math.isnan(settlement.grid_bought_kw) or math.isnan(own_deficit_kw):
        logger.warning(
            "NaN detected in r_import inputs for agent '%s'. Falling back to 0.0.",
            agent_id,
        )
        return 0.0

    fraction_from_grid = settlement.grid_bought_kw / max(own_deficit_kw, EPSILON)
    # Clamp to [0, 1] to guard floating-point overshoot
    fraction_from_grid = max(0.0, min(1.0, fraction_from_grid))
    r_import = w_import * (1.0 - fraction_from_grid)
    return float(r_import)
