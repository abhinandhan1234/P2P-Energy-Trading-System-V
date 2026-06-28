"""Economic reward component for the P2P Energy Trading reward system.

Computes the normalised economic reward ``r_econ`` from a per-agent
settlement record.  The raw monetary cost/income is normalised once by
the experiment-level maximum possible cost so that the component is
approximately in [−1, +1].

Design reference: docs/module_5_reward_system.md §2 and §7
"""

from __future__ import annotations

# standard library
import logging
import math

# local
from p2p_energy_trading.constants import EPSILON
from p2p_energy_trading.modules.market.models import SettlementRecord

logger = logging.getLogger(__name__)


def compute_economic_reward(
    settlement: SettlementRecord,
    max_possible_cost: float,
) -> float:
    """Compute the normalised economic reward for one agent at one timestep.

    The raw reward is the negated net cost of the settlement.  A profitable
    agent (net income) receives a positive reward; a costly agent receives a
    negative reward.  The value is then normalised by the experiment-level
    ``max_possible_cost`` so that the result is dimensionless and
    approximately in [−1, +1].

    Normalisation is performed exactly once here.  The final aggregator uses
    this value directly without further scaling.

    Args:
        settlement: Per-agent financial settlement from the Market Engine.
        max_possible_cost: Experiment-level cost ceiling used for
            normalisation.  Defined as
            ``peak_demand_college_kw × grid_buy_rate`` and is fixed for the
            duration of an experiment.  Must be positive.

    Returns:
        Normalised economic reward.  Approximately in [−1, +1].

    Raises:
        ValueError: If ``max_possible_cost`` is not positive.
    """
    if max_possible_cost <= 0.0:
        raise ValueError(
            f"max_possible_cost must be positive, got {max_possible_cost}. "
            "Compute it as peak_demand_kw × grid_buy_rate before calling this function."
        )

    r_econ_raw = -settlement.net_cost

    # Guard against NaN in settlement values
    if math.isnan(r_econ_raw):
        logger.warning(
            "NaN detected in r_econ_raw (settlement.net_cost=%.4f). "
            "Falling back to 0.0.",
            settlement.net_cost,
        )
        return 0.0

    r_econ = r_econ_raw / max(max_possible_cost, EPSILON)
    return float(r_econ)
