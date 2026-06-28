"""Reward aggregator for the P2P Energy Trading reward system.

Computes the complete per-agent scalar reward by:

1. Deriving auxiliary quantities (``own_surplus``, ``own_deficit``,
   ``solar_used_locally``) from raw inputs without modifying upstream
   modules.
2. Calling each reward component function from the sub-modules.
3. Summing components, applying curriculum-phase weights, and clipping
   the final value to ``[REWARD_CLIP_MIN, REWARD_CLIP_MAX]`` = [−10, +10].
4. Logging a WARNING for any NaN component and substituting 0.0 (the
   module never crashes during ``step()``).

The aggregator is the sole function called by Module 6 (environment).
All internal reward components are pure functions with no hidden state.

Design reference: docs/module_5_reward_system.md §1–§7
"""

from __future__ import annotations

# standard library
import logging
import math

# local
from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    COLLEGE_AGENT_ID,
    CONSUMER_AGENT_IDS,
    REWARD_CLIP_MAX,
    REWARD_CLIP_MIN,
    SOLAR_AGENT_IDS,
)
from p2p_energy_trading.modules.market.models import SettlementRecord
from p2p_energy_trading.modules.network.powerflow import PowerFlowResult
from p2p_energy_trading.modules.reward.battery import (
    compute_cycling_penalty,
    compute_soc_penalty,
    compute_storage_reward,
)
from p2p_energy_trading.modules.reward.economic import compute_economic_reward
from p2p_energy_trading.modules.reward.grid_safety import (
    compute_thermal_penalty,
    compute_transformer_penalty,
    compute_voltage_penalty,
)
from p2p_energy_trading.modules.reward.trading import (
    compute_import_reduction_reward,
    compute_p2p_reward,
    compute_self_consumption_reward,
)

logger = logging.getLogger(__name__)


def _derive_energy_quantities(
    agent_id: str,
    demand_kw: float,
    solar_kw: float,
    battery_dispatch_kw: float,
) -> tuple[float, float]:
    """Derive pre-market own_surplus and own_deficit for one agent.

    These are the available energy surpluses/deficits computed before
    any P2P clearing takes place.  To match the worked examples and the
    observation builder specification, these are the raw (pre-battery)
    quantities based only on solar generation and demand.

    Args:
        agent_id: Agent identifier string.
        demand_kw: Raw demand profile at this timestep (kW).
        solar_kw: Raw solar generation at this timestep (kW).
        battery_dispatch_kw: Actual battery dispatch (kW).
            Included for signature compatibility, but unused as these
            are pre-battery quantities.

    Returns:
        Tuple of (own_surplus_kw, own_deficit_kw), both non-negative.
    """
    net_power = demand_kw - solar_kw
    own_surplus_kw = max(0.0, -net_power)
    own_deficit_kw = max(0.0, net_power)
    return own_surplus_kw, own_deficit_kw


def compute_agent_reward(
    agent_id: str,
    settlement: SettlementRecord,
    demand_kw: float,
    solar_kw: float,
    battery_dispatch_kw: float,
    battery_soc: float,
    prev_battery_dispatch_kw: float,
    power_flow_result: PowerFlowResult | None,
    max_possible_cost: float,
    curriculum_phase: int = 1,
) -> float:
    """Compute the complete clipped scalar reward for one agent at one timestep.

    This is the primary public function of Module 5.  Module 6 calls it once
    per agent per ``step()``.

    Reward component applicability by agent type:

    +-----------------+--------+-------+----------+
    | Component       | College| Solar | Consumer |
    +=================+========+=======+==========+
    | r_econ          |   ✓    |   ✓   |    ✓     |
    | r_p2p           |   ✓    |   ✓   |    ✓     |
    | r_self          |   ✓    |   ✓   |    —     |
    | r_import        |   —    |   —   |    ✓     |
    | r_voltage       |   ✓    |   ✓   |    ✓     |
    | r_thermal       |   ✓    |   ✓   |    ✓     |
    | r_transformer   |   ✓    |   —   |    —     |
    | r_soc           |   ✓    |   —   |    —     |
    | r_cycling       |   ✓    |   —   |    —     |
    | r_storage       |   ✓    |   —   |    —     |
    +-----------------+--------+-------+----------+

    Args:
        agent_id: Agent identifier string (e.g. ``"college"``,
            ``"solar_01"``, ``"consumer_03"``).
        settlement: Per-agent financial settlement from Module 3.
        demand_kw: Raw demand at this timestep (kW).  Non-negative.
        solar_kw: Raw solar generation at this timestep (kW).
            Pass 0.0 for consumer agents.
        battery_dispatch_kw: Actual battery dispatch this step (kW).
            Positive = discharging, negative = charging.
            Pass 0.0 for non-college agents.
        battery_soc: Current battery State of Charge in [0.0, 1.0].
            Pass any value for non-college agents (ignored).
        prev_battery_dispatch_kw: Battery dispatch from the previous
            timestep (kW).  Required for cycling detection.
            Pass 0.0 for non-college agents.
        power_flow_result: Power flow result from Module 2, or ``None``
            when power flow is bypassed.
        max_possible_cost: Experiment-level normalisation constant (₹).
            Defined as ``peak_demand_college_kw × grid_buy_rate``.
        curriculum_phase: Training curriculum phase (1 or 2).  Controls
            grid penalty weights.

    Returns:
        Total reward, clipped to [``REWARD_CLIP_MIN``, ``REWARD_CLIP_MAX``]
        = [−10.0, +10.0].
    """
    # --- Derive auxiliary quantities -------------------------------------------
    own_surplus_kw, own_deficit_kw = _derive_energy_quantities(
        agent_id, demand_kw, solar_kw, battery_dispatch_kw
    )

    # --- Economic reward -------------------------------------------------------
    r_econ = compute_economic_reward(settlement, max_possible_cost)

    # --- Market participation rewards ------------------------------------------
    r_p2p = compute_p2p_reward(
        settlement,
        own_surplus_kw=own_surplus_kw,
        own_deficit_kw=own_deficit_kw,
        agent_id=agent_id,
    )

    is_college = agent_id == COLLEGE_AGENT_ID
    is_solar = agent_id in SOLAR_AGENT_IDS
    is_consumer = agent_id in CONSUMER_AGENT_IDS

    r_self = 0.0
    if is_college or is_solar:
        r_self = compute_self_consumption_reward(
            settlement,
            solar_kw=solar_kw,
            battery_dispatch_kw=battery_dispatch_kw if is_college else 0.0,
            agent_id=agent_id,
        )

    r_import = 0.0
    if is_consumer:
        r_import = compute_import_reduction_reward(
            settlement,
            own_deficit_kw=own_deficit_kw,
            agent_id=agent_id,
        )

    # --- Grid safety penalties ------------------------------------------------
    r_voltage = compute_voltage_penalty(
        power_flow_result,
        agent_id=agent_id,
        curriculum_phase=curriculum_phase,
    )
    r_thermal = compute_thermal_penalty(
        power_flow_result,
        agent_id=agent_id,
        curriculum_phase=curriculum_phase,
    )

    r_transformer = 0.0
    if is_college:
        r_transformer = compute_transformer_penalty(
            power_flow_result,
            curriculum_phase=curriculum_phase,
        )

    # --- Battery rewards (College only) ----------------------------------------
    r_soc = 0.0
    r_cycling = 0.0
    r_storage = 0.0
    if is_college:
        r_soc = compute_soc_penalty(battery_soc)
        r_cycling = compute_cycling_penalty(
            battery_dispatch_kw=battery_dispatch_kw,
            prev_dispatch_kw=prev_battery_dispatch_kw,
        )
        r_storage = compute_storage_reward(
            battery_dispatch_kw=battery_dispatch_kw,
            own_surplus_kw=own_surplus_kw,
            own_deficit_kw=own_deficit_kw,
        )

    # --- Aggregate and guard against NaN --------------------------------------
    components = {
        "r_econ": r_econ,
        "r_p2p": r_p2p,
        "r_self": r_self,
        "r_import": r_import,
        "r_voltage": r_voltage,
        "r_thermal": r_thermal,
        "r_transformer": r_transformer,
        "r_soc": r_soc,
        "r_cycling": r_cycling,
        "r_storage": r_storage,
    }

    total = 0.0
    for name, value in components.items():
        if math.isnan(value):
            logger.warning(
                "NaN reward component '%s' for agent '%s'. Substituting 0.0.",
                name,
                agent_id,
            )
            value = 0.0
        total += value

    # --- Clip to training-safe range ------------------------------------------
    clipped = max(REWARD_CLIP_MIN, min(REWARD_CLIP_MAX, total))

    logger.debug(
        "Agent '%s' | phase=%d | econ=%.3f p2p=%.3f self=%.3f import=%.3f "
        "volt=%.3f therm=%.3f trafo=%.3f soc=%.3f cyc=%.3f stor=%.3f "
        "→ total=%.3f clipped=%.3f",
        agent_id,
        curriculum_phase,
        r_econ,
        r_p2p,
        r_self,
        r_import,
        r_voltage,
        r_thermal,
        r_transformer,
        r_soc,
        r_cycling,
        r_storage,
        total,
        clipped,
    )

    return float(clipped)


def compute_all_rewards(
    settlements: dict[str, SettlementRecord],
    demands_kw: dict[str, float],
    solar_kw: dict[str, float],
    battery_dispatch_kw: float,
    battery_soc: float,
    prev_battery_dispatch_kw: float,
    power_flow_result: PowerFlowResult | None,
    max_possible_cost: float,
    curriculum_phase: int = 1,
) -> dict[str, float]:
    """Compute per-agent rewards for all 21 agents at one timestep.

    Convenience wrapper that calls ``compute_agent_reward`` for every
    agent in ``ALL_AGENT_IDS`` order.  The result is a ``RewardDict``
    consumed directly by ``MultiAgentEnv.step()``.

    Args:
        settlements: Dict mapping agent ID to ``SettlementRecord``.
        demands_kw: Dict mapping agent ID to demand (kW).
        solar_kw: Dict mapping agent ID to solar generation (kW).
        battery_dispatch_kw: College battery dispatch this step (kW).
        battery_soc: College battery SoC at this step.
        prev_battery_dispatch_kw: College battery dispatch last step (kW).
        power_flow_result: Power flow result or ``None`` (bypass mode).
        max_possible_cost: Experiment-level normalisation constant (₹).
        curriculum_phase: Training curriculum phase (1 or 2).

    Returns:
        Dict mapping each agent ID to its clipped scalar reward.
    """
    rewards: dict[str, float] = {}
    for agent_id in ALL_AGENT_IDS:
        if agent_id not in settlements:
            logger.warning(
                "Missing settlement for agent '%s'. Reward defaults to 0.0.",
                agent_id,
            )
            rewards[agent_id] = 0.0
            continue

        rewards[agent_id] = compute_agent_reward(
            agent_id=agent_id,
            settlement=settlements[agent_id],
            demand_kw=demands_kw.get(agent_id, 0.0),
            solar_kw=solar_kw.get(agent_id, 0.0),
            battery_dispatch_kw=battery_dispatch_kw
            if agent_id == COLLEGE_AGENT_ID
            else 0.0,
            battery_soc=battery_soc,
            prev_battery_dispatch_kw=prev_battery_dispatch_kw
            if agent_id == COLLEGE_AGENT_ID
            else 0.0,
            power_flow_result=power_flow_result,
            max_possible_cost=max_possible_cost,
            curriculum_phase=curriculum_phase,
        )
    return rewards
