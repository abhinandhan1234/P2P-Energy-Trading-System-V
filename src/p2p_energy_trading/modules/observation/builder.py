"""Observation Builder for the P2P Energy Trading microgrid.

Assembles uniform 23-dimensional actor observations and the 243-dimensional
centralised critic state, applying standard normalizations and grid/market
bypass fallbacks.

Design reference: docs/module_4_observation_builder.md
"""

from __future__ import annotations

# standard library
import logging
from typing import Any

# third party
import numpy as np
import pandas as pd

# local
from p2p_energy_trading.constants import (
    AGENT_TO_BUS,
    ALL_AGENT_IDS,
    BATTERY_SOC_MAX,
    BATTERY_SOC_MIN,
    COLLEGE_AGENT_ID,
    GRID_IMPORT_EXPORT_LIMIT_KW,
    NUM_BUSES,
)
from p2p_energy_trading.modules.market.models import MarketState
from p2p_energy_trading.modules.network.powerflow import PowerFlowResult
from p2p_energy_trading.modules.observation.normalisation import (
    compute_demand_ratio,
    cyclical_time_encoding,
    normalise_energy,
    normalise_grid_flow,
    normalise_grid_price,
    normalise_loading,
    normalise_p2p_price,
    normalise_voltage,
)

logger = logging.getLogger(__name__)


def build_observations(
    demands_kw: dict[str, float],
    solar_kw: dict[str, float],
    battery_state: dict[str, float],
    last_actions: dict[str, np.ndarray],
    grid_result: PowerFlowResult | None,
    market_state: MarketState,
    timestamp: pd.Timestamp,
    grid_buy_rate: float,
    grid_sell_rate: float,
    metadata: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Construct actor and critic observations for all 21 agents.

    Args:
        demands_kw: Current demand profiles mapped by agent ID (kW).
        solar_kw: Current solar generation profiles mapped by agent ID (kW).
        battery_state: Dict with college battery state ('soc', 'dispatch_kw', etc.).
        last_actions: Previous 3-dim actions mapped by agent ID.
        grid_result: PandaPower power flow result or None if bypassed.
        market_state: Uniform P2P market state for the current timestep.
        timestamp: Current profile datetime/timestamp.
        grid_buy_rate: Current grid buy tariff (Rs/kWh).
        grid_sell_rate: Current grid sell tariff (Rs/kWh).
        metadata: Portfolio metadata dictionary containing peak stats.

    Returns:
        Dict mapping each agent_id to a Dict with:
        - "obs": Actor observation array of shape (23,) and float32 dtype.
        - "state": Centralised critic state array of shape (243,) and float32 dtype.
    """
    # 1. Time encoding
    hour = timestamp.hour
    if hasattr(timestamp, "dayofweek"):
        day_of_week = int(timestamp.dayofweek)
    elif hasattr(timestamp, "weekday"):
        day_of_week = int(timestamp.weekday())
    else:
        day_of_week = 0

    hour_sin, hour_cos, day_sin, day_cos = cyclical_time_encoding(hour, day_of_week)

    # 2. Build local observations for all agents
    local_obs_dict: dict[str, np.ndarray] = {}
    local_obs_list: list[np.ndarray] = []

    for aid in ALL_AGENT_IDS:
        loc_vec = _build_local_obs(
            aid, demands_kw, solar_kw, battery_state, last_actions, metadata
        )
        local_obs_dict[aid] = loc_vec
        local_obs_list.append(loc_vec)

    # Concatenate all local observations in ALL_AGENT_IDS order
    all_local_obs = np.concatenate(local_obs_list, dtype=np.float32)

    # 3. Construct global features for the critic state
    # Global grid stats
    total_solar_peak = sum(
        metadata.get("buildings", {})
        .get(aid, {})
        .get("profile_stats", {})
        .get("solar_generation_kw", {})
        .get("peak", 0.0)
        for aid in ALL_AGENT_IDS
    )
    total_demand_peak = sum(
        metadata.get("buildings", {})
        .get(aid, {})
        .get("profile_stats", {})
        .get("demand_kw", {})
        .get("peak", 0.0)
        for aid in ALL_AGENT_IDS
    )

    total_solar_gen = sum(solar_kw.get(aid, 0.0) for aid in ALL_AGENT_IDS)
    total_demand_val = sum(demands_kw.get(aid, 0.0) for aid in ALL_AGENT_IDS)

    total_gen_norm = normalise_energy(total_solar_gen, total_solar_peak)
    total_demand_norm = normalise_energy(total_demand_val, total_demand_peak)

    if grid_result is not None:
        p_grid = grid_result.p_grid_kw
    else:
        p_grid = 0.0
    net_flow_norm = normalise_grid_flow(p_grid, GRID_IMPORT_EXPORT_LIMIT_KW)

    global_grid_state = [total_gen_norm, total_demand_norm, net_flow_norm]

    # Aggregate market state
    price_norm = normalise_p2p_price(
        market_state.p2p_clearing_price, grid_buy_rate, grid_sell_rate
    )
    volume_norm = normalise_energy(market_state.total_p2p_volume, total_demand_peak)
    utilisation_ratio = market_state.p2p_utilisation_ratio

    aggregate_market = [price_norm, volume_norm, utilisation_ratio]

    # PowerFlow summary
    if grid_result is not None:
        primary_voltages = [
            grid_result.bus_vm_pu[i]
            for i in range(NUM_BUSES)
            if i in grid_result.bus_vm_pu
        ]
        v_min = min(primary_voltages) if primary_voltages else 1.0
        v_min_norm = normalise_voltage(v_min)

        line_loadings = list(grid_result.line_loading_pct.values())
        l_max = max(line_loadings) if line_loadings else 0.0
        l_max_norm = normalise_loading(l_max)

        t_load = grid_result.trafo_loading_pct.get(0, 0.0)
        t_load_norm = normalise_loading(t_load)
    else:
        v_min_norm = 0.5  # corresponds to 1.0 p.u. (1.0 - 0.95) / 0.10
        l_max_norm = 0.0
        t_load_norm = 0.0

    pp_summary = [v_min_norm, l_max_norm, t_load_norm]

    # Battery state
    college_soc = battery_state.get("soc", 0.5)
    available_charge_norm = max(
        0.0,
        BATTERY_SOC_MAX - college_soc,
    )
    available_discharge_norm = max(
        0.0,
        college_soc - BATTERY_SOC_MIN,
    )

    battery_state_vec = [
        college_soc,
        available_charge_norm,
        available_discharge_norm,
    ]

    # Assemble centralized critic state vector (243 dims)
    critic_state = np.concatenate(
        [
            all_local_obs,
            np.array(global_grid_state, dtype=np.float32),
            np.array(aggregate_market, dtype=np.float32),
            np.array(pp_summary, dtype=np.float32),
            np.array(battery_state_vec, dtype=np.float32),
        ],
        dtype=np.float32,
    )

    # 4. Build uniform actor observations for all agents
    observations: dict[str, dict[str, np.ndarray]] = {}

    for aid in ALL_AGENT_IDS:
        # Local bus metrics
        if grid_result is not None:
            bus_idx = AGENT_TO_BUS[aid]
            v_pu = grid_result.bus_vm_pu.get(bus_idx, 1.0)
            l_pct = grid_result.line_loading_pct.get(bus_idx - 1, 0.0)
            t_pct = grid_result.trafo_loading_pct.get(0, 0.0)
        else:
            v_pu = 1.0
            l_pct = 0.0
            t_pct = 0.0

        v_norm = normalise_voltage(v_pu)
        l_norm = normalise_loading(l_pct)
        p_grid_norm = normalise_grid_flow(p_grid, GRID_IMPORT_EXPORT_LIMIT_KW)
        t_norm = normalise_loading(t_pct)

        grid_feats = [v_norm, l_norm, p_grid_norm, t_norm]

        # Market metrics
        buy_norm = normalise_grid_price(grid_buy_rate)
        sell_norm = normalise_grid_price(grid_sell_rate)
        demand_ratio = compute_demand_ratio(
            market_state.total_bids, market_state.total_offers
        )

        market_feats = [price_norm, buy_norm, sell_norm, demand_ratio]

        # Time metrics
        time_feats = [hour_sin, hour_cos, day_sin, day_cos]

        # Combine into actor observation vector (23 dims)
        actor_obs = np.concatenate(
            [
                local_obs_dict[aid],
                np.array(grid_feats, dtype=np.float32),
                np.array(market_feats, dtype=np.float32),
                np.array(time_feats, dtype=np.float32),
            ],
            dtype=np.float32,
        )

        observations[aid] = {"obs": actor_obs, "state": critic_state}

    return observations


def _build_local_obs(
    agent_id: str,
    demands_kw: dict[str, float],
    solar_kw: dict[str, float],
    battery_state: dict[str, float],
    last_actions: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> np.ndarray:
    """Helper to build the 11-dimensional local observation vector for an agent."""
    # 1. Retrieve stats from metadata
    b_meta = metadata.get("buildings", {}).get(agent_id, {})
    profile_stats = b_meta.get("profile_stats", {})
    peak_demand = profile_stats.get("demand_kw", {}).get("peak", 0.0)
    peak_solar = profile_stats.get("solar_generation_kw", {}).get("peak", 0.0)

    # 2. Local power values
    demand = demands_kw.get(agent_id, 0.0)
    solar = solar_kw.get(agent_id, 0.0)
    surplus = max(0.0, solar - demand)
    deficit = max(0.0, demand - solar)

    # 3. Battery state (college only)
    soc = battery_state.get("soc", 0.0) if agent_id == COLLEGE_AGENT_ID else 0.0

    # 4. Previous action
    default_act = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    last_action = last_actions.get(agent_id, default_act)
    last_action = np.clip(
        last_action,
        0.0,
        1.0,
    )

    # 5. One-hot role representation
    role = [0.0, 0.0, 0.0]
    if agent_id == COLLEGE_AGENT_ID:
        role[0] = 1.0
    elif agent_id.startswith("solar_"):
        role[1] = 1.0
    elif agent_id.startswith("consumer_"):
        role[2] = 1.0

    # Local vector layout
    return np.array(
        [
            normalise_energy(solar, peak_solar),
            normalise_energy(demand, peak_demand),
            soc,
            normalise_energy(surplus, peak_solar),
            normalise_energy(deficit, peak_demand),
            float(last_action[0]),
            float(last_action[1]),
            float(last_action[2]),
            role[0],
            role[1],
            role[2],
        ],
        dtype=np.float32,
    )
