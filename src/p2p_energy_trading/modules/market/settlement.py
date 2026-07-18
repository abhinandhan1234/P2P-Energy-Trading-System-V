"""Market Settlement calculations and validations.

This module computes individual agent settlements, handles the HESCOM grid
utility fallbacks, and executes strict energy balance audits.

Design reference: docs/module_3_market_engine.md
"""

from __future__ import annotations

# standard library
import logging
import math

# third party
import numpy as np

# local
from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    BATTERY_CAPACITY_KWH,
    BATTERY_EFFICIENCY,
    BATTERY_MIN_DISPATCH_KW,
    BATTERY_POWER_KW,
    BATTERY_SOC_MAX,
    BATTERY_SOC_MIN,
    COLLEGE_AGENT_ID,
    ENERGY_BALANCE_TOLERANCE_KW,
)
from p2p_energy_trading.exceptions import MarketClearingError
from p2p_energy_trading.modules.market.clearing import clear_market_p2p
from p2p_energy_trading.modules.market.models import MarketState, SettlementRecord

logger = logging.getLogger(__name__)


def process_settlements(
    demands_kw: dict[str, float],
    solar_kw: dict[str, float],
    actions: dict[str, np.ndarray],
    battery_soc: float,
    grid_buy_rate: float,
    grid_sell_rate: float,
) -> tuple[dict[str, SettlementRecord], MarketState]:
    """Execute battery dispatch scaling, P2P clearing, utility fallbacks,
    and settlements.

    This function does not mutate or store any state, operating as a pure calculation.

    Args:
        demands_kw: Dict mapping agent ID to raw profile demand (kW).
        solar_kw: Dict mapping agent ID to raw profile solar generation (kW).
        actions: Dict mapping agent ID to 3-dim action vector
            [buy_frac, sell_frac, dispatch].
        battery_soc: Current State of Charge (fraction) of the college battery.
        grid_buy_rate: Grid buy rate (Rs/kWh).
        grid_sell_rate: Grid sell rate (Rs/kWh).

    Returns:
        tuple containing:
        - dict[str, SettlementRecord]: Per-agent settlement records.
        - MarketState: Global market state summary.

    Raises:
        MarketClearingError: If negative values occur or energy balance checks fail.
    """
    # 1. Input validations
    for aid in ALL_AGENT_IDS:
        if demands_kw.get(aid, 0.0) < 0.0 or solar_kw.get(aid, 0.0) < 0.0:
            raise MarketClearingError(
                f"Negative demand or solar input values for agent '{aid}'"
                " are not permitted."
            )
        if aid not in actions:
            raise MarketClearingError(f"Missing action vector for agent '{aid}'.")
        if len(actions[aid]) != 3:
            raise MarketClearingError(
                f"Action vector for agent '{aid}' must have length 3,"
                f" got {len(actions[aid])}."
            )

    # 2. Pure evaluation of College battery dispatch for the current timestep (dt=1.0 h)
    college_action = actions[COLLEGE_AGENT_ID]
    action_charge_fraction = float(college_action[2])

    # Symmetrical efficiency for charging and discharging: eta = sqrt(0.90)
    eta_charge = math.sqrt(BATTERY_EFFICIENCY)
    eta_discharge = math.sqrt(BATTERY_EFFICIENCY)
    dt = 1.0

    # Convert action to desired power: 0.5 -> 0, 0.0 -> 250 kW, 1.0 -> -250 kW
    desired_power_kw = (0.5 - action_charge_fraction) * 2.0 * BATTERY_POWER_KW
    desired_power_kw = max(-BATTERY_POWER_KW, min(BATTERY_POWER_KW, desired_power_kw))

    # Apply SoC limits and calculate actual power
    if desired_power_kw < 0:  # Charging (absorbing)
        charge_power_demand = -desired_power_kw
        kwh_to_fill = (BATTERY_SOC_MAX - battery_soc) * BATTERY_CAPACITY_KWH
        max_charge_power = kwh_to_fill / (eta_charge * dt)
        actual_charge_power = min(charge_power_demand, max_charge_power)
        battery_dispatch_kw = -actual_charge_power
    elif desired_power_kw > 0:  # Discharging (injecting)
        discharge_power_demand = desired_power_kw
        kwh_to_drain = (battery_soc - BATTERY_SOC_MIN) * BATTERY_CAPACITY_KWH
        max_discharge_power = kwh_to_drain * eta_discharge / dt
        actual_discharge_power = min(discharge_power_demand, max_discharge_power)
        battery_dispatch_kw = actual_discharge_power
    else:
        battery_dispatch_kw = 0.0

    # Enforce minimum dispatch threshold (25 kW)
    if abs(battery_dispatch_kw) < BATTERY_MIN_DISPATCH_KW:
        battery_dispatch_kw = 0.0

    # 3. Compute bids and offers for all agents based on post-battery surplus/deficit
    bids: dict[str, float] = {}
    offers: dict[str, float] = {}
    available_deficits: dict[str, float] = {}
    available_surpluses: dict[str, float] = {}

    for aid in ALL_AGENT_IDS:
        demand = demands_kw.get(aid, 0.0)
        solar = solar_kw.get(aid, 0.0)

        if aid == COLLEGE_AGENT_ID:
            # Net power includes battery dispatch
            net_power = demand - solar - battery_dispatch_kw
        else:
            net_power = demand - solar

        if net_power > 0.0:
            available_deficit = net_power
            available_surplus = 0.0
        else:
            available_deficit = 0.0
            available_surplus = -net_power

        available_deficits[aid] = available_deficit
        available_surpluses[aid] = available_surplus

        # Clip fractions to [0.0, 1.0] before scaling
        buy_frac = max(0.0, min(1.0, float(actions[aid][0])))
        sell_frac = max(0.0, min(1.0, float(actions[aid][1])))

        bids[aid] = buy_frac * available_deficit
        offers[aid] = sell_frac * available_surplus

    # 4. Clear P2P Market
    p2p_bought, p2p_sold, total_volume, curtailment_applied = clear_market_p2p(
        bids, offers
    )

    # 5. Financial rates and fallback quantities
    p2p_price = (grid_buy_rate + grid_sell_rate) / 2.0
    settlement_records: dict[str, SettlementRecord] = {}

    grid_import_total = 0.0
    grid_export_total = 0.0

    for aid in ALL_AGENT_IDS:
        demand = demands_kw.get(aid, 0.0)
        solar = solar_kw.get(aid, 0.0)
        p2p_b = p2p_bought[aid]
        p2p_s = p2p_sold[aid]

        grid_b = available_deficits[aid] - p2p_b
        grid_s = available_surpluses[aid] - p2p_s

        # Financial values
        p2p_revenue = p2p_s * p2p_price
        p2p_cost = p2p_b * p2p_price
        grid_revenue = grid_s * grid_sell_rate
        grid_cost = grid_b * grid_buy_rate
        net_cost = grid_cost + p2p_cost - grid_revenue - p2p_revenue

        # Keep running grid totals
        grid_import_total += grid_b
        grid_export_total += grid_s

        record = SettlementRecord(
            p2p_sold_kw=p2p_s,
            p2p_bought_kw=p2p_b,
            grid_sold_kw=grid_s,
            grid_bought_kw=grid_b,
            p2p_price=p2p_price,
            p2p_revenue=p2p_revenue,
            p2p_cost=p2p_cost,
            grid_revenue=grid_revenue,
            grid_cost=grid_cost,
            net_cost=net_cost,
        )
        settlement_records[aid] = record

        # 6. Energy Balance Validation
        # Check non-negativity of cleared values
        if p2p_s < -1e-9 or p2p_b < -1e-9 or grid_s < -1e-9 or grid_b < -1e-9:
            raise MarketClearingError(
                f"Negative quantities in cleared trades or fallback"
                f" values for agent '{aid}'."
            )

        # Separate battery components for validation equations
        b_discharge = max(0.0, battery_dispatch_kw) if aid == COLLEGE_AGENT_ID else 0.0
        b_charge = max(0.0, -battery_dispatch_kw) if aid == COLLEGE_AGENT_ID else 0.0

        # Assertions/Checks for battery limits (only for college agent)
        if aid == COLLEGE_AGENT_ID:
            # 1. Power limit
            if abs(battery_dispatch_kw) > BATTERY_POWER_KW + 1e-5:
                raise MarketClearingError(
                    f"Battery dispatch {battery_dispatch_kw:.4f} kW exceeds "
                    f"power limit {BATTERY_POWER_KW} kW"
                )

            # 2. SoC limits
            if battery_dispatch_kw < 0.0:  # Charging
                charge_power = -battery_dispatch_kw
                kwh_to_fill = (BATTERY_SOC_MAX - battery_soc) * BATTERY_CAPACITY_KWH
                max_charge = kwh_to_fill / (eta_charge * dt)
                if charge_power > max_charge + 1e-5:
                    raise MarketClearingError(
                        f"Battery charge {charge_power:.4f} kW violates max charge "
                        f"limit {max_charge:.4f} kW based on SoC max."
                    )
            elif battery_dispatch_kw > 0.0:  # Discharging
                discharge_power = battery_dispatch_kw
                kwh_to_drain = (battery_soc - BATTERY_SOC_MIN) * BATTERY_CAPACITY_KWH
                max_discharge = kwh_to_drain * eta_discharge / dt
                if discharge_power > max_discharge + 1e-5:
                    raise MarketClearingError(
                        f"Battery discharge {discharge_power:.4f} kW violates "
                        f"max discharge limit {max_discharge:.4f} kW based on SoC min."
                    )

        # 3. Energy conservation check (total energy in = total energy out)
        energy_in = solar + p2p_b + grid_b + b_discharge
        energy_out = demand + p2p_s + grid_s + b_charge
        if abs(energy_in - energy_out) > ENERGY_BALANCE_TOLERANCE_KW:
            raise MarketClearingError(
                f"Energy conservation violation for agent '{aid}': "
                f"Energy In = {energy_in:.4f} kW, Energy Out = {energy_out:.4f} kW"
            )

        # 4. Export limit check: exports must not exceed available surplus
        available_surplus = available_surpluses[aid]
        exports = p2p_s + grid_s
        if exports > available_surplus + ENERGY_BALANCE_TOLERANCE_KW:
            raise MarketClearingError(
                f"Exports {exports:.4f} kW exceed available surplus "
                f"{available_surplus:.4f} kW for agent '{aid}'."
            )

        # 5. Local solar consumption definition and limits
        local_load = demand + b_charge
        solar_to_load = min(solar, local_load)
        solar_surplus = solar - solar_to_load
        solar_exported = min(solar_surplus, exports)
        local_solar_consumption = solar - solar_exported

        if local_solar_consumption < -1e-9 or local_solar_consumption > solar + 1e-9:
            raise MarketClearingError(
                f"Calculated local solar consumption is out of bounds for "
                f"agent '{aid}': local_solar_consumption = "
                f"{local_solar_consumption:.4f} kW, solar = {solar:.4f} kW"
            )

    # 7. Construct MarketState
    total_bids = sum(bids.values())
    total_offers = sum(offers.values())
    max_clearable = min(total_bids, total_offers)
    utilisation_ratio = total_volume / max_clearable if max_clearable > 0.0 else 0.0

    market_state = MarketState(
        p2p_clearing_price=p2p_price,
        total_p2p_volume=total_volume,
        p2p_utilisation_ratio=utilisation_ratio,
        grid_import_total=grid_import_total,
        grid_export_total=grid_export_total,
        voltage_violation=False,
        thermal_violation=False,
        curtailment_applied=curtailment_applied,
        total_bids=total_bids,
        total_offers=total_offers,
    )

    return settlement_records, market_state
