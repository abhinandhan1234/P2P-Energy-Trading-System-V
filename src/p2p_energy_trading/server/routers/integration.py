from __future__ import annotations

import random
import secrets
from datetime import datetime, timezone
from fastapi import APIRouter, Depends

from p2p_energy_trading.api.experiment_api import P2PExperimentAPI
from p2p_energy_trading.server.dependencies import get_api

router = APIRouter(prefix="/api", tags=["SolarXChange Integration"])


@router.get("/status")
def get_status(api: P2PExperimentAPI = Depends(get_api)) -> dict[str, object]:
    active = api.list_experiments()
    return {
        "status": "ok",
        "experiment_count": len(active),
        "last_experiment": active[-1].experiment_id if active else None,
    }


@router.get("/recommendation")
def get_recommendation(api: P2PExperimentAPI = Depends(get_api)) -> dict[str, object]:
    experiments = api.list_experiments()
    if experiments:
        last = experiments[-1]
        headline = f"Last model run: {last.experiment_name or last.experiment_id}"
    else:
        headline = "Ready for your first training experiment."

    signal = random.choice(["SELL", "BUY", "HOLD"])
    price = round(random.uniform(0.24, 0.42), 2)
    soc = round(random.uniform(0.62, 0.92), 2)
    p2p_volume = round(random.uniform(10.0, 32.0), 1)

    return {
        "headline": headline,
        "body": (
            f"MARL agents recommend {signal} at ₹{price}/kWh. "
            f"Battery SoC is {int(soc * 100)}%. "
            f"Expected P2P volume: {p2p_volume} kWh."
        ),
        "confidence": round(random.uniform(76.0, 98.0), 1),
        "signal": signal,
        "price_rs_kwh": price,
        "battery_soc": soc,
        "p2p_volume": p2p_volume,
    }


@router.get("/marketplace")
def get_marketplace() -> dict[str, list[dict[str, object]]]:
    orders = [
        {
            "name": "SolarFarm_P04",
            "type": "Producer",
            "price": 0.28,
            "kwh": 48,
            "status": "SELL",
            "location": "2.1km away",
        },
        {
            "name": "RoofTop_Sunny_7",
            "type": "Producer",
            "price": 0.31,
            "kwh": 12,
            "status": "SELL",
            "location": "0.8km away",
        },
        {
            "name": "CampusArray_B12",
            "type": "Producer",
            "price": 0.26,
            "kwh": 200,
            "status": "SELL",
            "location": "3.4km away",
        },
    ]
    return {"orders": orders}


@router.post("/trade/execute")
def execute_trade(body: dict[str, object]) -> dict[str, object]:
    tx_hash = f"0x{secrets.token_hex(16)}"
    return {
        "status": "confirmed",
        "tx_hash": tx_hash,
        "settled_at": datetime.now(timezone.utc).isoformat(),
        "details": body,
    }


@router.get("/blockchain/history")
def get_blockchain_history() -> dict[str, list[dict[str, object]]]:
    now = datetime.now(timezone.utc)
    sample = [
        {
            "tx_hash": f"0x{secrets.token_hex(10)}",
            "origin": "Sol-Node-Paris-04",
            "kwh": 24.5,
            "timestamp": (now).strftime("%H:%M:%S UTC"),
            "status": "CONFIRMED",
        },
        {
            "tx_hash": f"0x{secrets.token_hex(10)}",
            "origin": "Storage-Unit-A12",
            "kwh": -11.2,
            "timestamp": (now).strftime("%H:%M:%S UTC"),
            "status": "CONFIRMED",
        },
        {
            "tx_hash": f"0x{secrets.token_hex(10)}",
            "origin": "Consumer-Res-09",
            "kwh": -45.0,
            "timestamp": (now).strftime("%H:%M:%S UTC"),
            "status": "CONFIRMED",
        },
    ]
    return {"trades": sample}


@router.get("/smartgrid")
def get_smartgrid() -> dict[str, object]:
    return {
        "network_load": round(random.uniform(28.0, 42.0), 1),
        "grid_stability": round(random.uniform(98.8, 99.99), 2),
        "active_nodes": random.randint(6, 9),
        "grid_efficiency": round(random.uniform(92.0, 98.5), 1),
    }
