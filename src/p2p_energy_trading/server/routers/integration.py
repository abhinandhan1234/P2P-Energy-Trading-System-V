from __future__ import annotations

from fastapi import APIRouter, Depends

from p2p_energy_trading.server.dependencies import get_inference_service
from p2p_energy_trading.server.rl_inference import RLInferenceService

router = APIRouter(prefix="/api", tags=["SolarXChange Integration"])


@router.get("/status")
def get_status(service: RLInferenceService = Depends(get_inference_service)) -> dict[str, object]:
    return service.status()


@router.get("/recommendation")
def get_recommendation(service: RLInferenceService = Depends(get_inference_service)) -> dict[str, object]:
    run_output = service.run_cycle()
    return run_output["recommendation"]


@router.get("/marketplace")
def get_marketplace(service: RLInferenceService = Depends(get_inference_service)) -> dict[str, list[dict[str, object]]]:
    run_output = service.run_cycle()
    return {"orders": run_output["market_orders"]}


@router.get("/blockchain/history")
def get_blockchain_history(service: RLInferenceService = Depends(get_inference_service)) -> dict[str, list[dict[str, object]]]:
    run_output = service.run_cycle()
    return {"trades": run_output["blockchain_history"]}


@router.get("/smartgrid")
def get_smartgrid(service: RLInferenceService = Depends(get_inference_service)) -> dict[str, object]:
    run_output = service.run_cycle()
    return run_output["smartgrid"]
