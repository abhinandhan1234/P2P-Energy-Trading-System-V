from typing import Any
from fastapi import APIRouter, Depends, HTTPException

from p2p_energy_trading.api.experiment_api import P2PExperimentAPI
from p2p_energy_trading.api.models import (
    ExperimentRecord,
    StatusResponse,
    TrainingRequest,
    EvaluationRequest,
    BaselineRequest,
    AblationRequest,
)
from p2p_energy_trading.exceptions import ResourceError, ExperimentNotFoundError
from p2p_energy_trading.server.dependencies import get_api

router = APIRouter(prefix="/experiments", tags=["Experiments"])

@router.get("/", response_model=list[ExperimentRecord])
def list_experiments(api: P2PExperimentAPI = Depends(get_api)):
    """List all experiments."""
    return api.list_experiments()

@router.get("/{experiment_id}", response_model=ExperimentRecord)
def get_experiment(experiment_id: str, api: P2PExperimentAPI = Depends(get_api)):
    """Get details of a specific experiment."""
    try:
        return api.get_experiment(experiment_id)
    except ExperimentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.get("/{experiment_id}/status", response_model=StatusResponse)
def get_status(experiment_id: str, api: P2PExperimentAPI = Depends(get_api)):
    """Get the status of a specific experiment."""
    try:
        return api.get_status(experiment_id)
    except ExperimentNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@router.post("/training", response_model=ExperimentRecord)
def start_training(request: TrainingRequest, api: P2PExperimentAPI = Depends(get_api)):
    """Start a new training experiment."""
    try:
        return api.start_training(request)
    except ResourceError as e:
        raise HTTPException(status_code=429, detail=str(e))

@router.post("/evaluation", response_model=ExperimentRecord)
def start_evaluation(request: EvaluationRequest, api: P2PExperimentAPI = Depends(get_api)):
    """Start a new evaluation experiment."""
    try:
        return api.start_evaluation(request)
    except ResourceError as e:
        raise HTTPException(status_code=429, detail=str(e))
