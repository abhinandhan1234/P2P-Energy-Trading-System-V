from fastapi import APIRouter, Depends, HTTPException

from p2p_energy_trading.api.experiment_api import P2PExperimentAPI
from p2p_energy_trading.api.models import ConfigInfo
from p2p_energy_trading.exceptions import ConfigNotFoundError
from p2p_energy_trading.server.dependencies import get_api

router = APIRouter(prefix="/configs", tags=["Configuration"])

@router.get("/", response_model=list[ConfigInfo])
def list_configs(api: P2PExperimentAPI = Depends(get_api)):
    """List all available configurations."""
    return api.config.list_configs()

@router.get("/{name}", response_model=ConfigInfo)
def get_config(name: str, api: P2PExperimentAPI = Depends(get_api)):
    """Get details of a specific configuration."""
    try:
        return api.config.get_config(name)
    except ConfigNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
