from fastapi import Request
from p2p_energy_trading.api.experiment_api import P2PExperimentAPI

def get_api(request: Request) -> P2PExperimentAPI:
    """Dependency to retrieve the global P2PExperimentAPI instance."""
    return request.app.state.api
