from fastapi import Request

from p2p_energy_trading.server.rl_inference import RLInferenceService


def get_inference_service(request: Request) -> RLInferenceService:
    """Dependency to retrieve the global RL inference service instance."""
    return request.app.state.inference_service
