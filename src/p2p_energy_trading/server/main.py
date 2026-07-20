from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn

from p2p_energy_trading.api.experiment_api import P2PExperimentAPI
from p2p_energy_trading.server.routers import experiments, config

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the global API instance
    app.state.api = P2PExperimentAPI()
    yield
    # Cleanup if necessary
    pass

app = FastAPI(
    title="P2P Energy Trading API",
    description="REST API for the P2P Multi-Agent Reinforcement Learning Energy Trading System.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all CORS origins for easier frontend integration during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(experiments.router)
app.include_router(config.router)

root_dir = Path(__file__).resolve().parent.parent.parent.parent
frontend_dir = root_dir / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
async def serve_frontend_index():
    return FileResponse(frontend_dir / "index.html")

if __name__ == "__main__":
    uvicorn.run("p2p_energy_trading.server.main:app", host="127.0.0.1", port=8000, reload=True)
