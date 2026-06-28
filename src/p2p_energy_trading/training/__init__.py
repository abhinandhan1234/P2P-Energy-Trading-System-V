"""Training Pipeline Package — Module 8.

Orchestrates RLlib MAPPO training for the 21-agent energy trading system. Exposes
configuration loading, curriculum management, checkpointing, and CLI execution.

Design reference: docs/module_8_training_pipeline.md
"""

from __future__ import annotations

# local
from p2p_energy_trading.training.checkpoint_manager import CheckpointManager
from p2p_energy_trading.training.config_loader import load_training_config
from p2p_energy_trading.training.curriculum import CurriculumManager
from p2p_energy_trading.training.train import main

__all__ = [
    "CheckpointManager",
    "load_training_config",
    "CurriculumManager",
    "main",
]
