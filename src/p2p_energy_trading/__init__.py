"""P2P Energy Trading System.

A Peer-to-Peer Multi-Agent Reinforcement Learning energy trading system
for a college campus microgrid using MAPPO with CTDE.

Reference: docs/architecture.md
"""

from __future__ import annotations

# standard library
import sys

__version__ = "1.0.0"


def get_version_info() -> dict[str, str]:
    """Return version information for reproducibility.

    Returns:
        Dictionary mapping package names to their installed versions.
    """
    versions: dict[str, str] = {
        "p2p_energy_trading": __version__,
        "python": sys.version,
    }

    try:
        # third party
        import numpy as np

        versions["numpy"] = np.__version__
    except ImportError:
        versions["numpy"] = "not installed"

    try:
        # third party
        import ray

        versions["ray"] = ray.__version__
    except ImportError:
        versions["ray"] = "not installed"

    try:
        # third party
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = "not installed"

    try:
        # third party
        import pandapower

        versions["pandapower"] = pandapower.__version__
    except ImportError:
        versions["pandapower"] = "not installed"

    return versions
