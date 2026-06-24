"""Global seed setting for reproducibility.

Provides utilities to set random seeds across Python, NumPy, and PyTorch
to ensure deterministic behavior during training and evaluation.

Reference: docs/module_12_repository_structure.md §11
"""

from __future__ import annotations

import random

import numpy as np

from p2p_energy_trading.constants import DEFAULT_SEED


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Set all random seeds for reproducibility.

    Sets seeds for Python's random module, NumPy, and PyTorch (if available).
    Must be called before any stochastic operations.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def set_numpy_seed(seed: int) -> None:
    """Set NumPy seed only.

    Useful when only NumPy randomness needs to be controlled
    (e.g., in profile generation where PyTorch is not used).

    Args:
        seed: Integer seed value.
    """
    np.random.seed(seed)


def set_torch_seed(seed: int) -> None:
    """Set PyTorch seed only.

    Args:
        seed: Integer seed value.
    """
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
