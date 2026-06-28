"""Project-wide exception hierarchy.

All custom exceptions inherit from P2PEnergyTradingError to enable
selective catching at module boundaries.

Reference: docs/module_12_repository_structure.md §9
"""

from __future__ import annotations


class P2PEnergyTradingError(Exception):
    """Base exception for the P2P Energy Trading System."""


class ProfileGenerationError(P2PEnergyTradingError):
    """Error during profile generation (Module 1).

    Raised when raw data loading, profile synthesis, or validation fails.
    """


class PowerFlowError(P2PEnergyTradingError):
    """PandaPower convergence or network error (Module 2).

    Raised when Newton-Raphson solver fails to converge after all retries,
    or when the network topology is invalid.
    """


class MarketClearingError(P2PEnergyTradingError):
    """Market clearing error (Module 3).

    Raised when the clearing algorithm encounters an unrecoverable state,
    such as negative quantities or energy balance violation beyond tolerance.
    """


class ConfigValidationError(P2PEnergyTradingError):
    """Configuration validation failure (Module 8/10).

    Raised when YAML config values are missing, out of range, or
    internally inconsistent.
    """


class CheckpointError(P2PEnergyTradingError):
    """Checkpoint save/restore error (Module 8/10).

    Raised when a checkpoint cannot be saved, loaded, or is corrupted.
    """


class ExperimentNotFoundError(P2PEnergyTradingError):
    """Experiment ID not found in registry (Module 10).

    Raised when an API call references a non-existent experiment.
    """


class ConfigNotFoundError(P2PEnergyTradingError):
    """Configuration file not found (Module 10)."""


class ConfigMismatchError(P2PEnergyTradingError):
    """Unsafe configuration change during resume (Module 10)."""


class InvalidStateError(P2PEnergyTradingError):
    """Operation not allowed in the current state (Module 10)."""


class ProcessError(P2PEnergyTradingError):
    """Subprocess failed to launch or run correctly (Module 10)."""


class ResourceError(P2PEnergyTradingError):
    """Insufficient resources on local machine (Module 10)."""


class RegistryError(P2PEnergyTradingError):
    """Registry data corrupted or inaccessible (Module 10)."""
