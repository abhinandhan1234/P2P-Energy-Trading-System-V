"""API Layer for the P2P Energy Trading System (Module 10).

Exposes public programmatic models, the main orchestrator P2PExperimentAPI,
and exception classes.

Design reference: docs/module_10_api_layer.md §2, §8
"""

from __future__ import annotations

# local
from p2p_energy_trading.api.experiment_api import P2PExperimentAPI
from p2p_energy_trading.api.models import (
    AblationRequest,
    BaselineRequest,
    CheckpointInfo,
    ComparisonResult,
    ConfigDiff,
    ConfigInfo,
    ErrorResponse,
    EvaluationRequest,
    ExperimentRecord,
    FigureInfo,
    MetricsResult,
    ReportResult,
    ResumeRequest,
    StatusResponse,
    TrainingRequest,
    ValidationError,
    ValidationResult,
)
from p2p_energy_trading.exceptions import (
    CheckpointError,
    ConfigMismatchError,
    ConfigNotFoundError,
    ConfigValidationError,
    ExperimentNotFoundError,
    InvalidStateError,
    ProcessError,
    RegistryError,
    ResourceError,
)

__all__ = [
    "P2PExperimentAPI",
    "TrainingRequest",
    "EvaluationRequest",
    "BaselineRequest",
    "AblationRequest",
    "ResumeRequest",
    "ExperimentRecord",
    "StatusResponse",
    "CheckpointInfo",
    "ValidationError",
    "ValidationResult",
    "ConfigDiff",
    "ComparisonResult",
    "MetricsResult",
    "FigureInfo",
    "ReportResult",
    "ConfigInfo",
    "ErrorResponse",
    # Exceptions
    "ConfigValidationError",
    "CheckpointError",
    "ExperimentNotFoundError",
    "ConfigNotFoundError",
    "ConfigMismatchError",
    "InvalidStateError",
    "ProcessError",
    "ResourceError",
    "RegistryError",
]
