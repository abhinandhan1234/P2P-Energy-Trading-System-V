"""Data models for the programmatic interface of Module 10.

Defines requests, responses, status reports, configuration validation schemas,
and serializable experiment records.

Design reference: docs/module_10_api_layer.md §8
"""

from __future__ import annotations

# standard library
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrainingRequest:
    """Request model to launch PPO/MAPPO policy training."""

    config_path: str
    config_overrides: dict[str, Any] | None = None
    experiment_name: str | None = None
    tags: list[str] | None = None
    seed: int | None = None
    stage: str | None = None
    max_iterations: int | None = None
    num_workers: int | None = None
    use_gpu: bool = True
    log_level: str = "INFO"


@dataclass
class EvaluationRequest:
    """Request model to run evaluation rollouts on a trained checkpoint."""

    checkpoint_path: str
    config_path: str | None = None
    eval_config_overrides: dict[str, Any] | None = None
    experiment_name: str | None = None
    tags: list[str] | None = None
    num_episodes: int = 20
    num_seeds: int = 5
    seed_values: list[int] | None = None
    episode_length: int = 168
    deterministic: bool = True


@dataclass
class BaselineRequest:
    """Request model to evaluate non-learning baseline controllers."""

    baselines: list[str]
    config_path: str
    num_episodes: int = 20
    num_seeds: int = 5
    seed_values: list[int] | None = None
    experiment_name: str | None = None
    tags: list[str] | None = None


@dataclass
class AblationRequest:
    """Request model to run an ablation study (e.g. no battery or no P2P trading)."""

    ablation_type: (
        str  # "no_battery" | "no_p2p" | "no_grid_penalties" | "independent_policies"
    )
    config_path: str
    checkpoint_path: str | None = None
    config_overrides: dict[str, Any] | None = None
    num_episodes: int = 20
    num_seeds: int = 5
    seed_values: list[int] | None = None
    experiment_name: str | None = None
    tags: list[str] | None = None
    requires_training: bool = False


@dataclass
class ResumeRequest:
    """Request model to resume training from a checkpoint."""

    experiment_id: str
    checkpoint_path: str | None = None
    config_overrides: dict[str, Any] | None = None
    max_iterations: int | None = None


@dataclass
class ExperimentRecord:
    """State record for a single training, evaluation, baseline, or ablation run."""

    experiment_id: str
    experiment_type: str
    config_path: str
    log_dir: str
    results_dir: str
    created_at: str
    experiment_name: str | None = None
    state: str = "QUEUED"
    tags: list[str] = field(default_factory=list)
    parent_id: str | None = None
    checkpoint_dir: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    duration_s: float | None = None
    error_message: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    artifacts: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert ExperimentRecord to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentRecord:
        """Construct ExperimentRecord from dictionary."""
        return cls(**d)


@dataclass
class StatusResponse:
    """Consolidated execution status report of an experiment."""

    experiment_id: str
    state: str
    is_alive: bool

    # Training-specific properties
    current_iteration: int | None = None
    total_iterations: int | None = None
    current_stage: str | None = None
    agent_steps: int | None = None
    best_reward: float | None = None
    latest_checkpoint: str | None = None
    elapsed_time_s: float | None = None
    metrics_summary: dict[str, float] | None = None

    # Evaluation-specific properties
    experiments_completed: int | None = None
    total_experiments: int | None = None
    current_experiment: str | None = None
    episodes_completed: int | None = None
    total_episodes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert StatusResponse to dictionary."""
        return asdict(self)


@dataclass
class CheckpointInfo:
    """Metadata regarding a saved RL training checkpoint."""

    checkpoint_path: str
    checkpoint_type: str
    iteration: int
    agent_steps: int
    stage: str
    created_at: str
    size_bytes: int
    mean_reward: float | None = None


@dataclass
class ValidationError:
    """Individual configuration validation error."""

    field: str
    message: str
    value: Any
    expected: str


@dataclass
class ValidationResult:
    """Aggregated validation results."""

    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConfigDiff:
    """Detailed difference between two configurations."""

    field: str
    value_a: Any
    value_b: Any
    category: str  # "safe" | "breaking"


@dataclass
class ComparisonResult:
    """Result of configuration comparison."""

    identical: bool
    differences: list[ConfigDiff] = field(default_factory=list)
    safe_differences: list[ConfigDiff] = field(default_factory=list)
    breaking_differences: list[ConfigDiff] = field(default_factory=list)


@dataclass
class MetricsResult:
    """Metric query result data container."""

    experiment_id: str
    level: str
    data: Any
    columns: list[str]
    row_count: int
    generated_at: str


@dataclass
class FigureInfo:
    """Generated Matplotlib plot metadata details."""

    name: str
    path_png: str
    path_pdf: str
    description: str
    generated_at: str
    size_bytes: int


@dataclass
class ReportResult:
    """Evaluation framework report compilation."""

    experiment_id: str
    format: str
    content: str | dict[str, Any]
    tables: list[str]
    figures: list[str]
    generated_at: str


@dataclass
class ConfigInfo:
    """Information regarding a uploaded configuration template."""

    config_id: str
    name: str
    path: str
    schema_version: str
    validated: bool
    uploaded_at: str


@dataclass
class ErrorResponse:
    """API error response wrapper."""

    error_type: str
    message: str
    timestamp: str
    details: dict[str, Any] | None = None
    experiment_id: str | None = None
