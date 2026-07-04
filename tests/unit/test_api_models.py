"""Unit tests for Module 10 API Data Models.

Verifies construction, default fields, and serialization properties
of the request and record models.

Design reference: docs/module_12_repository_structure.md §7
"""

from __future__ import annotations

# local
from p2p_energy_trading.api.models import (
    AblationRequest,
    BaselineRequest,
    EvaluationRequest,
    ExperimentRecord,
    StatusResponse,
    TrainingRequest,
)


def test_training_request_defaults() -> None:
    """Verify TrainingRequest instantiates with default parameters."""
    req = TrainingRequest(config_path="config/training_config.yaml")
    assert req.config_path == "config/training_config.yaml"
    assert req.config_overrides is None
    assert req.use_gpu is True
    assert req.log_level == "INFO"


def test_evaluation_request_defaults() -> None:
    """Verify EvaluationRequest instantiates with defaults."""
    req = EvaluationRequest(checkpoint_path="checkpoints/checkpoint_000100")
    assert req.checkpoint_path == "checkpoints/checkpoint_000100"
    assert req.num_episodes == 20
    assert req.num_seeds == 5
    assert req.episode_length == 168
    assert req.deterministic is True


def test_baseline_request() -> None:
    """Verify BaselineRequest properties."""
    req = BaselineRequest(
        baselines=["grid_only", "random"], config_path="config/eval_config.yaml"
    )
    assert req.baselines == ["grid_only", "random"]
    assert req.num_episodes == 20


def test_ablation_request() -> None:
    """Verify AblationRequest fields."""
    req = AblationRequest(
        ablation_type="no_battery", config_path="config/eval_config.yaml"
    )
    assert req.ablation_type == "no_battery"
    assert req.requires_training is False


def test_experiment_record_serialization() -> None:
    """Verify ExperimentRecord to_dict and from_dict works correctly."""
    record = ExperimentRecord(
        experiment_id="exp_20260627_123456_a1b2c3",
        experiment_type="training",
        config_path="experiments/exp_20260627_123456_a1b2c3/config/effective_config.yaml",
        log_dir="experiments/exp_20260627_123456_a1b2c3/logs",
        results_dir="experiments/exp_20260627_123456_a1b2c3/results",
        created_at="2026-06-27T12:00:00",
        state="QUEUED",
        tags=["v1", "debug"],
    )

    d = record.to_dict()
    assert isinstance(d, dict)
    assert d["experiment_id"] == "exp_20260627_123456_a1b2c3"
    assert d["state"] == "QUEUED"
    assert d["tags"] == ["v1", "debug"]

    # Reconstruct from dict
    restored = ExperimentRecord.from_dict(d)
    assert restored.experiment_id == record.experiment_id
    assert restored.experiment_type == record.experiment_type
    assert restored.config_path == record.config_path
    assert restored.log_dir == record.log_dir
    assert restored.results_dir == record.results_dir
    assert restored.created_at == record.created_at
    assert restored.state == record.state
    assert restored.tags == record.tags


def test_status_response_serialization() -> None:
    """Verify StatusResponse serialization and field structures."""
    status = StatusResponse(
        experiment_id="exp_123",
        state="RUNNING",
        is_alive=True,
        current_iteration=5,
        total_iterations=100,
        metrics_summary={"mean_reward": -2.3, "entropy": 0.8},
    )

    d = status.to_dict()
    assert d["experiment_id"] == "exp_123"
    assert d["state"] == "RUNNING"
    assert d["is_alive"] is True
    assert d["metrics_summary"] == {"mean_reward": -2.3, "entropy": 0.8}
