"""Main Programmatic API Entry Point for Module 10.

Aggregates sub-services (Config, Training, Evaluation, Results) and enforces
configurable concurrency limits (default: 1 active training and 1 active evaluation subprocess).

Design reference: docs/module_10_api_layer.md §2, §3, §4
"""

from __future__ import annotations

# standard library
import datetime
from pathlib import Path
from typing import Any

# local
from p2p_energy_trading.api.models import ExperimentRecord, StatusResponse
from p2p_energy_trading.api.registry import ExperimentRegistry
from p2p_energy_trading.api.services.config_service import ConfigService
from p2p_energy_trading.api.services.evaluation_service import EvaluationService
from p2p_energy_trading.api.services.results_service import ResultsService
from p2p_energy_trading.api.services.training_service import (
    TrainingService,
    _is_process_alive,
)
from p2p_energy_trading.exceptions import ResourceError


class P2PExperimentAPI:
    """Core entry point aggregating services for configurations, training, rollouts, and results."""

    def __init__(self, base_dir: str | Path = "experiments") -> None:
        """Initialize registry and sub-services.

        Args:
            base_dir: Registry root directory path (default: 'experiments').
        """
        self._registry = ExperimentRegistry(base_dir)
        self._config_service = ConfigService(self._registry.base_dir)
        self._training_service = TrainingService(self._registry)
        self._evaluation_service = EvaluationService(self._registry)
        self._results_service = ResultsService(self._registry.base_dir)

    @property
    def registry(self) -> ExperimentRegistry:
        """Access the local ExperimentRegistry."""
        return self._registry

    @property
    def config(self) -> ConfigService:
        """Access the ConfigService."""
        return self._config_service

    @property
    def training(self) -> TrainingService:
        """Access the TrainingService."""
        return self._training_service

    @property
    def evaluation(self) -> EvaluationService:
        """Access the EvaluationService."""
        return self._evaluation_service

    @property
    def results(self) -> ResultsService:
        """Access the ResultsService."""
        return self._results_service

    def _sync_stale_processes(self) -> None:
        """Scan running records and transition dead subprocesses to FAILED."""
        running = self._registry.list_experiments({"state": "RUNNING"})
        for r in running:
            if not _is_process_alive(r.pid):
                r.state = "FAILED"
                r.error_message = "Subprocess exited unexpectedly."
                r.completed_at = datetime.datetime.now().isoformat()
                if r.started_at:
                    t_start = datetime.datetime.fromisoformat(r.started_at)
                    t_end = datetime.datetime.fromisoformat(r.completed_at)
                    r.duration_s = (t_end - t_start).total_seconds()
                self._registry.update_experiment(r)

    def start_training(self, request: Any) -> ExperimentRecord:
        """Shortcut to start a training subprocess, enforcing concurrency guards."""
        self._sync_stale_processes()

        # Enforce training concurrency guard
        running_train = [
            r
            for r in self._registry.list_experiments({"state": "RUNNING"})
            if r.experiment_type == "training"
        ]
        if running_train:
            raise ResourceError(
                f"A training subprocess is already executing: {running_train[0].experiment_id}. "
                "Concurrency limit reached (max: 1 training process)."
            )

        return self._training_service.start(request)

    def start_evaluation(self, request: Any) -> ExperimentRecord:
        """Shortcut to start an evaluation subprocess, enforcing concurrency guards."""
        self._sync_stale_processes()

        # Enforce evaluation concurrency guard
        running_eval = [
            r
            for r in self._registry.list_experiments({"state": "RUNNING"})
            if r.experiment_type in ["evaluation", "baseline", "ablation"]
        ]
        if running_eval:
            raise ResourceError(
                f"An evaluation subprocess is already executing: {running_eval[0].experiment_id}. "
                "Concurrency limit reached (max: 1 evaluation process)."
            )

        return self._evaluation_service.run(request)

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Shortcut method to fetch experiment details by ID."""
        return self._registry.get_experiment(experiment_id)

    def list_experiments(
        self, filters: dict[str, Any] | None = None
    ) -> list[ExperimentRecord]:
        """Shortcut method to list or filter experiment catalog."""
        self._sync_stale_processes()
        return self._registry.list_experiments(filters)

    def get_status(self, experiment_id: str) -> StatusResponse:
        """Shortcut method to query the status report of an experiment."""
        record = self._registry.get_experiment(experiment_id)
        if record.experiment_type == "training":
            return self._training_service.status(experiment_id)
        else:
            return self._evaluation_service.status(experiment_id)
