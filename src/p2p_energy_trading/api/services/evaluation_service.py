"""Evaluation Service for Module 10 API Layer.

Orchestrates evaluation subprocess lifecycles by launching checkpoint evaluations,
baselines, and ablation runs via evaluate.py.

Design reference: docs/module_10_api_layer.md §4
"""

from __future__ import annotations

# standard library
import datetime
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# third party
import yaml

# local
from p2p_energy_trading.api.models import ExperimentRecord, StatusResponse
from p2p_energy_trading.api.registry import ExperimentRegistry
from p2p_energy_trading.exceptions import (
    ProcessError,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _is_process_alive(pid: int | None) -> bool:
    """Check if process is alive cross-platform."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


class EvaluationService:
    """Manages active evaluation, baseline, and ablation subprocesses."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        """Initialize the evaluation service.

        Args:
            registry: The active ExperimentRegistry instance.
        """
        self.registry = registry
        self.active_processes: dict[str, subprocess.Popen[bytes]] = {}

    def _launch_evaluation_subprocess(
        self,
        record: ExperimentRecord,
        eval_cfg_dict: dict[str, Any],
        experiments_list: list[str],
        checkpoint_path: str | None = None,
    ) -> ExperimentRecord:
        """Helper to write evaluation config and launch subprocess."""
        # Write config YAML to experiment's config folder
        eval_config_file = Path(record.config_path)
        with open(eval_config_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(eval_cfg_dict, f, default_flow_style=False)

        cmd = [
            sys.executable,
            "-m",
            "p2p_energy_trading.evaluation.evaluate",
            "--config",
            str(eval_config_file),
            "--output-dir",
            record.results_dir,
        ]
        if checkpoint_path:
            cmd.extend(["--checkpoint", checkpoint_path])

        if experiments_list:
            cmd.extend(["--experiments", ",".join(experiments_list)])

        log_file_path = Path(record.log_dir) / "evaluation.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            stdout_file = open(log_file_path, "w", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
            )
            stdout_file.close()
        except Exception as e:
            record.state = "FAILED"
            record.error_message = f"Process launch error: {e}"
            self.registry.update_experiment(record)
            raise ProcessError(f"Failed to start evaluation subprocess: {e}") from e

        record.pid = process.pid
        record.state = "RUNNING"
        record.started_at = datetime.datetime.now().isoformat()
        self.registry.update_experiment(record)

        self.active_processes[record.experiment_id] = process
        logger.info(
            "Launched evaluation experiment %s (PID: %d)",
            record.experiment_id,
            process.pid,
        )
        return record

    def run(self, request: Any) -> ExperimentRecord:
        """Run the evaluation suite on a trained checkpoint."""
        record = self.registry.create_experiment(
            experiment_type="evaluation",
            config_path=request.config_path or "config/eval_config.yaml",
            name=request.experiment_name or "Trained Model Evaluation",
            tags=request.tags or ["evaluation"],
        )

        # Build dynamic evaluation config
        seeds = request.seed_values or [42, 123, 456, 789, 1024]
        seeds = seeds[: request.num_seeds]

        eval_cfg = {
            "training_config": "config/training_config.yaml",
            "checkpoint_path": request.checkpoint_path,
            "evaluation": {
                "results_dir": record.results_dir,
                "seeds": seeds,
                "eval_episode_starts": [0],  # default representative start hour
                "experiments": ["trained", "grid_only"],
            },
        }

        # Apply overrides
        if request.eval_config_overrides:
            eval_cfg.update(request.eval_config_overrides)

        return self._launch_evaluation_subprocess(
            record=record,
            eval_cfg_dict=eval_cfg,
            experiments_list=["trained", "grid_only"],
            checkpoint_path=request.checkpoint_path,
        )

    def baseline(self, request: Any) -> ExperimentRecord:
        """Run baseline comparison rollouts without checkpoints."""
        record = self.registry.create_experiment(
            experiment_type="baseline",
            config_path=request.config_path or "config/eval_config.yaml",
            name=request.experiment_name or "Baseline Comparison",
            tags=request.tags or ["baselines"],
        )

        seeds = request.seed_values or [42, 123, 456, 789, 1024]
        seeds = seeds[: request.num_seeds]

        eval_cfg = {
            "training_config": "config/training_config.yaml",
            "evaluation": {
                "results_dir": record.results_dir,
                "seeds": seeds,
                "eval_episode_starts": [0],
                "experiments": request.baselines,
            },
        }

        return self._launch_evaluation_subprocess(
            record=record,
            eval_cfg_dict=eval_cfg,
            experiments_list=request.baselines,
        )

    def ablation(self, request: Any) -> ExperimentRecord:
        """Run a specific ablation rollout (no battery, no P2P, etc.)."""
        record = self.registry.create_experiment(
            experiment_type="ablation",
            config_path=request.config_path or "config/eval_config.yaml",
            name=request.experiment_name or f"Ablation: {request.ablation_type}",
            tags=request.tags or ["ablation", request.ablation_type],
        )

        seeds = request.seed_values or [42, 123, 456, 789, 1024]
        seeds = seeds[: request.num_seeds]

        # In V1, we map ablation types to baseline/trained runs.
        # e.g., no_battery evaluates the trained checkpoint but overrides
        # actions in evaluate.py
        eval_cfg = {
            "training_config": "config/training_config.yaml",
            "checkpoint_path": request.checkpoint_path,
            "evaluation": {
                "results_dir": record.results_dir,
                "seeds": seeds,
                "eval_episode_starts": [0],
                "experiments": [request.ablation_type, "grid_only"],
            },
        }

        return self._launch_evaluation_subprocess(
            record=record,
            eval_cfg_dict=eval_cfg,
            experiments_list=[request.ablation_type, "grid_only"],
            checkpoint_path=request.checkpoint_path,
        )

    def status(self, experiment_id: str) -> StatusResponse:
        """Query execution status of evaluation run."""
        record = self.registry.get_experiment(experiment_id)

        # Check process liveness
        is_alive = False
        if record.state == "RUNNING":
            is_alive = _is_process_alive(record.pid)
            if not is_alive:
                proc = self.active_processes.get(experiment_id)
                exit_code = proc.poll() if proc else -1
                record.exit_code = exit_code

                if exit_code == 0:
                    record.state = "COMPLETED"
                else:
                    record.state = "FAILED"
                    record.error_message = f"Subprocess exited with code {exit_code}."

                record.completed_at = datetime.datetime.now().isoformat()
                if record.started_at:
                    t_start = datetime.datetime.fromisoformat(record.started_at)
                    t_end = datetime.datetime.fromisoformat(record.completed_at)
                    record.duration_s = (t_end - t_start).total_seconds()

                self.registry.update_experiment(record)
                if experiment_id in self.active_processes:
                    del self.active_processes[experiment_id]

        # Parse evaluation log for progress tracking
        log_file = Path(record.log_dir) / "evaluation.log"
        total_exps = 1
        exps_done = 0
        curr_exp = None

        # Load expected experiments from YAML
        try:
            with open(record.config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                total_exps = len(cfg.get("evaluation", {}).get("experiments", [1]))
        except Exception:
            pass

        if log_file.exists():
            try:
                with open(log_file, encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Find all unique "Starting experiment:" lines
                started = re.findall(r"Starting experiment:\s+(\w+)", content)
                exps_done = len(started)
                if started:
                    curr_exp = started[-1]
            except Exception:
                pass

        return StatusResponse(
            experiment_id=experiment_id,
            state=record.state,
            is_alive=is_alive,
            experiments_completed=exps_done,
            total_experiments=total_exps,
            current_experiment=curr_exp,
        )
