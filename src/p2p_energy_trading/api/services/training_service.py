"""Training Service for Module 10 API Layer.

Orchestrates training subprocess lifecycles by launching, resuming, stopping,
and querying the status of train.py processes.

Design reference: docs/module_10_api_layer.md §3
"""

from __future__ import annotations

# standard library
import datetime
import json
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
from p2p_energy_trading.api.models import (
    CheckpointInfo,
    ExperimentRecord,
    StatusResponse,
)
from p2p_energy_trading.api.registry import ExperimentRegistry
from p2p_energy_trading.exceptions import (
    CheckpointError,
    InvalidStateError,
    ProcessError,
)
from p2p_energy_trading.training.config_loader import load_training_config

logger = logging.getLogger(__name__)

# Find project root
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


class TrainingService:
    """Manages active training processes, checkpoint registers, and logs parsing."""

    def __init__(self, registry: ExperimentRegistry) -> None:
        """Initialize the training service.

        Args:
            registry: The active ExperimentRegistry instance.
        """
        self.registry = registry
        self.active_processes: dict[str, subprocess.Popen[bytes]] = {}

    def start(self, request: Any) -> ExperimentRecord:
        """Launch a new training experiment as a background subprocess."""
        # 1. Load config and overrides using Module 8 loader
        try:
            config = load_training_config(
                request.config_path, overrides=request.config_overrides
            )
        except Exception as e:
            raise ValueError(f"Config load error: {e}") from e

        # 2. Register record in registry (state: QUEUED)
        record = self.registry.create_experiment(
            experiment_type="training",
            config_path=request.config_path,
            name=request.experiment_name,
            tags=request.tags,
            seed=request.seed,
        )

        # 3. Update effective config directory paths to use sandbox directories
        if "checkpoint" not in config:
            config["checkpoint"] = {}
        config["checkpoint"]["checkpoint_dir"] = str(
            Path(record.checkpoint_dir).resolve()
        )
        config["checkpoint"]["best_model_dir"] = str(
            Path(record.checkpoint_dir).resolve() / "best_model"
        )

        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["tensorboard_dir"] = str(
            Path(record.log_dir).resolve() / "tensorboard"
        )
        config["logging"]["metrics_dir"] = str(Path(record.results_dir).resolve())
        config["logging"]["evaluation_dir"] = str(
            Path(record.results_dir).resolve() / "evaluation"
        )
        config["logging"]["log_file"] = str(
            Path(record.log_dir).resolve() / "training.log"
        )

        # Write effective configuration snapshot for reproducibility
        with open(record.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        # 4. Construct command line arguments
        cmd = [
            sys.executable,
            "-m",
            "p2p_energy_trading.training.train",
            "--config",
            record.config_path,
        ]
        if request.seed is not None:
            cmd.extend(["--seed", str(request.seed)])
        if request.stage is not None:
            cmd.extend(["--stage", request.stage])
        if request.max_iterations is not None:
            cmd.extend(["--iterations", str(request.max_iterations)])
        if request.num_workers is not None:
            cmd.extend(["--num-workers", str(request.num_workers)])
        if not request.use_gpu:
            cmd.append("--no-gpu")
        if request.log_level is not None:
            cmd.extend(["--log-level", request.log_level])

        # 5. Launch subprocess
        log_file_path = Path(record.log_dir) / "training.log"
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
            stdout_file.close()  # Subprocess now owns the handle
        except Exception as e:
            record.state = "FAILED"
            record.error_message = f"Process launch error: {e}"
            self.registry.update_experiment(record)
            raise ProcessError(f"Failed to start training subprocess: {e}") from e

        # 6. Update record to RUNNING
        record.pid = process.pid
        record.state = "RUNNING"
        record.started_at = datetime.datetime.now().isoformat()
        self.registry.update_experiment(record)

        self.active_processes[record.experiment_id] = process
        logger.info(
            "Launched training experiment %s (PID: %d)",
            record.experiment_id,
            process.pid,
        )
        return record

    def resume(self, request: Any) -> ExperimentRecord:
        """Resume training from a checkpoint."""
        # 1. Fetch parent record
        parent = self.registry.get_experiment(request.experiment_id)

        # 2. Find resume checkpoint
        checkpoint_dir = request.checkpoint_path
        if not checkpoint_dir:
            # Find latest checkpoint
            periodic_checkpoints = self.list_checkpoints(request.experiment_id)
            if not periodic_checkpoints:
                raise CheckpointError(
                    f"No checkpoints found for experiment"
                    f" '{request.experiment_id}' to resume."
                )
            checkpoint_dir = periodic_checkpoints[-1].checkpoint_path

        if not os.path.exists(checkpoint_dir):
            raise CheckpointError(
                f"Resume checkpoint directory not found: '{checkpoint_dir}'"
            )

        # 3. Load config and overrides
        try:
            config = load_training_config(
                parent.config_path, overrides=request.config_overrides
            )
        except Exception as e:
            raise ValueError(f"Config load error: {e}") from e

        # 4. Create resume record
        record = self.registry.create_experiment(
            experiment_type="training",
            config_path=parent.config_path,
            name=f"Resume of {parent.experiment_id}",
            tags=parent.tags + ["resume"],
            parent_id=parent.experiment_id,
        )

        # Keep output directories relative to new experiment sandbox
        if "checkpoint" not in config:
            config["checkpoint"] = {}
        config["checkpoint"]["checkpoint_dir"] = str(
            Path(record.checkpoint_dir).resolve()
        )
        config["checkpoint"]["best_model_dir"] = str(
            Path(record.checkpoint_dir).resolve() / "best_model"
        )

        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["tensorboard_dir"] = str(
            Path(record.log_dir).resolve() / "tensorboard"
        )
        config["logging"]["metrics_dir"] = str(Path(record.results_dir).resolve())
        config["logging"]["evaluation_dir"] = str(
            Path(record.results_dir).resolve() / "evaluation"
        )
        config["logging"]["log_file"] = str(
            Path(record.log_dir).resolve() / "training.log"
        )

        # Write effective configuration snapshot
        with open(record.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, default_flow_style=False)

        # 5. Build command using the --resume CLI override
        cmd = [
            sys.executable,
            "-m",
            "p2p_energy_trading.training.train",
            "--config",
            record.config_path,
            "--resume",
            checkpoint_dir,
        ]
        if request.max_iterations is not None:
            cmd.extend(["--iterations", str(request.max_iterations)])

        # 6. Launch subprocess
        log_file_path = Path(record.log_dir) / "training.log"
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
            raise ProcessError(f"Failed to resume training subprocess: {e}") from e

        record.pid = process.pid
        record.state = "RUNNING"
        record.started_at = datetime.datetime.now().isoformat()
        self.registry.update_experiment(record)

        self.active_processes[record.experiment_id] = process
        logger.info(
            "Resumed training experiment %s from checkpoint %s (PID: %d)",
            record.experiment_id,
            checkpoint_dir,
            process.pid,
        )
        return record

    def stop(self, experiment_id: str) -> ExperimentRecord:
        """Gracefully stop a running training experiment."""
        record = self.registry.get_experiment(experiment_id)

        if record.state != "RUNNING":
            raise InvalidStateError(
                f"Experiment '{experiment_id}' is not running (state: {record.state})."
            )

        pid = record.pid
        if not pid:
            raise ProcessError(
                f"No process PID stored for experiment '{experiment_id}'."
            )

        logger.info("Stopping training experiment %s (PID: %d)", experiment_id, pid)

        # Call terminate if process was started in this python process,
        # otherwise call kill
        proc = self.active_processes.get(experiment_id)
        if proc:
            try:
                # Terminate sends SIGTERM on Unix and calls TerminateProcess on Windows
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        else:
            try:
                # Windows and Unix compatible termination
                # standard library
                import signal

                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass

        # Update registry state
        record.state = "STOPPED"
        record.completed_at = datetime.datetime.now().isoformat()
        if record.started_at:
            t_start = datetime.datetime.fromisoformat(record.started_at)
            t_end = datetime.datetime.fromisoformat(record.completed_at)
            record.duration_s = (t_end - t_start).total_seconds()

        self.registry.update_experiment(record)
        if experiment_id in self.active_processes:
            del self.active_processes[experiment_id]

        return record

    def status(self, experiment_id: str) -> StatusResponse:
        """Query execution status and parse structured progress indicators."""
        record = self.registry.get_experiment(experiment_id)

        # Check process liveness
        is_alive = False
        if record.state == "RUNNING":
            is_alive = _is_process_alive(record.pid)
            if not is_alive:
                # Subprocess exited without API intervention
                proc = self.active_processes.get(experiment_id)
                exit_code = None
                if proc:
                    exit_code = proc.poll()
                else:
                    exit_code = -1  # Placeholder

                record.exit_code = exit_code
                if exit_code == 0:
                    record.state = "COMPLETED"
                else:
                    record.state = "FAILED"
                    record.error_message = (
                        f"Subprocess exited unexpectedly with code {exit_code}."
                    )

                record.completed_at = datetime.datetime.now().isoformat()
                if record.started_at:
                    t_start = datetime.datetime.fromisoformat(record.started_at)
                    t_end = datetime.datetime.fromisoformat(record.completed_at)
                    record.duration_s = (t_end - t_start).total_seconds()

                self.registry.update_experiment(record)
                if experiment_id in self.active_processes:
                    del self.active_processes[experiment_id]

        # Parse metrics from best_model_metadata.json
        best_reward = None
        best_model_meta_file = (
            Path(record.checkpoint_dir or "")
            / "best_model"
            / "best_model_metadata.json"
        )
        if best_model_meta_file.exists():
            try:
                with open(best_model_meta_file, encoding="utf-8") as f:
                    meta = json.load(f)
                    best_reward = meta.get("mean_eval_reward")
            except Exception:
                pass

        curr_iter = None
        curr_stage = None
        agent_steps = None
        metrics_summary = None
        log_file = Path(record.log_dir) / "training.log"

        if log_file.exists():
            try:
                # Read last 4000 characters from training.log
                with open(log_file, encoding="utf-8", errors="ignore") as lf:
                    lf.seek(0, 2)
                    size = lf.tell()
                    # Read at most 4KB from the end
                    read_size = min(4096, size)
                    lf.seek(size - read_size)
                    content = lf.read()

                iter_pattern = re.compile(
                    r"\[Iter\s+(\d+)\s+\|\s+Stage:\s+(\w+)\s+"
                    r"\|\s+Phase:\s+\d+\s+\|\s+Steps:\s+([\d.]+)M\]"
                )
                rew_pattern = re.compile(
                    r"Reward:\s+college=([\d.-]+)\s+solar=([\d.-]+)"
                    r"\s+consumer=([\d.-]+)\s+mean=([\d.-]+)"
                )
                mkt_pattern = re.compile(
                    r"Market:\s+P2P_vol=([\d.-]+)kWh\s+util=([\d.-]+)"
                    r"\s+campus_cost=₹?([\d,.-]+)"
                )
                grid_pattern = re.compile(
                    r"Grid:\s+violations=([\d.-]+)\s+min_V=([\d.-]+)"
                    r"\s+max_loading=([\d.-]+)"
                )
                loss_pattern = re.compile(
                    r"Training:\s+loss=([\d.-]+)\s+entropy=([\d.-]+)\s+KL=([\d.-]+)"
                )

                iter_matches = list(iter_pattern.finditer(content))
                if iter_matches:
                    last_iter_match = iter_matches[-1]
                    curr_iter = int(last_iter_match.group(1))
                    curr_stage = last_iter_match.group(2)
                    agent_steps = int(float(last_iter_match.group(3)) * 1_000_000)

                    # Extract up to 1KB of text following this match
                    # to find metrics lines
                    start_pos = last_iter_match.end()
                    block = content[start_pos : start_pos + 1024]

                    rew_match = rew_pattern.search(block)
                    mkt_match = mkt_pattern.search(block)
                    grid_match = grid_pattern.search(block)
                    loss_match = loss_pattern.search(block)

                    if rew_match and mkt_match and grid_match and loss_match:
                        metrics_summary = {
                            "mean_reward": float(rew_match.group(4)),
                            "policy_college_reward": float(rew_match.group(1)),
                            "policy_solar_reward": float(rew_match.group(2)),
                            "policy_consumer_reward": float(rew_match.group(3)),
                            "p2p_volume": float(mkt_match.group(1)),
                            "violation_rate": float(grid_match.group(1)),
                            "loss_total": float(loss_match.group(1)),
                            "entropy": float(loss_match.group(2)),
                            "kl_divergence": float(loss_match.group(3)),
                        }
            except Exception:
                pass

        # Fetch total iterations from YAML
        total_iterations = None
        try:
            with open(record.config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
                total_iterations = cfg.get("training", {}).get(
                    "max_training_iterations", 1000
                )
        except Exception:
            pass

        # Calculate elapsed time
        elapsed = None
        if record.started_at:
            t_start = datetime.datetime.fromisoformat(record.started_at)
            t_now = (
                datetime.datetime.fromisoformat(record.completed_at)
                if record.completed_at
                else datetime.datetime.now()
            )
            elapsed = (t_now - t_start).total_seconds()

        # Find latest checkpoint directory
        latest_checkpoint = None
        checkpoints = self.list_checkpoints(experiment_id)
        if checkpoints:
            latest_checkpoint = checkpoints[-1].checkpoint_path

        return StatusResponse(
            experiment_id=experiment_id,
            state=record.state,
            is_alive=is_alive,
            current_iteration=curr_iter,
            total_iterations=total_iterations,
            current_stage=curr_stage,
            agent_steps=agent_steps,
            best_reward=best_reward,
            latest_checkpoint=latest_checkpoint,
            elapsed_time_s=elapsed,
            metrics_summary=metrics_summary,
        )

    def list_checkpoints(self, experiment_id: str) -> list[CheckpointInfo]:
        """Scan experiment checkpoints directory and list available checkpoints."""
        record = self.registry.get_experiment(experiment_id)

        chk_dir = Path(record.checkpoint_dir or "")
        if not chk_dir.exists():
            return []

        checkpoints = []
        # Match checkpoints directories: checkpoint_XXXXXX or stage_XXXX or emergency
        for item in chk_dir.iterdir():
            if item.is_dir():
                meta_file = item / "checkpoint_metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, encoding="utf-8") as f:
                            meta = json.load(f)

                        chk_type = "periodic"
                        if "stage_" in item.name:
                            chk_type = "stage_transition"
                        elif "emergency" in item.name:
                            chk_type = "emergency"

                        stat = item.stat()
                        size = sum(
                            f.stat().st_size for f in item.glob("**/*") if f.is_file()
                        )

                        info = CheckpointInfo(
                            checkpoint_path=str(item),
                            checkpoint_type=chk_type,
                            iteration=meta.get("iteration", 0),
                            agent_steps=meta.get("agent_steps", 0),
                            stage=meta.get("curriculum_stage", "unknown"),
                            created_at=datetime.datetime.fromtimestamp(
                                stat.st_mtime
                            ).isoformat(),
                            size_bytes=size,
                        )
                        checkpoints.append(info)
                    except Exception:
                        pass

        # Sort chronologically by iteration
        checkpoints.sort(key=lambda x: x.iteration)
        return checkpoints
