"""Experiment Registry for Module 10 API Layer.

Handles persistence, CRUD operations, state transitions, and artifact indexing
on the local filesystem under the 'experiments/' directory.

Design reference: docs/module_10_api_layer.md §7
"""

from __future__ import annotations

# standard library
import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

# local
from p2p_energy_trading.api.models import ExperimentRecord
from p2p_energy_trading.exceptions import ExperimentNotFoundError, RegistryError

logger = logging.getLogger(__name__)

# File locking compatibility helpers
try:
    # standard library
    import msvcrt

    def _lock_file(f: Any) -> None:
        # Lock 1 byte from current position
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def _unlock_file(f: Any) -> None:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
except ImportError:
    try:
        # standard library
        import fcntl

        def _lock_file(f: Any) -> None:
            fcntl.flock(f, fcntl.LOCK_EX)

        def _unlock_file(f: Any) -> None:
            fcntl.flock(f, fcntl.LOCK_UN)
    except ImportError:

        def _lock_file(f: Any) -> None:
            pass

        def _unlock_file(f: Any) -> None:
            pass


class ExperimentRegistry:
    """Manages local filesystem storage and index tracking for all experiments."""

    def __init__(self, base_dir: str | Path = "experiments") -> None:
        """Initialize registry directories and JSON file.

        Args:
            base_dir: Path to directory housing the registry and experiment directories.
        """
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / "registry.json"
        self._init_registry()

    def _init_registry(self) -> None:
        """Create empty registry.json if not present."""
        if not self.registry_file.exists():
            with open(self.registry_file, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=4)

    def _load_registry_data(self, f: Any) -> dict[str, Any]:
        """Read data from registry file with default fallback."""
        try:
            f.seek(0)
            content = f.read()
            if not content.strip():
                return {}
            return json.loads(content)
        except Exception as e:
            raise RegistryError(f"Failed to parse registry file: {e}") from e

    def _write_registry_data(self, f: Any, data: dict[str, Any]) -> None:
        """Write registry data after truncating file."""
        try:
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=4)
            f.flush()
        except Exception as e:
            raise RegistryError(f"Failed to write registry file: {e}") from e

    def generate_experiment_id(self, config_path: str, seed: int | None = None) -> str:
        """Generate chronologically sortable unique experiment ID.

        Format: exp_{YYYYMMDD}_{HHMMSS}_{hash6}
        """
        now = datetime.datetime.now()
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")

        # Read config content if exists to add entropy
        config_content = ""
        try:
            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as cf:
                    config_content = cf.read()
        except Exception:
            pass

        entropy = f"{timestamp_str}_{config_content}_{seed or 42}"
        hash6 = hashlib.sha256(entropy.encode("utf-8")).hexdigest()[:6]
        return f"exp_{timestamp_str}_{hash6}"

    def create_experiment(
        self,
        experiment_type: str,
        config_path: str,
        name: str | None = None,
        tags: list[str] | None = None,
        parent_id: str | None = None,
        seed: int | None = None,
    ) -> ExperimentRecord:
        """Create and initialize a new experiment record and directory sandbox."""
        experiment_id = self.generate_experiment_id(config_path, seed=seed)
        exp_dir = self.base_dir / experiment_id

        # Enforce sandbox paths
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "config").mkdir(exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        (exp_dir / "results").mkdir(exist_ok=True)

        created_at = datetime.datetime.now().isoformat()

        record = ExperimentRecord(
            experiment_id=experiment_id,
            experiment_type=experiment_type,
            config_path=str(exp_dir / "config" / "effective_config.yaml"),
            log_dir=str(exp_dir / "logs"),
            results_dir=str(exp_dir / "results"),
            checkpoint_dir=str(exp_dir / "checkpoints")
            if experiment_type == "training"
            else None,
            created_at=created_at,
            experiment_name=name,
            state="QUEUED",
            tags=tags or [],
            parent_id=parent_id,
        )

        # Persist directory specific metadata.json
        meta_file = exp_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as mf:
            json.dump(record.to_dict(), mf, indent=4)

        # Update root registry index
        with open(self.registry_file, "r+", encoding="utf-8") as f:
            _lock_file(f)
            try:
                registry = self._load_registry_data(f)
                registry[experiment_id] = record.to_dict()
                self._write_registry_data(f, registry)
            finally:
                _unlock_file(f)

        return record

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Retrieve an experiment record by ID."""
        with open(self.registry_file, encoding="utf-8") as f:
            registry = json.load(f)

        if experiment_id not in registry:
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

        return ExperimentRecord.from_dict(registry[experiment_id])

    def update_experiment(self, record: ExperimentRecord) -> None:
        """Update an existing experiment record."""
        experiment_id = record.experiment_id
        exp_dir = self.base_dir / experiment_id

        if not exp_dir.exists():
            raise ExperimentNotFoundError(
                f"Experiment '{experiment_id}' dir not found."
            )

        record_dict = record.to_dict()

        # Update metadata.json inside the experiment directory
        meta_file = exp_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as mf:
            json.dump(record_dict, mf, indent=4)

        # Update root registry index
        with open(self.registry_file, "r+", encoding="utf-8") as f:
            _lock_file(f)
            try:
                registry = self._load_registry_data(f)
                if experiment_id not in registry:
                    raise ExperimentNotFoundError(
                        f"Experiment '{experiment_id}' not indexed."
                    )
                registry[experiment_id] = record_dict
                self._write_registry_data(f, registry)
            finally:
                _unlock_file(f)

    def list_experiments(
        self, filters: dict[str, Any] | None = None
    ) -> list[ExperimentRecord]:
        """List experiments from registry, optionally filtered.

        Allowed filters:
        - state: str (e.g. "COMPLETED")
        - experiment_type: str (e.g. "training")
        - tags: list[str] (e.g. ["v1"])
        """
        with open(self.registry_file, encoding="utf-8") as f:
            registry = json.load(f)

        records = [ExperimentRecord.from_dict(d) for d in registry.values()]

        if not filters:
            return records

        filtered = []
        for r in records:
            match = True
            for k, v in filters.items():
                if k == "state" and r.state != v:
                    match = False
                elif k == "experiment_type" and r.experiment_type != v:
                    match = False
                elif k == "tags":
                    if not all(tag in r.tags for tag in v):
                        match = False
            if match:
                filtered.append(r)

        return filtered
