"""Configuration Service for Module 10 API Layer.

Handles upload, schema validation, configuration comparison (safe vs breaking
differences), and effective configuration exports by reusing Module 8's config
loader.

Design reference: docs/module_10_api_layer.md §6
"""

from __future__ import annotations

# standard library
import datetime
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

# local
from p2p_energy_trading.api.models import (
    ComparisonResult,
    ConfigDiff,
    ConfigInfo,
    ValidationError,
    ValidationResult,
)
from p2p_energy_trading.exceptions import ConfigNotFoundError, ConfigValidationError
from p2p_energy_trading.training.config_loader import load_training_config


def _flatten_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Helper to flatten nested dictionaries into dot-separated keys."""
    flat = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, key))
        else:
            flat[key] = v
    return flat


class ConfigService:
    """Provides configuration upload, validation, comparison, and exporting services."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize configurations storage folder.

        Args:
            base_dir: Registry root directory experiments/.
        """
        self.base_dir = base_dir
        self.configs_dir = self.base_dir / "configs"
        self.configs_dir.mkdir(exist_ok=True)

    def upload(self, config_path: str, name: str | None = None) -> ConfigInfo:
        """Register and store a validation-passing configuration YAML in registry.

        Args:
            config_path: Source path to the YAML file.
            name: Optional descriptive template name.

        Returns:
            Uploaded ConfigInfo metadata.
        """
        path = Path(config_path)
        if not path.exists():
            raise ConfigNotFoundError(f"Configuration file not found: {config_path}")

        # Validate before storing
        val_res = self.validate(config_path)
        if not val_res.valid:
            raise ConfigValidationError(
                f"Configuration failed validation:"
                f" {[e.message for e in val_res.errors]}"
            )

        with open(path, encoding="utf-8") as f:
            content = f.read()

        config_id = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        dest_path = self.configs_dir / f"{config_id}.yaml"

        shutil.copy(path, dest_path)

        return ConfigInfo(
            config_id=config_id,
            name=name or path.name,
            path=str(dest_path),
            schema_version="1.0",
            validated=True,
            uploaded_at=datetime.datetime.now().isoformat(),
        )

    def validate(self, config_path: str) -> ValidationResult:
        """Validate YAML configuration file against Module 8 validation rules.

        Returns:
            ValidationResult with list of errors if invalid.
        """
        if not os.path.exists(config_path):
            return ValidationResult(
                valid=False,
                errors=[
                    ValidationError(
                        field="config_path",
                        message=f"Config file not found at: '{config_path}'",
                        value=config_path,
                        expected="Valid file path",
                    )
                ],
            )

        try:
            # Reuses Module 8's loading and validate_config pipeline
            load_training_config(config_path)
            return ValidationResult(valid=True)
        except (ValueError, TypeError, FileNotFoundError) as e:
            msg = str(e)
            # Infer field if possible
            field_name = "unknown"
            if "must be" in msg:
                field_name = msg.split(" must be")[0].strip()
            elif "missing" in msg:
                field_name = "sections"

            err = ValidationError(
                field=field_name,
                message=msg,
                value=None,
                expected="Schema-compliant value",
            )
            return ValidationResult(valid=False, errors=[err])

    def compare(self, config_a_path: str, config_b_path: str) -> ComparisonResult:
        """Compare two configurations and classify diffs into breaking or safe.

        Breaking changes are neural network architectures/spaces that prevent resume.
        Safe changes are hyperparameters/epochs/learning rates.
        """
        # Reuses load_training_config to load parsed and validated dictionaries
        try:
            cfg_a = load_training_config(config_a_path)
        except Exception as e:
            raise ConfigValidationError(f"Config A load failed: {e}") from e

        try:
            cfg_b = load_training_config(config_b_path)
        except Exception as e:
            raise ConfigValidationError(f"Config B load failed: {e}") from e

        flat_a = _flatten_dict(cfg_a)
        flat_b = _flatten_dict(cfg_b)

        differences = []
        safe_diffs = []
        breaking_diffs = []

        all_keys = set(flat_a.keys()).union(flat_b.keys())

        # Structural or space constraints that are fatal for policy resumption
        breaking_patterns = [
            "hidden_layers",
            "observation_space",
            "action_space",
            "num_agents",
            "policy_count",
            "model.type",
        ]

        for k in all_keys:
            val_a = flat_a.get(k)
            val_b = flat_b.get(k)

            if val_a != val_b:
                is_breaking = any(pattern in k for pattern in breaking_patterns)
                category = "breaking" if is_breaking else "safe"

                diff = ConfigDiff(
                    field=k,
                    value_a=val_a,
                    value_b=val_b,
                    category=category,
                )
                differences.append(diff)
                if is_breaking:
                    breaking_diffs.append(diff)
                else:
                    safe_diffs.append(diff)

        identical = len(differences) == 0
        return ComparisonResult(
            identical=identical,
            differences=differences,
            safe_differences=safe_diffs,
            breaking_differences=breaking_diffs,
        )

    def export(self, effective_config_path: str, dest_path: str | Path) -> str:
        """Copy the effective config file of an experiment to a destination path."""
        src = Path(effective_config_path)
        if not src.exists():
            raise ConfigNotFoundError(f"Effective config file not found at: '{src}'")

        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)
        return str(dest)
