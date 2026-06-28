"""Checkpoint Manager for the P2P Energy Trading training pipeline.

Orchestrates saving, loading, metadata logging, and pruning of RLlib algorithm checkpoints.
Supports periodic, stage-transition, and emergency checkpoints, as well as saving lightweight,
policy-weights-only deployments for the best model.

Design reference: docs/module_8_training_pipeline.md §7
"""

from __future__ import annotations

# standard library
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

# third party
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages training checkpoint lifecycles and prunes old periodic files."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the CheckpointManager.

        Args:
            config: Loaded and validated configuration dictionary.
        """
        self.config = config
        self.checkpoint_config = config.get("checkpoint", {})

        # Resolve paths
        self.checkpoint_dir = Path(
            self.checkpoint_config.get("checkpoint_dir", "checkpoints")
        )
        self.best_model_dir = Path(
            self.checkpoint_config.get("best_model_dir", "checkpoints/best_model")
        )
        self.keep_last_n = int(self.checkpoint_config.get("keep_last_n", 5))

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        self.best_eval_reward = -float("inf")

    def _save_algo(self, algo: Any, target_dir: Path) -> str:
        """Internal helper to call the correct RLlib save method."""
        target_dir.mkdir(parents=True, exist_ok=True)
        target_str = str(target_dir)

        # RLlib New API Stack Algorithm checkpoint saving
        if hasattr(algo, "save"):
            checkpoint_path = algo.save(checkpoint_dir=target_str)
        elif hasattr(algo, "save_checkpoint"):
            checkpoint_path = algo.save_checkpoint(target_str)
        else:
            raise AttributeError(
                "RLlib Algorithm object does not have a save or save_checkpoint method."
            )

        return checkpoint_path

    def save_periodic_checkpoint(
        self,
        algo: Any,
        iteration: int,
        steps: int,
        stage: str,
    ) -> str:
        """Save a periodic checkpoint to checkpoints/checkpoint_{iteration:06d}/.

        Args:
            algo: The RLlib Algorithm instance.
            iteration: Current training iteration number.
            steps: Cumulative count of steps.
            stage: Active curriculum stage name.

        Returns:
            The path to the saved checkpoint directory.
        """
        target_dir = self.checkpoint_dir / f"checkpoint_{iteration:06d}"
        checkpoint_path = self._save_algo(algo, target_dir)

        # Save config snapshot for reproducibility
        self._write_metadata(target_dir, iteration, steps, stage)

        logger.info("Saved periodic checkpoint: '%s'", checkpoint_path)
        return checkpoint_path

    def save_stage_checkpoint(
        self,
        algo: Any,
        iteration: int,
        steps: int,
        stage: str,
    ) -> str:
        """Save a checkpoint at a curriculum stage transition.

        Args:
            algo: The RLlib Algorithm instance.
            iteration: Current training iteration number.
            steps: Cumulative count of steps.
            stage: Curriculum stage name that has been completed.

        Returns:
            The path to the saved checkpoint directory.
        """
        target_dir = self.checkpoint_dir / f"stage_{stage}_{iteration:06d}"
        checkpoint_path = self._save_algo(algo, target_dir)

        self._write_metadata(target_dir, iteration, steps, stage)

        logger.info(
            "Saved curriculum stage-transition checkpoint: '%s'", checkpoint_path
        )
        return checkpoint_path

    def save_emergency_checkpoint(self, algo: Any, iteration: int) -> str:
        """Save an emergency checkpoint on training interruption or failure.

        Args:
            algo: The RLlib Algorithm instance.
            iteration: Current training iteration number.

        Returns:
            The path to the saved emergency checkpoint directory.
        """
        target_dir = self.checkpoint_dir / "emergency_checkpoint"
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)

        checkpoint_path = self._save_algo(algo, target_dir)
        self._write_metadata(target_dir, iteration, 0, "emergency")

        logger.warning("Saved emergency checkpoint: '%s'", checkpoint_path)
        return checkpoint_path

    def save_best_checkpoint(
        self,
        algo: Any,
        iteration: int,
        steps: int,
        eval_reward: float,
        metrics: dict[str, Any],
    ) -> None:
        """Save policy weights and metadata JSON if the evaluation reward improves.

        Args:
            algo: The RLlib Algorithm instance.
            iteration: Current training iteration number.
            steps: Cumulative count of steps.
            eval_reward: The computed evaluation reward.
            metrics: Dict containing P2P utilization, grid violations, etc.
        """
        min_improvement = float(
            self.checkpoint_config.get("min_improvement_threshold", 0.01)
        )
        if eval_reward <= self.best_eval_reward + min_improvement:
            return

        self.best_eval_reward = eval_reward
        logger.info(
            "New best evaluation reward reached: %.3f (Previous best: %.3f). Saving best model.",
            eval_reward,
            self.best_eval_reward,
        )

        # Clear best model directory
        if self.best_model_dir.exists():
            shutil.rmtree(self.best_model_dir, ignore_errors=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Export policy weights only (deployment ready)
        weights = algo.get_weights()
        weights_path = self.best_model_dir / "weights.pt"
        torch.save(weights, weights_path)

        # 2. Write metadata JSON
        metadata = {
            "iteration": iteration,
            "agent_steps": steps,
            "mean_eval_reward": eval_reward,
            "p2p_utilisation_ratio": metrics.get("p2p_utilisation_ratio", 0.0),
            "grid_violation_rate": metrics.get("grid_violation_rate", 0.0),
            "total_campus_cost": metrics.get("total_campus_cost", 0.0),
        }

        with open(
            self.best_model_dir / "best_model_metadata.json", "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4)

        logger.info(
            "Best model weights and metadata exported to: '%s'", self.best_model_dir
        )

    def prune_checkpoints(self) -> None:
        """Prune older periodic checkpoints, retaining only the last keep_last_n files."""
        # Find all periodic checkpoints: checkpoints/checkpoint_XXXXXX/
        periodic_dirs: list[tuple[int, Path]] = []
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint_"):
                # Extract iteration number
                match = re.search(r"checkpoint_(\d+)", item.name)
                if match:
                    iter_num = int(match.group(1))
                    periodic_dirs.append((iter_num, item))

        # Sort by iteration number ascending
        periodic_dirs.sort(key=lambda x: x[0])

        # Delete older directories if limit exceeded
        if len(periodic_dirs) > self.keep_last_n:
            to_delete = periodic_dirs[: -self.keep_last_n]
            for iter_num, path in to_delete:
                try:
                    shutil.rmtree(path, ignore_errors=True)
                    logger.info("Pruned old periodic checkpoint: '%s'", path)
                except Exception as e:
                    logger.warning(
                        "Failed to prune checkpoint directory '%s': %s", path, e
                    )

    def _write_metadata(
        self,
        target_dir: Path,
        iteration: int,
        steps: int,
        stage: str,
    ) -> None:
        """Helper to write configuration and state snapshot metadata into the checkpoint directory."""
        metadata = {
            "iteration": iteration,
            "agent_steps": steps,
            "curriculum_stage": stage,
        }
        with open(target_dir / "checkpoint_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        # Write YAML config copy
        with open(
            target_dir / "training_config_snapshot.yaml", "w", encoding="utf-8"
        ) as f:
            # Simple conversion of config dict to yaml
            # third party
            import yaml

            yaml.safe_dump(self.config, f, default_flow_style=False)
