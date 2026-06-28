"""Curriculum Manager for the P2P Energy Trading training pipeline.

Automates progression through three training stages (Debug, Training, and Constraint-Aware)
based on performance and stability metrics (episode count, policy entropy decay, reward
improvement, grid violations, and P2P utilization ratios).

Design reference: docs/module_8_training_pipeline.md §6
"""

from __future__ import annotations

# standard library
import logging
import math
from typing import Any

# local
from p2p_energy_trading.constants import (
    POLICY_COLLEGE,
    POLICY_CONSUMER,
    POLICY_SOLAR,
)

logger = logging.getLogger(__name__)


class CurriculumManager:
    """Manages training stage progression and checks convergence criteria."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the CurriculumManager.

        Args:
            config: Loaded and validated configuration dictionary.
        """
        self.config = config
        self.curriculum_config = config["curriculum"]
        self.training_control = config.get("training", {})

        # Track history for progression checks
        self.history_entropy: dict[str, list[float]] = {
            POLICY_COLLEGE: [],
            POLICY_SOLAR: [],
            POLICY_CONSUMER: [],
        }
        self.history_reward: list[float] = []
        self.history_truncated: list[int] = []

    def get_policy_entropy(
        self, results: dict[str, Any], policy_id: str
    ) -> float | None:
        """Retrieve policy entropy value from RLlib training results.

        Args:
            results: RLlib results dictionary.
            policy_id: Policy ID string.

        Returns:
            The entropy value or None if not found.
        """
        # Look in standard RLlib 2.x New API Stack locations
        paths = [
            ["learner", policy_id, "entropy"],
            ["learner", policy_id, "learner_stats", "entropy"],
            ["info", "learner", policy_id, "entropy"],
            ["info", "learner", policy_id, "learner_stats", "entropy"],
        ]
        for path in paths:
            val = results
            try:
                for key in path:
                    val = val[key]
                return float(val)
            except (KeyError, TypeError):
                continue
        return None

    def get_custom_metric(
        self, results: dict[str, Any], metric_name: str
    ) -> float | None:
        """Retrieve custom metrics (e.g. from callbacks) from training results.

        Args:
            results: RLlib results dictionary.
            metric_name: Name of the custom metric.

        Returns:
            The metric value or None if not found.
        """
        custom = results.get("custom_metrics", {})
        if metric_name in custom:
            return float(custom[metric_name])
        mean_name = f"{metric_name}_mean"
        if mean_name in custom:
            return float(custom[mean_name])
        return None

    def get_stage_overrides(self, stage_name: str) -> dict[str, Any]:
        """Get environment and runner overrides for the given curriculum stage.

        Args:
            stage_name: Name of the stage (debug, training, constraint_aware).

        Returns:
            Dictionary of config overrides.

        Raises:
            ValueError: If the stage name is unrecognized.
        """
        for stage in self.curriculum_config["stages"]:
            if stage["name"] == stage_name:
                overrides = {
                    "episode_length": stage.get("episode_length", 168),
                    "pandapower_bypass": stage.get("pandapower_bypass", False),
                }
                # Map reward phase: 1 -> Phase 1, 2 -> Phase 2
                # Phase 2 sets curriculum_transition_step to 0 to bypass Phase 1
                if stage.get("reward_phase") == 2:
                    overrides["curriculum_transition_step"] = 0
                else:
                    # In Stage 2, transition step is read from config or defaulted to 2M
                    overrides["curriculum_transition_step"] = stage.get(
                        "curriculum_transition_step", 2000000
                    )
                return overrides
        raise ValueError(f"Unrecognized curriculum stage name: '{stage_name}'")

    def check_progression(
        self,
        results: dict[str, Any],
        current_stage: str,
        total_episodes: int,
        total_steps: int,
    ) -> tuple[bool, str]:
        """Evaluate progression criteria and return if a stage transition is required.

        Args:
            results: Latest training iteration results dictionary.
            current_stage: Name of the active stage.
            total_episodes: Cumulative count of training episodes completed.
            total_steps: Cumulative count of environment steps completed.

        Returns:
            Tuple containing:
            - bool: True if progression criteria are met and stage transitions should trigger.
            - str: Name of the next stage (or the current stage if no transition).
        """
        # Append iteration metrics to history
        for pid in self.history_entropy:
            ent = self.get_policy_entropy(results, pid)
            if ent is not None:
                self.history_entropy[pid].append(ent)

        r_mean = results.get("episode_reward_mean") or results.get("info", {}).get(
            "episode_reward_mean"
        )
        if r_mean is not None:
            self.history_reward.append(float(r_mean))

        self.history_truncated.append(results.get("episodes_truncated", 0))

        # Check if auto_curriculum is enabled
        if not self.training_control.get("auto_curriculum", True):
            return False, current_stage

        if current_stage == "debug":
            # Criteria 1: At least 100 episodes completed
            if total_episodes < 100:
                return False, current_stage

            # Criteria 2: Policy entropy is decreasing (requires at least 10 iterations of history)
            if len(self.history_entropy[POLICY_COLLEGE]) < 10:
                return False, current_stage

            entropy_decay = True
            for pid, h in self.history_entropy.items():
                recent_avg = sum(h[-5:]) / 5.0
                initial_avg = sum(h[:5]) / 5.0
                if recent_avg >= initial_avg:
                    entropy_decay = False
                    break

            # Criteria 3: Mean episode reward is increasing
            reward_gain = False
            if len(self.history_reward) >= 10:
                recent_reward = sum(self.history_reward[-5:]) / 5.0
                initial_reward = sum(self.history_reward[:5]) / 5.0
                if recent_reward > initial_reward:
                    reward_gain = True

            # Criteria 4: No environment errors/truncations in the last 5 iterations
            no_truncations = False
            if len(self.history_truncated) >= 5:
                if self.history_truncated[-1] == self.history_truncated[-5]:
                    no_truncations = True

            # Criteria 5: Training loss contains no NaNs
            no_nans = True
            for k in ["loss", "policy_loss", "vf_loss"]:
                for pid in self.history_entropy:
                    val = results.get("learner", {}).get(pid, {}).get(k)
                    if val is not None and (math.isnan(val) or math.isinf(val)):
                        no_nans = False

            if entropy_decay and reward_gain and no_truncations and no_nans:
                logger.info("Curriculum progression met: debug -> training")
                return True, "training"

        elif current_stage == "training":
            # Criteria 1: P2P volume ratio > 50%
            p2p_util = self.get_custom_metric(results, "p2p_volume_total")
            p2p_sufficient = p2p_util is None or p2p_util > 0.5
            # If we don't have enough statistics yet, require iterations
            if len(self.history_reward) < 20:
                return False, current_stage

            # Criteria 2: Mean reward positive for all policy types
            rewards_positive = True
            for pid in self.history_entropy:
                # Check custom metrics or policy_reward_mean
                p_rew = results.get("policy_reward_mean", {}).get(pid)
                if p_rew is not None and p_rew <= 0:
                    rewards_positive = False

            # Criteria 3: Reward variance is decreasing
            variance_decay = False
            if len(self.history_reward) >= 20:
                recent_var = self._compute_variance(self.history_reward[-10:])
                early_var = self._compute_variance(self.history_reward[:10])
                if recent_var < early_var:
                    variance_decay = True

            # Criteria 4: Phase 2 transition step has been reached
            phase_transition_done = total_steps >= self.config["curriculum"]["stages"][
                1
            ].get("curriculum_transition_step", 2000000)

            if (
                rewards_positive
                and variance_decay
                and phase_transition_done
                and p2p_sufficient
            ):
                logger.info("Curriculum progression met: training -> constraint_aware")
                return True, "constraint_aware"

        return False, current_stage

    def is_converged(self, results: dict[str, Any]) -> bool:
        """Evaluate convergence criteria in the final stage.

        Args:
            results: Latest training iteration results dictionary.

        Returns:
            True if all convergence criteria are satisfied.
        """
        # Criteria 1: Mean episode reward stable over last 1000 episodes (represented by last 20 iterations)
        if len(self.history_reward) < 20:
            return False

        recent_mean = sum(self.history_reward[-10:]) / 10.0
        prior_mean = sum(self.history_reward[-20:-10]) / 10.0
        reward_change = abs(recent_mean - prior_mean) / max(abs(prior_mean), 1e-5)
        reward_stable = reward_change < 0.01

        # Criteria 2: Grid violation rate < 1% (read from custom metrics)
        voltage_violations = self.get_custom_metric(results, "voltage_violations_total")
        thermal_violations = self.get_custom_metric(results, "thermal_violations_total")
        # Check if they are low
        violations_low = False
        if voltage_violations is not None and thermal_violations is not None:
            # We check if mean violations per episode are very low (< 1.68 timesteps out of 168)
            if (voltage_violations + thermal_violations) < 1.68:
                violations_low = True
        else:
            violations_low = True  # default to pass if metrics are not logged

        # Criteria 3: P2P utilization ratio > 60%
        p2p_util_metric = self.get_custom_metric(results, "p2p_volume_total")
        p2p_sufficient = True
        if p2p_util_metric is not None and p2p_util_metric < 0.6:
            p2p_sufficient = False

        # Criteria 4: Policy entropy above floor (0.01)
        entropy_healthy = True
        for pid in self.history_entropy:
            if self.history_entropy[pid]:
                if self.history_entropy[pid][-1] <= 0.01:
                    entropy_healthy = False
                    break

        return reward_stable and violations_low and p2p_sufficient and entropy_healthy

    @staticmethod
    def _compute_variance(data: list[float]) -> float:
        """Helper to compute variance of a list of numbers."""
        n = len(data)
        if n < 2:
            return 0.0
        mean = sum(data) / n
        return sum((x - mean) ** 2 for x in data) / (n - 1)
