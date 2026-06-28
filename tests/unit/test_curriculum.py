"""Unit tests for curriculum.py."""

from __future__ import annotations

# third party
import pytest

# local
from p2p_energy_trading.constants import (
    POLICY_COLLEGE,
    POLICY_CONSUMER,
    POLICY_SOLAR,
)
from p2p_energy_trading.training.curriculum import CurriculumManager


@pytest.fixture
def base_config() -> dict:
    """Provide a standard configuration structure."""
    return {
        "curriculum": {
            "stages": [
                {
                    "name": "debug",
                    "episode_length": 24,
                    "pandapower_bypass": True,
                    "reward_phase": 1,
                    "curriculum_transition_step": 100000,
                },
                {
                    "name": "training",
                    "episode_length": 168,
                    "pandapower_bypass": False,
                    "reward_phase": 1,
                    "curriculum_transition_step": 2000000,
                },
                {
                    "name": "constraint_aware",
                    "episode_length": 168,
                    "pandapower_bypass": False,
                    "reward_phase": 2,
                },
            ]
        },
        "training": {
            "auto_curriculum": True,
        },
    }


def test_stage_overrides_generation(base_config: dict) -> None:
    """Verify that stage configuration overrides are correctly parsed."""
    manager = CurriculumManager(base_config)

    debug_overrides = manager.get_stage_overrides("debug")
    assert debug_overrides["episode_length"] == 24
    assert debug_overrides["pandapower_bypass"] is True
    assert debug_overrides["curriculum_transition_step"] == 100000

    constraint_overrides = manager.get_stage_overrides("constraint_aware")
    assert constraint_overrides["episode_length"] == 168
    assert constraint_overrides["pandapower_bypass"] is False
    assert constraint_overrides["curriculum_transition_step"] == 0  # Forced Phase 2


def test_debug_to_training_transition(base_config: dict) -> None:
    """Verify debug to training stage progression conditions."""
    manager = CurriculumManager(base_config)

    # 1. Start with initial high entropy, low reward
    results_initial = {
        "episode_reward_mean": -5.0,
        "episodes_truncated": 0,
        "learner": {
            POLICY_COLLEGE: {"entropy": 1.0, "total_loss": 0.5},
            POLICY_SOLAR: {"entropy": 1.0, "total_loss": 0.5},
            POLICY_CONSUMER: {"entropy": 1.0, "total_loss": 0.5},
        },
    }

    # Simulate 5 iterations of initialization
    for _ in range(5):
        manager.check_progression(
            results_initial, "debug", total_episodes=50, total_steps=1000
        )

    # 2. Results show entropy decreasing and reward increasing
    results_improved = {
        "episode_reward_mean": 2.0,
        "episodes_truncated": 0,
        "learner": {
            POLICY_COLLEGE: {"entropy": 0.5, "total_loss": 0.1},
            POLICY_SOLAR: {"entropy": 0.5, "total_loss": 0.1},
            POLICY_CONSUMER: {"entropy": 0.5, "total_loss": 0.1},
        },
    }

    # Run check before episode threshold is met
    transitioned, next_stage = manager.check_progression(
        results_improved, "debug", total_episodes=50, total_steps=2000
    )
    assert not transitioned
    assert next_stage == "debug"

    # Simulate more iterations to populate history
    for _ in range(4):
        manager.check_progression(
            results_improved, "debug", total_episodes=110, total_steps=3000
        )

    # Final check with episode count met
    transitioned, next_stage = manager.check_progression(
        results_improved, "debug", total_episodes=110, total_steps=4000
    )
    assert transitioned
    assert next_stage == "training"


def test_training_to_constraint_aware_transition(base_config: dict) -> None:
    """Verify training to constraint-aware stage progression conditions."""
    manager = CurriculumManager(base_config)

    # Populate 25 iterations of history (must be >= 20)
    results = {
        "episode_reward_mean": 1.5,
        "policy_reward_mean": {
            POLICY_COLLEGE: 0.8,
            POLICY_SOLAR: 0.9,
            POLICY_CONSUMER: 0.4,
        },
        "custom_metrics": {
            "p2p_volume_total_mean": 0.7,
        },
    }

    for i in range(25):
        # High variance for early iterations, zero variance for later iterations
        reward = 1.0 if i < 10 and i % 2 == 0 else 1.5
        iter_results = dict(results)
        iter_results["episode_reward_mean"] = reward
        manager.check_progression(
            iter_results, "training", total_episodes=200, total_steps=100000
        )

    # Run check before step threshold is reached
    transitioned, next_stage = manager.check_progression(
        results, "training", total_episodes=300, total_steps=1500000
    )
    assert not transitioned

    # Run check after step threshold is met (2M)
    transitioned, next_stage = manager.check_progression(
        results, "training", total_episodes=300, total_steps=2100000
    )
    assert transitioned
    assert next_stage == "constraint_aware"


def test_constraint_aware_convergence(base_config: dict) -> None:
    """Verify convergence criteria checks in constraint_aware stage."""
    manager = CurriculumManager(base_config)

    # Populate stable reward history (last 20 iterations)
    for r in [10.0] * 20:
        manager.history_reward.append(r)

    # Setup healthy entropy
    for pid in manager.history_entropy:
        manager.history_entropy[pid] = [0.05] * 20

    # Under-threshold metrics
    results_not_converged = {
        "custom_metrics": {
            "p2p_volume_total_mean": 0.5,  # Needs > 60%
            "voltage_violations_total_mean": 5.0,  # Needs low
            "thermal_violations_total_mean": 2.0,
        }
    }
    assert not manager.is_converged(results_not_converged)

    # Meeting convergence metrics
    results_converged = {
        "custom_metrics": {
            "p2p_volume_total_mean": 0.75,
            "voltage_violations_total_mean": 0.2,
            "thermal_violations_total_mean": 0.1,
        }
    }
    assert manager.is_converged(results_converged)
