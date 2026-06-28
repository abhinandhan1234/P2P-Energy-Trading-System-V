"""Integration test for Ray RLlib MAPPO configuration and environment.

Runs a single training iteration of PPO using the custom P2PEnergyTradingEnv environment
and the custom CentralizedCriticRLModule. Verifies that the training iteration completes
successfully without errors, and that it returns a valid RLlib result dictionary.

Design reference: docs/module_7_mappo_integration.md
docs/module_11_implementation_roadmap.md
"""

from __future__ import annotations

# third party
import pytest

# local
from p2p_energy_trading.rl.centralized_critic import RAY_AVAILABLE


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray is not installed")
def test_rllib_integration_one_iteration() -> None:
    """Validate PPO config and complete one training iteration.

    Verifies the return of a valid training result dictionary with metrics.
    """
    # third party
    import ray

    # local
    from p2p_energy_trading.rl.policy_config import build_ppo_config

    # 1. Initialize Ray if not already active
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
        )

    try:
        # Use debug/minimum sizes for speed and memory efficiency
        env_config = {
            "episode_length": 24,  # 1 day debug length
            "pandapower_bypass": True,  # Bypass powerflow for speed
        }

        ppo_config = {
            "lr": 3e-4,
            "train_batch_size_per_learner": 504,  # 24 steps * 21 agents
            "sgd_minibatch_size": 252,
            "num_sgd_iter": 1,
            "rollout_fragment_length": 24,
            "gamma": 0.99,
        }

        hardware_config = {
            "num_env_runners": 0,  # Local execution runner for speed
            "num_envs_per_env_runner": 1,
            "num_learner_workers": 0,  # Local learner
            "num_gpus_per_learner_worker": 0,  # Use CPU for speed on CPU runners
        }

        # 2. Build the RLlib algorithm configuration
        config = build_ppo_config(env_config, ppo_config, hardware_config)

        # 3. Instantiate the algorithm
        algo = config.build()

        # 4. Execute one training iteration
        result = algo.train()

        # 5. Verify the full public contract of the training result
        assert isinstance(result, dict)
        assert "training_iteration" in result
        assert result["training_iteration"] == 1

        # Check standard training keys are populated
        assert any(k in result for k in ["info", "learner", "hist_stats", "timers"])

        # Verify result contains episode statistics
        assert "episode_reward_mean" in result or (
            "env_runners" in result and "episode_return_mean" in result["env_runners"]
        )

        algo.stop()

    finally:
        # Shutdown Ray to clean up resources
        if ray.is_initialized():
            ray.shutdown()
