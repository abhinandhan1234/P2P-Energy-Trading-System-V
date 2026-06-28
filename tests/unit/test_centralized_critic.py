"""Unit tests for Module 7 (RLlib MAPPO Integration).

Verifies the actor and critic PyTorch network behavior, policy mapping, public API
exposure, configuration builder correctness, custom callback recording, and
import safety. If Ray is not installed, all Ray-dependent tests are skipped.

Design reference: docs/module_7_mappo_integration.md
docs/module_12_repository_structure.md
"""

from __future__ import annotations

# third party
import pytest
import torch

# local
import p2p_energy_trading.rl as rl
from p2p_energy_trading.rl.callbacks import P2PCallbacks
from p2p_energy_trading.rl.centralized_critic import (
    RAY_AVAILABLE,
    CentralizedCriticRLModule,
)
from p2p_energy_trading.rl.policy_config import build_ppo_config, policy_mapping_fn


def test_public_api_exposure() -> None:
    """Verify that p2p_energy_trading.rl exposes approved public components."""
    expected_api = {"CentralizedCriticRLModule", "build_ppo_config", "P2PCallbacks"}
    actual_api = set(rl.__all__)
    assert actual_api == expected_api
    assert hasattr(rl, "CentralizedCriticRLModule")
    assert hasattr(rl, "build_ppo_config")
    assert hasattr(rl, "P2PCallbacks")


def test_import_safety_when_ray_absent() -> None:
    """Verify that instantiating classes raises ImportError when Ray is absent."""
    if not RAY_AVAILABLE:
        # Verify instantiation of RLModule raises ImportError
        with pytest.raises(ImportError) as exc_info:
            CentralizedCriticRLModule()
        assert "requires Ray/RLlib" in str(exc_info.value)

        # Verify build_ppo_config raises ImportError
        with pytest.raises(ImportError) as exc_info_config:
            build_ppo_config({}, {}, {})
        assert "requires Ray/RLlib to run" in str(exc_info_config.value)

        # Verify P2PCallbacks instantiation raises ImportError
        with pytest.raises(ImportError) as exc_info_cb:
            P2PCallbacks()
        assert "requires Ray/RLlib" in str(exc_info_cb.value)


def test_policy_mapping_fn() -> None:
    """Verify that policy_mapping_fn correctly maps agents to policies."""
    # Test college agent mapping
    assert policy_mapping_fn("college") == "policy_college"

    # Test all 15 solar agents mapping
    for i in range(1, 16):
        assert policy_mapping_fn(f"solar_{i:02d}") == "policy_solar"

    # Test all 5 consumer agents mapping
    for i in range(1, 6):
        assert policy_mapping_fn(f"consumer_{i:02d}") == "policy_consumer"

    # Test unrecognized agent ID raises ValueError
    with pytest.raises(ValueError) as exc_info:
        policy_mapping_fn("unrecognized_agent")
    assert "Unrecognized agent_id" in str(exc_info.value)


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray is not installed")
def test_rl_module_behavior() -> None:
    """Verify behavioral outputs and gradient flow of CentralizedCriticRLModule."""
    # third party
    import gymnasium as gym
    import numpy as np
    from ray.rllib.core.rl_module.rl_module import RLModuleConfig

    # 1. Initialize RLModule with standard spaces config
    obs_space = gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(low=-1.0, high=1.0, shape=(23,), dtype=np.float32),
            "state": gym.spaces.Box(low=-1.0, high=1.0, shape=(243,), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)

    config = RLModuleConfig(
        observation_space=obs_space,
        action_space=action_space,
    )

    try:
        module = CentralizedCriticRLModule(
            observation_space=obs_space,
            action_space=action_space,
        )
    except (TypeError, ValueError):
        # Fallback for older Ray versions that only accepted config
        module = CentralizedCriticRLModule(config)
    module.setup()

    # 2. Construct mock input batch
    batch_size = 8
    batch = {
        "obs": {
            "obs": torch.randn(batch_size, 23),
            "state": torch.randn(batch_size, 243),
        }
    }

    # 3. Test forward_exploration returns (B, 6) distribution inputs
    outputs_expl = module.forward_exploration(batch)
    assert isinstance(outputs_expl, dict)
    assert "action_dist_inputs" in outputs_expl
    action_dist_inputs = outputs_expl["action_dist_inputs"]
    assert isinstance(action_dist_inputs, torch.Tensor)
    assert action_dist_inputs.shape == (batch_size, 6)
    assert action_dist_inputs.dtype == torch.float32

    # 4. Test forward_inference returns identical action distribution inputs
    outputs_inf = module.forward_inference(batch)
    assert isinstance(outputs_inf, dict)
    assert "action_dist_inputs" in outputs_inf
    assert outputs_inf["action_dist_inputs"].shape == (batch_size, 6)

    # 5. Test forward_train returns action distribution and value predictions
    outputs_train = module.forward_train(batch)
    assert isinstance(outputs_train, dict)
    assert "action_dist_inputs" in outputs_train
    assert "vf_preds" in outputs_train
    assert outputs_train["action_dist_inputs"].shape == (batch_size, 6)
    vf_preds = outputs_train["vf_preds"]
    assert isinstance(vf_preds, torch.Tensor)
    assert vf_preds.shape == (batch_size,)
    assert vf_preds.dtype == torch.float32
    # 6. Test compute_values returns squeezed critic estimations only
    outputs_val = module.compute_values(batch)
    assert isinstance(outputs_val, torch.Tensor)
    assert outputs_val.shape == (batch_size,)
    # 7. Test PyTorch backward pass / gradient flow through networks
    loss = outputs_train["action_dist_inputs"].sum() + outputs_train["vf_preds"].sum()
    loss.backward()

    # Ensure gradients propagate to actor layers
    for name, param in module.actor.named_parameters():
        assert param.grad is not None
        assert torch.nonzero(param.grad).size(0) > 0, (
            f"Actor layer {name} has zero gradients"
        )

    # Ensure gradients propagate to log_std parameter
    assert module.log_std.grad is not None
    assert torch.nonzero(module.log_std.grad).size(0) > 0

    # Ensure gradients propagate to critic layers
    for name, param in module.critic.named_parameters():
        assert param.grad is not None
        assert torch.nonzero(param.grad).size(0) > 0, (
            f"Critic layer {name} has zero gradients"
        )


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray is not installed")
def test_ppo_config_and_spec_integration() -> None:
    """Verify that RLModuleSpec, MultiAgentRLModuleSpec, and PPOConfig integrate."""
    # third party
    from ray.rllib.algorithms.ppo import PPOConfig

    env_config = {"episode_length": 168, "pandapower_bypass": True}
    ppo_config = {
        "lr": 3e-4,
        "train_batch_size_per_learner": 7056,
        "sgd_minibatch_size": 512,
        "num_sgd_iter": 10,
        "clip_param": 0.2,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.01,
        "rollout_fragment_length": 168,
    }
    hardware_config = {
        "num_env_runners": 2,
        "num_envs_per_env_runner": 1,
        "num_learner_workers": 0,
        "num_gpus_per_learner_worker": 0,
    }

    config = build_ppo_config(env_config, ppo_config, hardware_config)
    assert isinstance(config, PPOConfig)

    # Validate parameters propagated correctly
    assert config.lr == 3e-4
    assert config.train_batch_size_per_learner == 7056

    # Verify policies definition
    policies = config.policies
    assert "policy_college" in policies
    assert "policy_solar" in policies
    assert "policy_consumer" in policies

    # Verify custom Multi-Agent RLModule specifications
    rl_module_spec = config.rl_module_spec
    assert rl_module_spec is not None
    module_specs = rl_module_spec.module_specs
    assert "policy_college" in module_specs
    assert "policy_solar" in module_specs
    assert "policy_consumer" in module_specs
    assert module_specs["policy_college"].module_class is CentralizedCriticRLModule


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray is not installed")
def test_callbacks_metric_recording() -> None:
    """Verify that callbacks initialize and log microgrid metrics correctly."""
    # standard library
    from unittest.mock import MagicMock

    episode = MagicMock()
    episode.user_data = {}
    episode.custom_metrics = {}

    # Define mock step info returns
    step_info = {
        "college": {
            "net_cost": 10.0,
            "p2p_sold_kw": 2.0,
            "p2p_bought_kw": 0.0,
            "grid_sold_kw": 0.0,
            "grid_bought_kw": 5.0,
            "voltage_violation": True,
            "thermal_violation": False,
        },
        "solar_01": {
            "net_cost": -5.0,
            "p2p_sold_kw": 0.0,
            "p2p_bought_kw": 0.0,
            "grid_sold_kw": 3.0,
            "grid_bought_kw": 0.0,
            "voltage_violation": True,
            "thermal_violation": False,
        },
    }
    episode.get_infos.return_value = step_info

    callbacks = P2PCallbacks()

    # Test initialization
    callbacks.on_episode_start(episode=episode)
    assert episode.user_data["p2p_volume"] == 0.0
    assert episode.user_data["voltage_violations"] == 0
    assert "college" in episode.user_data["agent_net_cost"]

    # Test step-wise accumulation
    callbacks.on_episode_step(episode=episode)
    assert episode.user_data["p2p_volume"] == 2.0
    assert episode.user_data["grid_import"] == 5.0
    assert episode.user_data["grid_export"] == 3.0
    assert episode.user_data["agent_net_cost"]["college"] == 10.0
    assert episode.user_data["agent_net_cost"]["solar_01"] == -5.0
    assert episode.user_data["voltage_violations"] == 1
    assert episode.user_data["thermal_violations"] == 0

    # Test publication
    callbacks.on_episode_end(episode=episode)
    assert episode.custom_metrics["p2p_volume_total"] == 2.0
    assert episode.custom_metrics["grid_import_total"] == 5.0
    assert episode.custom_metrics["voltage_violations_total"] == 1
    assert episode.custom_metrics["net_cost_college"] == 10.0
    assert episode.custom_metrics["net_cost_solar"] == -5.0
    assert episode.custom_metrics["net_cost_total_campus"] == 5.0
