"""PPO Configuration Builder for Ray RLlib New API Stack.

This module provides the helper function `build_ppo_config` to construct
and configure the Proximal Policy Optimization (PPO) algorithm for the
21-agent campus microgrid environment. It registers the custom environment,
defines policy sharing groups, sets up the custom Multi-Agent RLModule Spec,
and configures hardware resources.

Design reference: docs/module_7_mappo_integration.md
docs/rllib_compatibility_audit.md
"""

from __future__ import annotations

# standard library
import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    # standard library
    import importlib

    # third party
    import torch
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.tune.registry import register_env

    # Resolve RLModuleSpec dynamically
    RLModuleSpec = None
    rl_module_spec_candidates = [
        ("ray.rllib.core.rl_module.rl_module", "RLModuleSpec"),
        ("ray.rllib.core.rl_module", "RLModuleSpec"),
    ]
    for module_path, class_name in rl_module_spec_candidates:
        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, class_name):
                RLModuleSpec = getattr(mod, class_name)
                break
        except ImportError:
            continue

    # Resolve MultiAgentRLModuleSpec / MultiRLModuleSpec dynamically
    MultiAgentRLModuleSpec = None
    marl_spec_candidates = [
        ("ray.rllib.core.rl_module.rl_module", "MultiAgentRLModuleSpec"),
        ("ray.rllib.core.rl_module.multi_rl_module", "MultiRLModuleSpec"),
        ("ray.rllib.core.rl_module.rl_module", "MultiRLModuleSpec"),
        ("ray.rllib.core.rl_module", "MultiRLModuleSpec"),
        ("ray.rllib.core.rl_module", "MultiAgentRLModuleSpec"),
    ]
    for module_path, class_name in marl_spec_candidates:
        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, class_name):
                MultiAgentRLModuleSpec = getattr(mod, class_name)
                break
        except ImportError:
            continue

    if RLModuleSpec is None or MultiAgentRLModuleSpec is None:
        raise ImportError(
            "Could not resolve RLModuleSpec or "
            "MultiAgentRLModuleSpec in version-resilient candidates."
        )

    RAY_AVAILABLE = True
except ImportError:
    PPOConfig = None  # type: ignore
    RLModuleSpec = None  # type: ignore
    MultiAgentRLModuleSpec = None  # type: ignore
    register_env = None  # type: ignore
    RAY_AVAILABLE = False


def policy_mapping_fn(agent_id: str, episode: Any = None, **kwargs: Any) -> str:
    """Map an agent ID to its corresponding policy ID.

    Omit deprecated 'worker' argument per R-06 of the compatibility audit.

    Args:
        agent_id: String ID of the agent (e.g. "college", "solar_03").
        episode: Episode object (unused but required by RLlib signature).
        kwargs: Keyword arguments for future compatibility.

    Returns:
        Policy ID string matching the agent type group.

    Raises:
        ValueError: If the agent ID prefix is unrecognized.
    """
    # local
    from p2p_energy_trading.constants import (
        POLICY_COLLEGE,
        POLICY_CONSUMER,
        POLICY_SOLAR,
    )

    if agent_id == "college":
        return POLICY_COLLEGE
    elif agent_id.startswith("solar_"):
        return POLICY_SOLAR
    elif agent_id.startswith("consumer_"):
        return POLICY_CONSUMER
    else:
        raise ValueError(f"Unrecognized agent_id: {agent_id}")


def build_ppo_config(
    env_config: dict[str, Any],
    ppo_config: dict[str, Any],
    hardware_config: dict[str, Any],
) -> Any:
    """Build the PPO configuration for RLlib New API Stack.

    Conforms to RLlib New API Stack requirements specified in the compatibility audit.

    Args:
        env_config: Environment configuration dictionary.
        ppo_config: PPO algorithm specific hyperparameters.
        hardware_config: Hardware worker/learner parameters.

    Returns:
        Built and configured PPOConfig object.

    Raises:
        ImportError: If Ray/RLlib is not installed in the environment.
    """
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray/RLlib is not installed in the current Python environment. "
            "build_ppo_config requires Ray/RLlib to run."
        )

    # local
    from p2p_energy_trading.constants import (
        ENV_NAME,
        POLICY_COLLEGE,
        POLICY_CONSUMER,
        POLICY_SOLAR,
    )
    from p2p_energy_trading.environment.env import P2PEnergyTradingEnv
    from p2p_energy_trading.rl.centralized_critic import CentralizedCriticRLModule

    # 1. Register environment with Tune registry
    from p2p_energy_trading.rl.env_registration import register_p2p_environment

    register_p2p_environment()

    # 2. Get environment action/observation spaces for policy specifications
    temp_env = P2PEnergyTradingEnv(env_config)
    observation_space = temp_env.observation_space
    action_space = temp_env.action_space
    temp_env.close()

    # 3. Define multi-agent policy specifications
    # Using None for policy_class allows the New API Stack to use the
    # default policy logic
    policies = {
        POLICY_COLLEGE: (None, observation_space, action_space, {}),
        POLICY_SOLAR: (None, observation_space, action_space, {}),
        POLICY_CONSUMER: (None, observation_space, action_space, {}),
    }

    # 4. Construct PPO configuration builder
    config = PPOConfig()

    # 5. Enable the RLModule and Learner stack dynamically in a version-resilient way
    if hasattr(config, "api_stack"):
        config = config.api_stack(enable_rl_module_and_learner=True)
    elif hasattr(config, "experimental"):
        config = config.experimental(enable_rl_module_and_learner=True)

    # 6. Apply configuration settings
    config = (
        config.environment(
            env=ENV_NAME,
            env_config=env_config,
            disable_env_checking=True,
        )
        .training(
            lr=ppo_config.get("lr", 3e-4),
            train_batch_size_per_learner=ppo_config.get(
                "train_batch_size_per_learner", 7056
            ),
            minibatch_size=ppo_config.get("sgd_minibatch_size", 512),
            num_epochs=ppo_config.get("num_sgd_iter", 10),
            clip_param=ppo_config.get("clip_param", 0.2),
            lambda_=ppo_config.get("gae_lambda", 0.95),
            gamma=ppo_config.get("gamma", 0.99),
            vf_loss_coeff=ppo_config.get("vf_loss_coeff", 0.5),
            entropy_coeff=ppo_config.get("entropy_coeff", 0.01),
        )
        .env_runners(
            num_env_runners=hardware_config.get("num_env_runners", 2),
            num_envs_per_env_runner=hardware_config.get("num_envs_per_env_runner", 1),
            num_gpus_per_env_runner=0,
            rollout_fragment_length=ppo_config.get("rollout_fragment_length", 168),
            batch_mode="complete_episodes",
        )
        .learners(
            num_learners=hardware_config.get("num_learner_workers", 0),
            num_gpus_per_learner=(1 if torch.cuda.is_available() else 0),
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[POLICY_COLLEGE, POLICY_SOLAR, POLICY_CONSUMER],
        )
    )

    # 7. Set up custom Multi-Agent RLModule Spec
    module_specs = {
        POLICY_COLLEGE: RLModuleSpec(
            module_class=CentralizedCriticRLModule,
            observation_space=observation_space,
            action_space=action_space,
        ),
        POLICY_SOLAR: RLModuleSpec(
            module_class=CentralizedCriticRLModule,
            observation_space=observation_space,
            action_space=action_space,
        ),
        POLICY_CONSUMER: RLModuleSpec(
            module_class=CentralizedCriticRLModule,
            observation_space=observation_space,
            action_space=action_space,
        ),
    }

    sig = inspect.signature(MultiAgentRLModuleSpec.__init__)
    if "rl_module_specs" in sig.parameters:
        rl_module_spec = MultiAgentRLModuleSpec(rl_module_specs=module_specs)
    else:
        rl_module_spec = MultiAgentRLModuleSpec(module_specs=module_specs)
    config = config.rl_module(rl_module_spec=rl_module_spec)

    return config
