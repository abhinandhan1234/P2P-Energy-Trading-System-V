"""Centralized Critic RLModule for Multi-Agent PPO training.

This module implements the actor and critic neural network architectures for MAPPO
using Ray RLlib's New API Stack. The actor network routes local observations to produce
continuous Gaussian actions, while the critic network routes global microgrid states
to produce value estimations (Centralized Training with Decentralized Execution).

Design reference: docs/module_7_mappo_integration.md, docs/rllib_compatibility_audit.md
"""

from __future__ import annotations

# standard library
import logging
from typing import Any

# third party
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    # third party
    from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
    from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
    from ray.rllib.utils.annotations import override

    RAY_AVAILABLE = True
except ImportError:
    TorchRLModule = None  # type: ignore
    ValueFunctionAPI = object  # type: ignore

    def override(parent_cls: Any) -> Any:
        def decorator(func: Any) -> Any:
            return func

        return decorator

    RAY_AVAILABLE = False


if RAY_AVAILABLE:

    class CentralizedCriticRLModule(TorchRLModule, ValueFunctionAPI):
        """RLModule for Centralized Training with Decentralized Execution (CTDE).

        Contains separate actor (23 -> 128 -> 128 -> 3) and critic
        (243 -> 256 -> 256 -> 1) networks. Actor receives local observations,
        while the critic receives global states.
        """

        def setup(self) -> None:
            """Instantiate PyTorch actor and critic neural network layers.

            Sets up MLP layers for actor, learnable log standard deviation,
            and MLP layers for critic.
            """
            # Actor network: 23-dim input -> 128 -> 128 -> 3-dim mean action output
            self.actor = nn.Sequential(
                nn.Linear(23, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 3),
            )

            # Learnable state-independent log standard deviation (init 0.0 -> std 1.0)
            self.log_std = nn.Parameter(torch.zeros(3))

            # Critic network: 243-dim input -> 256 -> 256 -> 1-dim value output
            self.critic = nn.Sequential(
                nn.Linear(243, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

        def _get_action_dist_inputs(self, batch: dict[str, Any]) -> torch.Tensor:
            """Compute action distribution inputs from the actor network.

            Args:
                batch: Dictionary containing observation batch.

            Returns:
                Concat tensor of mean and log std of shape (B, 6).
            """
            obs_entry = batch["obs"]
            if isinstance(obs_entry, dict) or hasattr(obs_entry, "keys"):
                obs = obs_entry["obs"]
            else:
                obs = obs_entry

            if not isinstance(obs, torch.Tensor):
                obs = torch.as_tensor(obs, dtype=torch.float32)
            else:
                obs = obs.to(dtype=torch.float32)

            mean = self.actor(obs)
            log_std = self.log_std.expand_as(mean)
            return torch.cat([mean, log_std], dim=-1)

        @override(TorchRLModule)
        def forward_exploration(
            self, batch: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            """Compute forward pass for exploration.

            Args:
                batch: Trajectory batch from env runner.

            Returns:
                Dict containing action distribution parameters.
            """
            action_dist_inputs = self._get_action_dist_inputs(batch)
            return {"action_dist_inputs": action_dist_inputs}

        @override(TorchRLModule)
        def _forward_exploration(
            self, batch: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            return self.forward_exploration(batch, **kwargs)

        @override(TorchRLModule)
        def forward_inference(
            self, batch: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            """Compute forward pass for deterministic policy inference.

            Args:
                batch: Trajectory batch.

            Returns:
                Dict containing action distribution parameters.
            """
            action_dist_inputs = self._get_action_dist_inputs(batch)
            return {"action_dist_inputs": action_dist_inputs}

        @override(TorchRLModule)
        def _forward_inference(
            self, batch: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            return self.forward_inference(batch, **kwargs)

        @override(TorchRLModule)
        def forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            """Compute forward pass for neural network training updates.

            Actor utilizes local observation tensors, while the critic estimates value
            predictions from the global microgrid state tensor.

            Args:
                batch: Training batch containing obs/state dictionaries.

            Returns:
                Dict containing action distribution parameters and value predictions.
            """
            action_dist_inputs = self._get_action_dist_inputs(batch)

            obs_entry = batch["obs"]
            if isinstance(obs_entry, dict) or hasattr(obs_entry, "keys"):
                state = obs_entry["state"]
            else:
                state = obs_entry

            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            else:
                state = state.to(dtype=torch.float32)

            vf_preds = self.critic(state)
            return {
                "action_dist_inputs": action_dist_inputs,
                "vf_preds": vf_preds.squeeze(-1),
            }

        @override(TorchRLModule)
        def _forward_train(
            self, batch: dict[str, Any], **kwargs: Any
        ) -> dict[str, Any]:
            return self.forward_train(batch, **kwargs)

        def compute_values(
            self, batch: dict[str, Any], embeddings: Any | None = None
        ) -> torch.Tensor:
            """Estimate state values using the centralized critic network.

            Args:
                batch: Dictionary containing observation and state tensors.
                embeddings: Optional pre-computed embeddings.

            Returns:
                Centralized state value predictions as a 1D tensor.
            """
            obs_entry = batch["obs"]
            if isinstance(obs_entry, dict) or hasattr(obs_entry, "keys"):
                state = obs_entry["state"]
            else:
                state = obs_entry

            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            else:
                state = state.to(dtype=torch.float32)

            vf_preds = self.critic(state)
            return vf_preds.squeeze(-1)
else:

    class CentralizedCriticRLModule:  # type: ignore
        """Placeholder class that raises ImportError if Ray is absent."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "Ray/RLlib is not installed in the current Python environment. "
                "CentralizedCriticRLModule requires Ray/RLlib to be instantiated."
            )
