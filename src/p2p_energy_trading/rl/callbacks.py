"""RLlib Callbacks for monitoring P2P Energy Trading training.

This module implements the custom `P2PCallbacks` class for Ray RLlib's New API Stack.
It monitors and aggregates episode metrics such as economic costs, P2P trading volume,
grid import/export, physical violations, and curriculum phase transitions, writing them
to custom metrics for TensorBoard visualization. Callbacks are strictly read-only.

Design reference: docs/module_7_mappo_integration.md, docs/rllib_compatibility_audit.md
"""

from __future__ import annotations

# standard library
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    # third party
    from ray.rllib.callbacks.callbacks import RLlibCallback

    RAY_AVAILABLE = True
except ImportError:
    try:
        # third party
        from ray.rllib.algorithms.callbacks import (
            DefaultCallbacks as RLlibCallback,  # type: ignore
        )

        RAY_AVAILABLE = True
    except ImportError:
        RLlibCallback = None  # type: ignore
        RAY_AVAILABLE = False


if RAY_AVAILABLE:

    class P2PCallbacks(RLlibCallback):
        """Custom RLlib callbacks class for read-only microgrid metrics monitoring.

        Accumulates episode metrics and exports them to TensorBoard without mutating any
        environment logic or training state.
        """

        def on_episode_start(
            self,
            *,
            episode: Any,
            env_runner: Any = None,
            env: Any = None,
            **kwargs: Any,
        ) -> None:
            """Initialize buffers in the episode's user data on start.

            Args:
                episode: The Episode object.
                env_runner: The EnvRunner instance.
                env: The MultiAgentEnv instance.
                kwargs: Keyword arguments for future compatibility.
            """
            # local
            from p2p_energy_trading.constants import ALL_AGENT_IDS

            episode.user_data["p2p_volume"] = 0.0
            episode.user_data["grid_import"] = 0.0
            episode.user_data["grid_export"] = 0.0
            episode.user_data["voltage_violations"] = 0
            episode.user_data["thermal_violations"] = 0
            episode.user_data["agent_net_cost"] = {aid: 0.0 for aid in ALL_AGENT_IDS}

        def on_episode_step(
            self,
            *,
            episode: Any,
            env_runner: Any = None,
            env: Any = None,
            **kwargs: Any,
        ) -> None:
            """Accumulate step-wise settlements, volumes, and physical violations.

            Args:
                episode: The Episode object.
                env_runner: The EnvRunner instance.
                env: The MultiAgentEnv instance.
                kwargs: Keyword arguments for future compatibility.
            """
            # local
            from p2p_energy_trading.constants import ALL_AGENT_IDS

            # Extract step info in a version-resilient manner
            last_infos = {}
            if hasattr(episode, "get_infos"):
                last_infos = episode.get_infos() or {}
            elif hasattr(episode, "last_info_for"):
                last_infos = {
                    aid: episode.last_info_for(aid)
                    for aid in ALL_AGENT_IDS
                    if episode.last_info_for(aid) is not None
                }

            # Update metrics using read-only logic
            for aid, info in last_infos.items():
                if not isinstance(info, dict):
                    continue
                episode.user_data["p2p_volume"] += info.get("p2p_sold_kw", 0.0)
                episode.user_data["grid_import"] += info.get("grid_bought_kw", 0.0)
                episode.user_data["grid_export"] += info.get("grid_sold_kw", 0.0)

                if aid in episode.user_data["agent_net_cost"]:
                    episode.user_data["agent_net_cost"][aid] += info.get(
                        "net_cost", 0.0
                    )

            # System-level grid violations checking
            if any(
                info.get("voltage_violation", False)
                for info in last_infos.values()
                if isinstance(info, dict)
            ):
                episode.user_data["voltage_violations"] += 1

            if any(
                info.get("thermal_violation", False)
                for info in last_infos.values()
                if isinstance(info, dict)
            ):
                episode.user_data["thermal_violations"] += 1

            # Extract curriculum phase dynamically from the base environment
            # if available
            if env is not None:
                base_env = None
                if hasattr(env, "get_sub_environments"):
                    sub_envs = env.get_sub_environments()
                    if sub_envs:
                        base_env = sub_envs[0]
                elif hasattr(env, "envs"):
                    base_env = env.envs[0]
                else:
                    base_env = env

                if base_env is not None and hasattr(base_env, "curriculum_phase"):
                    episode.user_data["curriculum_phase"] = base_env.curriculum_phase
                    episode.user_data["total_env_steps"] = base_env.total_env_steps

        def on_episode_end(
            self,
            *,
            episode: Any,
            env_runner: Any = None,
            env: Any = None,
            **kwargs: Any,
        ) -> None:
            """Aggregate and publish tracked microgrid metrics to custom_metrics.

            Args:
                episode: The Episode object.
                env_runner: The EnvRunner instance.
                env: The MultiAgentEnv instance.
                kwargs: Keyword arguments for future compatibility.
            """
            # local
            from p2p_energy_trading.constants import CONSUMER_AGENT_IDS, SOLAR_AGENT_IDS

            # Compute totals and export to RLlib custom metrics
            episode.custom_metrics["p2p_volume_total"] = episode.user_data.get(
                "p2p_volume", 0.0
            )
            episode.custom_metrics["grid_import_total"] = episode.user_data.get(
                "grid_import", 0.0
            )
            episode.custom_metrics["grid_export_total"] = episode.user_data.get(
                "grid_export", 0.0
            )
            episode.custom_metrics["voltage_violations_total"] = episode.user_data.get(
                "voltage_violations", 0
            )
            episode.custom_metrics["thermal_violations_total"] = episode.user_data.get(
                "thermal_violations", 0
            )

            if "curriculum_phase" in episode.user_data:
                episode.custom_metrics["curriculum_phase"] = episode.user_data[
                    "curriculum_phase"
                ]
            if "total_env_steps" in episode.user_data:
                episode.custom_metrics["total_env_steps"] = episode.user_data[
                    "total_env_steps"
                ]

            # Compute policy-group cost statistics
            agent_net_cost = episode.user_data.get("agent_net_cost", {})
            college_cost = agent_net_cost.get("college", 0.0)
            solar_cost = sum(agent_net_cost.get(aid, 0.0) for aid in SOLAR_AGENT_IDS)
            consumer_cost = sum(
                agent_net_cost.get(aid, 0.0) for aid in CONSUMER_AGENT_IDS
            )
            total_campus_cost = college_cost + solar_cost + consumer_cost

            episode.custom_metrics["net_cost_college"] = college_cost
            episode.custom_metrics["net_cost_solar"] = solar_cost
            episode.custom_metrics["net_cost_consumer"] = consumer_cost
            episode.custom_metrics["net_cost_total_campus"] = total_campus_cost

else:

    class P2PCallbacks:  # type: ignore
        """Placeholder class that raises ImportError if instantiated
        when Ray is absent."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "Ray/RLlib is not installed in the current Python environment. "
                "P2PCallbacks requires Ray/RLlib to be instantiated."
            )
