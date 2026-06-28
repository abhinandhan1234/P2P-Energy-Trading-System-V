"""Training orchestrator and entry point for the P2P Energy Trading System.

Parses command-line arguments, validates configurations, configures the RLlib
MAPPO algorithm, runs the training loop with curriculum stages and evaluation,
handles process interruption signals to save emergency checkpoints, and outputs
structured logs.

Design reference: docs/module_8_training_pipeline.md §3, §4, §5
"""

from __future__ import annotations

# standard library
import argparse
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Force pager to cat for command line executions
os.environ["PAGER"] = "cat"

try:
    # third party
    import ray
    from ray.rllib.algorithms.algorithm import Algorithm

    RAY_AVAILABLE = True
except ImportError:
    Algorithm = None  # type: ignore
    RAY_AVAILABLE = False


def setup_logger(log_level: str, log_file: str | None = None) -> None:
    """Configure the logging system with console and optional file handlers.

    Args:
        log_level: Severity string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path to write log output.
    """
    # local
    from p2p_energy_trading.constants import LOG_DATE_FORMAT, LOG_FORMAT

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


def update_env_config(env: Any, new_config: dict[str, Any]) -> None:
    """Updates environment instance variables dynamically for curriculum transitions.

    Avoids direct modification of Module 6 class code by modifying attributes in place.

    Args:
        env: The instantiated P2PEnergyTradingEnv object.
        new_config: Dictionary containing stage configuration overrides.
    """
    env.config.update(new_config)
    if "episode_length" in new_config:
        el = int(new_config["episode_length"])
        env.episode_length = el
        if hasattr(env, "episode_manager") and env.episode_manager is not None:
            env.episode_manager._episode_length = el
    if "pandapower_bypass" in new_config:
        env.pandapower_bypass = bool(new_config["pandapower_bypass"])
    if "curriculum_transition_step" in new_config:
        env.curriculum_transition_step = int(new_config["curriculum_transition_step"])

    logger.info(
        "Updated worker environment config: episode_length=%d, bypass=%s, transition_step=%d",
        env.episode_length,
        env.pandapower_bypass,
        env.curriculum_transition_step,
    )


def check_configuration_mismatch(algo: Any, config: dict[str, Any]) -> None:
    """Ensure that the resumed config does not have fatal mismatches with checkpoint.

    Fatal mismatches (aborts execution):
    - Actor/Critic observation/action space shapes.
    - Neural network hidden layers.

    Safe mismatches (applies and continues):
    - Learning rate, entropy, batch size, worker counts.

    Args:
        algo: Resumed Algorithm instance.
        config: Target YAML configuration dictionary.

    Raises:
        ValueError: If a fatal mismatch is identified.
    """
    # local
    from p2p_energy_trading.constants import POLICY_COLLEGE

    policy = algo.get_policy(POLICY_COLLEGE)
    if not policy:
        return

    # 1. Check space dimensions (Fatal)
    obs_space = policy.observation_space
    if hasattr(obs_space, "original_space"):
        obs_space = obs_space.original_space

    if "obs" not in obs_space.spaces or obs_space.spaces["obs"].shape != (23,):
        raise ValueError(
            f"FATAL: Observation space dimension mismatch! Checkpoint expects 23-dim local, "
            f"got {obs_space.spaces.get('obs')}"
        )
    if "state" not in obs_space.spaces or obs_space.spaces["state"].shape != (243,):
        raise ValueError(
            f"FATAL: Critic state dimension mismatch! Checkpoint expects 243-dim state, "
            f"got {obs_space.spaces.get('state')}"
        )
    if policy.action_space.shape != (3,):
        raise ValueError(
            f"FATAL: Action space dimension mismatch! Checkpoint expects 3-dim actions, "
            f"got {policy.action_space}"
        )

    # 2. Check safe parameter updates and log changes
    lr = config["ppo"].get("lr", 3e-4)
    if algo.config.lr != lr:
        logger.info(
            "Safe configuration update: lr changed from %s to %s. Applying to resumed runner.",
            algo.config.lr,
            lr,
        )
        algo.config.lr = lr

    entropy_coeff = config["ppo"].get("entropy_coeff", 0.01)
    if algo.config.entropy_coeff != entropy_coeff:
        logger.info(
            "Safe configuration update: entropy_coeff changed from %s to %s. Applying to resumed runner.",
            algo.config.entropy_coeff,
            entropy_coeff,
        )
        algo.config.entropy_coeff = entropy_coeff


def print_iteration_summary(results: dict[str, Any], stage: str, phase: int) -> None:
    """Print iteration metrics conforming to the required format in Module 8 §8.

    Args:
        results: Dictionary containing training iteration results.
        stage: Current curriculum stage name.
        phase: Current reward curriculum phase.
    """
    iteration = results.get("training_iteration", 0)
    steps = results.get("agent_steps_total") or results.get("info", {}).get(
        "agent_steps_total", 0
    )

    # Policy rewards
    rew_college = results.get("policy_reward_mean", {}).get("policy_college", 0.0)
    rew_solar = results.get("policy_reward_mean", {}).get("policy_solar", 0.0)
    rew_consumer = results.get("policy_reward_mean", {}).get("policy_consumer", 0.0)
    rew_mean = results.get("episode_reward_mean", 0.0)

    custom = results.get("custom_metrics", {})
    p2p_vol = custom.get("p2p_volume_total_mean", 0.0)
    util = custom.get("p2p_utilisation_ratio_mean", 0.0)
    campus_cost = custom.get("net_cost_total_campus_mean", 0.0)
    violations = custom.get("voltage_violations_total_mean", 0.0) + custom.get(
        "thermal_violations_total_mean", 0.0
    )
    min_v = custom.get("grid/min_bus_voltage_mean", 1.0)
    max_loading = custom.get("grid/max_line_loading_mean", 0.0)

    # Battery
    soc_mean = custom.get("battery/mean_soc_mean", 0.5)
    cycles = custom.get("battery/cycling_count_mean", 0)

    # Training losses / stats
    # Try different paths for loss / stats depending on RLlib stack variations
    policy_college_stats = (
        results.get("learner", {}).get("policy_college", {}).get("learner_stats", {})
    )
    loss = policy_college_stats.get(
        "total_loss",
        results.get("info", {})
        .get("learner", {})
        .get("policy_college", {})
        .get("total_loss", 0.0),
    )
    entropy = policy_college_stats.get("entropy", 0.0)
    kl = policy_college_stats.get("kl", 0.0)

    print(
        f"\n[Iter {iteration:04d} | Stage: {stage} | Phase: {phase} | Steps: {steps / 1e6:.2f}M]\n"
        f"  Reward: college={rew_college:.2f}  solar={rew_solar:.2f}  consumer={rew_consumer:.2f}  mean={rew_mean:.2f}\n"
        f"  Market: P2P_vol={p2p_vol:.1f}kWh  util={util:.2f}  campus_cost=₹{campus_cost:,.0f}\n"
        f"  Grid:   violations={violations:.1f}  min_V={min_v:.3f}  max_loading={max_loading:.2f}\n"
        f"  Battery: SoC_mean={soc_mean:.2f}  cycles={cycles:.1f}\n"
        f"  Training: loss={loss:.4f}  entropy={entropy:.3f}  KL={kl:.4f}"
    )
    sys.stdout.flush()


def parse_args() -> argparse.Namespace:
    """Parse command line parameters."""
    parser = argparse.ArgumentParser(
        description="Orchestrate Multi-Agent PPO training pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to YAML training configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint directory path to resume training.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=["debug", "training", "constraint_aware"],
        help="Force starting curriculum stage.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override maximum training iterations count.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation rolls only (requires --resume checkpoint).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override global random seed value.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override hardware rollout workers runner count.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU and execute training on CPU-only modes.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override active console log levels.",
    )
    return parser.parse_args()


def main() -> None:
    """Orchestrate training lifecycle execution."""
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray/RLlib is required but is not installed in the current environment."
        )

    args = parse_args()

    # 1. Load and merge configuration overrides
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["environment.seed"] = args.seed
        overrides["ppo.seed"] = args.seed
    if args.iterations is not None:
        overrides["training.max_training_iterations"] = args.iterations
    if args.num_workers is not None:
        overrides["hardware.num_env_runners"] = args.num_workers
    if args.no_gpu:
        overrides["hardware.num_gpus_per_learner_worker"] = 0
    if args.log_level is not None:
        overrides["logging.log_level"] = args.log_level

    # local
    from p2p_energy_trading.training.checkpoint_manager import CheckpointManager
    from p2p_energy_trading.training.config_loader import load_training_config
    from p2p_energy_trading.training.curriculum import CurriculumManager

    config = load_training_config(args.config, overrides)

    # 2. Configure logging systems
    log_level = config["logging"].get("log_level", "INFO")
    log_file = config["logging"].get("log_file")
    setup_logger(log_level, log_file)

    logger.info("Initializing Ray training orchestrator...")

    # 3. Ray Initialization
    ray_cpus = config["hardware"].get("ray_num_cpus")
    ray_gpus = config["hardware"].get("ray_num_gpus")
    ray.init(
        num_cpus=ray_cpus,
        num_gpus=ray_gpus,
        ignore_reinit_error=True,
        log_to_driver=False,
    )

    # local
    from p2p_energy_trading.rl.policy_config import build_ppo_config

    algo = None
    try:
        # Determine starting stage
        current_stage = args.stage or "debug"
        curriculum_manager = CurriculumManager(config)
        checkpoint_manager = CheckpointManager(config)

        # Get initial overrides for start stage
        stage_overrides = curriculum_manager.get_stage_overrides(current_stage)

        # Merge environment overrides into environment configuration
        env_config = {**config["environment"], **stage_overrides}
        ppo_config = config["ppo"]
        hardware_config = config["hardware"]

        # Build training algorithm config specification
        rllib_config = build_ppo_config(env_config, ppo_config, hardware_config)

        # 4. Construct or Restore Algorithm
        if args.resume:
            logger.info("Resuming algorithm from checkpoint: %s", args.resume)
            # Resilient checkpoint load fallback
            load_path = args.resume
            if not os.path.exists(load_path):
                # check fallback periodic directory
                fallback_path = (
                    Path(config["checkpoint"].get("checkpoint_dir", "checkpoints"))
                    / load_path
                )
                if fallback_path.exists():
                    load_path = str(fallback_path)

            algo = Algorithm.from_checkpoint(load_path)
            check_configuration_mismatch(algo, config)
        else:
            logger.info("Starting fresh training run in stage: %s", current_stage)
            algo = rllib_config.build()

        # 5. Handle SIGINT/SIGTERM gracefully
        def signal_handler(sig: int, frame: Any) -> None:
            logger.warning("Signal received (%d). Triggering emergency save...", sig)
            if algo:
                try:
                    checkpoint_manager.save_emergency_checkpoint(algo, algo.iteration)
                except Exception as e:
                    logger.error("Failed to save emergency checkpoint: %s", e)
            ray.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Evaluation only execution flag
        if args.eval_only:
            logger.info("Running evaluation-only suite...")
            eval_results = algo.evaluate()
            logger.info("Evaluation results: %s", eval_results)
            return

        # 6. Training Loop Orchestration
        max_iters = int(config["training"].get("max_training_iterations", 5000))
        max_steps = int(config["training"].get("max_agent_steps", 15500000))
        early_stopping = bool(config["training"].get("early_stopping", True))
        patience = int(config["training"].get("early_stopping_patience", 500))

        no_improvement_iters = 0
        best_eval_metric = -float("inf")

        for iteration in range(1, max_iters + 1):
            results = algo.train()
            total_steps = results.get("agent_steps_total") or results.get(
                "info", {}
            ).get("agent_steps_total", 0)
            total_episodes = results.get("episodes_total", 0)

            # Determine reward phase: Phase 1 or Phase 2 based on env step
            phase = env_config.get("reward_phase", 1)
            # If environmental steps exceed transition step in stage config, phase is 2
            if total_steps >= env_config.get("curriculum_transition_step", 2000000):
                phase = 2

            # Printiteration logs
            print_iteration_summary(results, current_stage, phase)

            # 7. Check Stage Progression
            should_transition, next_stage = curriculum_manager.check_progression(
                results, current_stage, total_episodes, total_steps
            )
            if should_transition:
                logger.info(
                    "Transitioning curriculum stage: %s -> %s at step %d",
                    current_stage,
                    next_stage,
                    total_steps,
                )
                if config["checkpoint"].get("save_on_stage_transition", True):
                    checkpoint_manager.save_stage_checkpoint(
                        algo, iteration, total_steps, current_stage
                    )

                current_stage = next_stage
                stage_overrides = curriculum_manager.get_stage_overrides(current_stage)

                # Mutate environments configs on running runners
                def runner_update(runner: Any) -> None:
                    runner.foreach_env(
                        lambda env: update_env_config(env, stage_overrides)
                    )

                if hasattr(algo, "env_runner_group"):
                    algo.env_runner_group.foreach_env_runner(runner_update)
                elif hasattr(algo, "workers"):
                    algo.workers.foreach_worker(runner_update)

            # 8. Periodic Evaluation
            eval_freq = config["evaluation"].get("eval_frequency", 100)
            if iteration % eval_freq == 0:
                logger.info("Starting periodic evaluation...")
                eval_results = algo.evaluate()

                eval_reward = eval_results.get("evaluation", {}).get(
                    "episode_reward_mean", -float("inf")
                )

                # Compile metrics for best checkpoint checks
                eval_metrics = {
                    "p2p_utilisation_ratio": eval_results.get("evaluation", {})
                    .get("custom_metrics", {})
                    .get("p2p_utilisation_ratio_mean", 0.0),
                    "grid_violation_rate": eval_results.get("evaluation", {})
                    .get("custom_metrics", {})
                    .get("voltage_violations_total_mean", 0.0),
                    "total_campus_cost": eval_results.get("evaluation", {})
                    .get("custom_metrics", {})
                    .get("net_cost_total_campus_mean", 0.0),
                }

                # Save best model if metrics improve
                checkpoint_manager.save_best_checkpoint(
                    algo, iteration, total_steps, eval_reward, eval_metrics
                )

                # Early stopping tracking
                if early_stopping:
                    if eval_reward > best_eval_metric:
                        best_eval_metric = eval_reward
                        no_improvement_iters = 0
                    else:
                        no_improvement_iters += eval_freq
                        if no_improvement_iters >= patience:
                            logger.warning(
                                "Early stopping triggered! No evaluation improvement for %d iterations.",
                                no_improvement_iters,
                            )
                            break

            # 9. Periodic Checkpoint
            chk_freq = config["checkpoint"].get("checkpoint_frequency", 50)
            if iteration % chk_freq == 0:
                checkpoint_manager.save_periodic_checkpoint(
                    algo, iteration, total_steps, current_stage
                )
                checkpoint_manager.prune_checkpoints()

            # 10. Check Termination Limits
            if total_steps >= max_steps:
                logger.info(
                    "Maximum agent steps reached (%d). Exiting training loop.",
                    max_steps,
                )
                break

            if current_stage == "constraint_aware" and curriculum_manager.is_converged(
                results
            ):
                logger.info("Training converged successfully. Exiting training loop.")
                break

        # Save final checkpoint on termination
        logger.info("Saving final training checkpoint...")
        checkpoint_manager.save_periodic_checkpoint(
            algo, iteration, total_steps, current_stage
        )

    except Exception as e:
        logger.exception("Fatal exception occurred in training execution: %s", e)
        if algo:
            try:
                checkpoint_manager.save_emergency_checkpoint(algo, algo.iteration)
            except Exception as save_err:
                logger.error("Failed to save emergency checkpoint: %s", save_err)
        raise e

    finally:
        if algo:
            algo.stop()
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Ray orchestrator shutdown complete.")


if __name__ == "__main__":
    main()
