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

# Reconfigure stdout/stderr to UTF-8 on Windows so that Unicode characters
# (e.g. the Indian Rupee sign ₹, U+20B9) printed via colorama do not raise a
# UnicodeEncodeError through the default 'charmap' codec.  This is a pure
# I/O-encoding change and has no effect on training semantics.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

try:
    # third party
    import numpy as np
    import ray
    import torch
    from ray.rllib.algorithms.algorithm import Algorithm
    from tensorboardX import SummaryWriter

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
        "Updated worker environment config:"
        " episode_length=%d, bypass=%s, transition_step=%d",
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
            f"FATAL: Observation space dimension mismatch!"
            f" Checkpoint expects 23-dim local, "
            f"got {obs_space.spaces.get('obs')}"
        )
    if "state" not in obs_space.spaces or obs_space.spaces["state"].shape != (243,):
        raise ValueError(
            f"FATAL: Critic state dimension mismatch!"
            f" Checkpoint expects 243-dim state, "
            f"got {obs_space.spaces.get('state')}"
        )
    if policy.action_space.shape != (3,):
        raise ValueError(
            f"FATAL: Action space dimension mismatch!"
            f" Checkpoint expects 3-dim actions, "
            f"got {policy.action_space}"
        )

    # 2. Check safe parameter updates and log changes
    lr = config["ppo"].get("lr", 3e-4)
    if algo.config.lr != lr:
        logger.info(
            "Safe configuration update: lr changed from %s to %s."
            " Applying to resumed runner.",
            algo.config.lr,
            lr,
        )
        algo.config.lr = lr

    entropy_coeff = config["ppo"].get("entropy_coeff", 0.01)
    if algo.config.entropy_coeff != entropy_coeff:
        logger.info(
            "Safe configuration update: entropy_coeff changed from %s to %s."
            " Applying to resumed runner.",
            algo.config.entropy_coeff,
            entropy_coeff,
        )
        algo.config.entropy_coeff = entropy_coeff


def print_iteration_summary(results: dict[str, Any], stage: str, phase: int) -> None:
    """Print iteration metrics conforming to the required format in Module 8 §8.

    RLlib 2.x New API Stack key layout
    -----------------------------------
    Episode rewards (from EnvRunners):
      results["env_runners"]["episode_return_mean"]               – global mean
      results["env_runners"]["module_episode_returns_mean"][mid]  – per-policy module

    Learner losses (per-module, flat dict under module ID):
      results["learners"][mid]["policy_loss"]   (POLICY_LOSS_KEY)
      results["learners"][mid]["vf_loss"]        (VF_LOSS_KEY)
      results["learners"][mid]["entropy"]        (ENTROPY_KEY)
      results["learners"][mid]["mean_kl_loss"]   (LEARNER_RESULTS_KL_KEY)

    Custom callback metrics (from on_episode_end):
      results["env_runners"]["custom_metrics"][key]

    Cumulative agent steps:
      results["num_agent_steps_sampled_lifetime"]

    Args:
        results: Dictionary containing training iteration results.
        stage: Current curriculum stage name.
        phase: Current reward curriculum phase.
    """
    iteration = results.get("training_iteration", 0)

    # ------------------------------------------------------------------ steps
    # New API Stack: "num_agent_steps_sampled_lifetime"
    # Fallback chain covers older builds or partial-result dicts.
    steps = (
        results.get("num_agent_steps_sampled_lifetime")
        or results.get("agent_steps_total")
        or results.get("info", {}).get("agent_steps_total", 0)
        or 0
    )

    # ----------------------------------------------------------------- rewards
    # New API Stack: rewards live under results["env_runners"].
    env_runner_results: dict[str, Any] = results.get("env_runners", {})

    # Per-policy-module mean episode return.
    module_returns: dict[str, Any] = env_runner_results.get(
        "module_episode_returns_mean", {}
    )
    rew_college = float(module_returns.get("policy_college", 0.0))
    rew_solar = float(module_returns.get("policy_solar", 0.0))
    rew_consumer = float(module_returns.get("policy_consumer", 0.0))

    # Global mean episode return (sum over all agents per episode).
    rew_mean = float(env_runner_results.get("episode_return_mean", 0.0))

    # ----------------------------------------------------------- custom metrics
    # Callbacks write to results["env_runners"]["custom_metrics"] in the New
    # API Stack; fall back to the top-level key for the Old API Stack.
    custom: dict[str, Any] = env_runner_results.get(
        "custom_metrics", results.get("custom_metrics", {})
    )

    p2p_vol = float(custom.get("p2p_volume_total_mean", 0.0))
    util = float(custom.get("p2p_utilisation_ratio_mean", 0.0))
    campus_cost = float(custom.get("net_cost_total_campus_mean", 0.0))
    violations = float(
        custom.get("voltage_violations_total_mean", 0.0)
        + custom.get("thermal_violations_total_mean", 0.0)
    )
    min_v = float(custom.get("grid/min_bus_voltage_mean", 1.0))
    max_loading = float(custom.get("grid/max_line_loading_mean", 0.0))

    # Battery
    soc_mean = float(custom.get("battery/mean_soc_mean", 0.5))
    cycles = float(custom.get("battery/cycling_count_mean", 0))

    # ---------------------------------------------------------- training losses
    # New API Stack: results["learners"][module_id] is a flat dict with keys
    # directly from the Learner (e.g. "policy_loss", "vf_loss", "entropy",
    # "mean_kl_loss").  The old stack used
    # results["learner"][module_id]["learner_stats"][...] which no longer exists.
    learners: dict[str, Any] = results.get("learners", {})
    college_learner: dict[str, Any] = learners.get("policy_college", {})

    policy_loss = float(college_learner.get("policy_loss", 0.0))
    vf_loss = float(college_learner.get("vf_loss", 0.0))
    entropy = float(college_learner.get("entropy", 0.0))
    # RLlib 2.x PPO reports KL as "mean_kl_loss"; older builds used "kl".
    kl = float(college_learner.get("mean_kl_loss", college_learner.get("kl", 0.0)))

    print(
        f"\n[Iter {iteration:04d} | Stage: {stage} | Phase: {phase}"
        f" | Steps: {steps / 1e6:.2f}M]\n"
        f"  Reward: college={rew_college:.2f}  solar={rew_solar:.2f}"
        f"  consumer={rew_consumer:.2f}  mean={rew_mean:.2f}\n"
        f"  Market: P2P_vol={p2p_vol:.1f}kWh  util={util:.2f}"
        f"  campus_cost=Rs.{campus_cost:,.0f}\n"
        f"  Grid:   violations={violations:.1f}  min_V={min_v:.3f}"
        f"  max_loading={max_loading:.2f}\n"
        f"  Battery: SoC_mean={soc_mean:.2f}  cycles={cycles:.1f}\n"
        f"  Training: policy_loss={policy_loss:.4f}  vf_loss={vf_loss:.4f}"
        f"  entropy={entropy:.3f}  KL={kl:.4f}"
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
    if ray_gpus is None:
        ray_gpus = 1 if torch.cuda.is_available() else 0
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

        # Initialize TensorBoard SummaryWriter
        tb_dir = config["logging"].get("tensorboard_dir", "logs")
        tb_writer = SummaryWriter(logdir=tb_dir)

        for iteration in range(1, max_iters + 1):
            results = algo.train()
            # New API Stack: cumulative agent steps are under
            # "num_agent_steps_sampled_lifetime"; fall back to the Old API
            # Stack key so that checkpoints/curriculum still work.
            total_steps = (
                results.get("num_agent_steps_sampled_lifetime")
                or results.get("agent_steps_total")
                or results.get("info", {}).get("agent_steps_total", 0)
                or 0
            )
            # Total episodes: New API Stack exposes this under env_runners.
            total_episodes = (
                results.get("env_runners", {}).get("num_episodes_lifetime", 0)
                or results.get("episodes_total", 0)
                or 0
            )

            # Determine reward phase: Phase 1 or Phase 2 based on env step
            phase = env_config.get("reward_phase", 1)
            # If environmental steps exceed transition step in stage config, phase is 2
            if total_steps >= env_config.get("curriculum_transition_step", 2000000):
                phase = 2

            # Print iteration logs; guard against console encoding errors on
            # Windows (e.g. charmap codec cannot encode U+20B9) so that a
            # failed console write never aborts the TensorBoard write path.
            try:
                print_iteration_summary(results, current_stage, phase)
            except UnicodeEncodeError as _ue:
                logger.warning(
                    "Console encoding error in print_iteration_summary"
                    " (iteration %d): %s. Continuing training.",
                    iteration,
                    _ue,
                )

            # Write metrics to TensorBoard
            if tb_writer is not None:
                # ------------------------------------------------------ rewards
                # New API Stack: rewards live under results["env_runners"].
                _er = results.get("env_runners", {})
                tb_writer.add_scalar(
                    "Reward/episode_return_mean",
                    float(_er.get("episode_return_mean", 0.0)),
                    iteration,
                )
                _mod_rets: dict[str, Any] = _er.get("module_episode_returns_mean", {})
                for policy_name in [
                    "policy_college",
                    "policy_solar",
                    "policy_consumer",
                ]:
                    p_rew = float(_mod_rets.get(policy_name, 0.0))
                    tb_writer.add_scalar(
                        f"Reward/module_episode_returns_mean/{policy_name}",
                        p_rew,
                        iteration,
                    )

                # ----------------------------------------- learner losses / stats
                # New API Stack: results["learners"][module_id] is a flat dict
                # with keys "policy_loss", "vf_loss", "entropy", "mean_kl_loss".
                _learners: dict[str, Any] = results.get("learners", {})
                for policy_name in [
                    "policy_college",
                    "policy_solar",
                    "policy_consumer",
                ]:
                    _stats: dict[str, Any] = _learners.get(policy_name, {})
                    if _stats:
                        tb_writer.add_scalar(
                            f"Loss/{policy_name}_policy_loss",
                            float(_stats.get("policy_loss", 0.0)),
                            iteration,
                        )
                        tb_writer.add_scalar(
                            f"Loss/{policy_name}_vf_loss",
                            float(_stats.get("vf_loss", 0.0)),
                            iteration,
                        )
                        tb_writer.add_scalar(
                            f"Entropy/{policy_name}_entropy",
                            float(_stats.get("entropy", 0.0)),
                            iteration,
                        )
                        tb_writer.add_scalar(
                            f"KL/{policy_name}_kl",
                            float(_stats.get("mean_kl_loss", _stats.get("kl", 0.0))),
                            iteration,
                        )

                # -------------------------------- custom metrics (Market/Grid/Battery)
                # New API Stack: callbacks write to
                # results["env_runners"]["custom_metrics"].
                _custom: dict[str, Any] = _er.get(
                    "custom_metrics", results.get("custom_metrics", {})
                )
                for key, val in _custom.items():
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        tb_writer.add_scalar(f"Custom/{key}", float(val), iteration)
                    elif isinstance(val, dict):
                        for subkey, subval in val.items():
                            if isinstance(
                                subval, (int, float, np.integer, np.floating)
                            ):
                                tb_writer.add_scalar(
                                    f"Custom/{key}/{subkey}",
                                    float(subval),
                                    iteration,
                                )

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
                def runner_update(
                    runner: Any,
                    stage_overrides: dict[str, Any] = stage_overrides,
                ) -> None:
                    runner.foreach_env(
                        lambda env, _so=stage_overrides: update_env_config(env, _so)
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
                                "Early stopping triggered! No evaluation"
                                " improvement for %d iterations.",
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
        if "tb_writer" in locals() and tb_writer is not None:
            tb_writer.close()
        if algo:
            algo.stop()
        if ray.is_initialized():
            ray.shutdown()
        logger.info("Ray orchestrator shutdown complete.")


if __name__ == "__main__":
    main()
