"""Orchestration CLI entry point for Module 9 (Evaluation Framework).

Loads defaults and evaluation overrides, configures the environment, and runs
multi-seed evaluation loops across Trained Policy, Grid-Only, Random, Heuristic,
and Ablation experiments.

Design reference: docs/module_9_evaluation_framework.md §1
"""

from __future__ import annotations

# standard library
import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

# third party
import pandas as pd
import yaml

# local
from p2p_energy_trading.constants import ALL_AGENT_IDS, COLLEGE_AGENT_ID
from p2p_energy_trading.environment.env import P2PEnergyTradingEnv
from p2p_energy_trading.evaluation.baselines import (
    GridOnlyBaseline,
    HeuristicBaseline,
    RandomBaseline,
)
from p2p_energy_trading.evaluation.metrics import MetricCollector
from p2p_energy_trading.evaluation.plotting import generate_plots, prepare_plot_data
from p2p_energy_trading.evaluation.report import generate_reports
from p2p_energy_trading.evaluation.statistical import run_significance_tests
from p2p_energy_trading.training.config_loader import load_training_config

# Handle Ray/RLlib presence
try:
    # third party
    from ray.rllib.algorithms.algorithm import Algorithm

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Checkpoint URI helper
# ---------------------------------------------------------------------------


def resolve_checkpoint_uri(path: str) -> str:
    """Return a proper ``file://`` URI for *path*, safe for RLlib / PyArrow.

    Accepts any of the following input formats and normalises them to a
    ``file:///`` URI that RLlib's ``Algorithm.from_checkpoint()`` accepts on
    every platform (Windows, Linux, macOS):

    * Relative filesystem path  – ``checkpoints/checkpoint_000003``
    * Absolute Windows path     – ``C:\\Users\\...\\checkpoint_000003``
    * Absolute POSIX path       – ``/home/user/checkpoint_000003``
    * Already-valid file URI    – ``file:///C:/Users/.../checkpoint_000003``

    Raises
    ------
    FileNotFoundError
        When the resolved path does not exist on disk.
    """
    # If the caller already supplied a valid file URI, validate and return it.
    parsed = urlparse(path)
    if parsed.scheme == "file":
        # Re-derive the filesystem path so we can verify existence.
        fs_path = Path(url2pathname(parsed.path))
        if not fs_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {fs_path}")
        return path  # keep the URI the caller passed in unchanged

    # Otherwise treat *path* as a filesystem path (relative or absolute).
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")
    return resolved.as_uri()


def run_evaluation(
    eval_config_path: str | Path,
    checkpoint_path_override: str | None = None,
    output_dir_override: str | None = None,
    experiments_override: list[str] | None = None,
) -> None:
    """Run the evaluation framework pipeline."""
    # 1. Load evaluation config
    with open(eval_config_path) as f:
        eval_cfg = yaml.safe_load(f)

    # 2. Merge defaults
    training_config_path = eval_cfg.get(
        "training_config", "config/training_config.yaml"
    )
    checkpoint_path = checkpoint_path_override or eval_cfg.get(
        "checkpoint_path", "checkpoints/best_model"
    )

    eval_settings = eval_cfg.get("evaluation", {})
    results_dir = Path(
        output_dir_override or eval_settings.get("results_dir", "results")
    )
    seeds = eval_settings.get("seeds", [42, 123, 456, 789, 1024])
    ep_starts = eval_settings.get("eval_episode_starts", [0])
    experiments = experiments_override or eval_settings.get(
        "experiments", ["trained", "grid_only", "random", "heuristic", "no_battery"]
    )

    # Load portfolio metadata from generated file (avoids private env attribute access)
    metadata_path = Path("data/processed/metadata.json")
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            env_metadata = json.load(f)
    else:
        env_metadata = {}

    # Extract start date and peak demand from metadata dynamically
    start_date_str = "2023-01-01T00:00:00"
    college_peak_demand = 361.0
    if env_metadata:
        b_meta = env_metadata.get("buildings", {}).get(COLLEGE_AGENT_ID, {})
        start_date_str = (
            b_meta.get("profile_stats", {})
            .get("date_range", {})
            .get("start", "2023-01-01T00:00:00")
        )
        college_peak_demand = (
            b_meta.get("profile_stats", {}).get("demand_kw", {}).get("peak", 361.0)
        )
    start_date = pd.Timestamp(start_date_str)

    # 3. Load Environment Configuration via Module 8 Loader
    full_train_config = load_training_config(training_config_path)
    env_config = full_train_config.get("environment", {}).copy()

    # Force evaluation mode parameters
    env_config["eval_mode"] = True
    env_config["pandapower_bypass"] = False
    env_config["curriculum_phase"] = 2
    env_config["episode_length"] = 168

    # 4. Instantiate evaluation environment
    logger.info("Initializing P2PEnergyTradingEnv in evaluation mode...")
    env = P2PEnergyTradingEnv(env_config)

    # 5. Load Trained RLlib Policy if requested
    algo = None
    modules = {}
    if "trained" in experiments or "no_battery" in experiments:
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray/RLlib is not installed, but evaluation requested"
                " Trained Policy. "
                "Baseline evaluation must continue functioning even when"
                " Ray is unavailable. "
                "Only checkpoint loading and trained-policy evaluation"
                " require Ray."
            )
        # Resolve checkpoint path → file:// URI before handing to RLlib/PyArrow
        checkpoint_uri = resolve_checkpoint_uri(checkpoint_path)
        logger.info(
            "Restoring trained MAPPO policies from checkpoint: %s", checkpoint_uri
        )
        # Register environment with RLlib before restoring the checkpoint
        # local
        from p2p_energy_trading.rl.env_registration import register_p2p_environment

        register_p2p_environment()

        # Restore algorithm using New API Stack checkpoint loading
        algo = Algorithm.from_checkpoint(checkpoint_uri)

        # Retrieve sub-modules for evaluation inference
        modules = {
            "policy_college": algo.get_module("policy_college"),
            "policy_solar": algo.get_module("policy_solar"),
            "policy_consumer": algo.get_module("policy_consumer"),
        }

    # 6. Initialize Metric Collector
    collector = MetricCollector(results_dir=results_dir)

    # 7. Run evaluation experiments
    for exp in experiments:
        logger.info("Starting experiment: %s", exp)

        # Instantiate baseline policy controllers
        baseline_policy = None
        if exp == "grid_only":
            baseline_policy = GridOnlyBaseline()
        elif exp == "random":
            # Seed-specific instantiation handled below
            pass
        elif exp == "heuristic":
            baseline_policy = HeuristicBaseline(peak_demand=college_peak_demand)

        for seed in seeds:
            logger.info("Running Seed %d for %s", seed, exp)

            # Re-seed random baseline policies
            if exp == "random":
                baseline_policy = RandomBaseline(
                    seed=seed, peak_demand=college_peak_demand
                )

            # Loop through the representative episode starting points
            for episode_idx, start_hour in enumerate(ep_starts):
                obs, info = env.reset(
                    seed=seed, options={"eval_start_hour": start_hour}
                )

                # Check for reset observation structure compatibility
                for aid in ALL_AGENT_IDS:
                    if aid not in obs:
                        raise ValueError(f"Reset observation missing agent ID: {aid}")

                # Log step 0
                rewards = {aid: 0.0 for aid in ALL_AGENT_IDS}
                collector.log_step(
                    experiment=exp,
                    seed=seed,
                    episode_id=episode_idx,
                    timestep=0,
                    timestamp=start_date + pd.Timedelta(hours=start_hour),
                    obs_dict=obs,
                    reward_dict=rewards,
                    info_dict=info,
                    env_metadata=env_metadata,
                )

                # Run episode rollout
                terminated = {"__all__": False}
                truncated = {"__all__": False}
                t = 0

                while not (terminated["__all__"] or truncated["__all__"]):
                    # Compute action for all agents
                    action = {}

                    if exp in ["trained", "no_battery"]:
                        # Compute action using loaded RLlib policies
                        # third party
                        import torch

                        for aid in ALL_AGENT_IDS:
                            # Standard MAPPO policy mapping logic
                            if aid == COLLEGE_AGENT_ID:
                                pid = "policy_college"
                            elif aid.startswith("solar_"):
                                pid = "policy_solar"
                            else:
                                pid = "policy_consumer"

                            module = modules[pid]
                            device = next(module.parameters()).device

                            # Convert observation to PyTorch tensor
                            obs_tensor = torch.tensor(
                                obs[aid]["obs"], dtype=torch.float32, device=device
                            ).unsqueeze(0)

                            # Run inference
                            with torch.no_grad():
                                output = module.forward_inference({"obs": obs_tensor})

                            # Action distribution inputs contains the mean at [:, :3]
                            # Extract continuous action and convert back to numpy
                            action_dist_inputs = output["action_dist_inputs"]
                            action[aid] = action_dist_inputs[0, :3].cpu().numpy()

                        # Apply battery override for No-Battery Ablation
                        if exp == "no_battery":
                            action[COLLEGE_AGENT_ID][2] = 0.5  # force battery to idle

                    else:
                        # Non-learning baselines
                        for aid in ALL_AGENT_IDS:
                            action[aid] = baseline_policy.compute_actions(obs[aid], aid)

                    # Step environment
                    next_obs, rewards, terminated, truncated, next_info = env.step(
                        action
                    )
                    t += 1

                    # Log timestep
                    collector.log_step(
                        experiment=exp,
                        seed=seed,
                        episode_id=episode_idx,
                        timestep=t,
                        timestamp=start_date + pd.Timedelta(hours=start_hour + t),
                        obs_dict=next_obs,
                        reward_dict=rewards,
                        info_dict=next_info,
                        env_metadata=env_metadata,
                    )

                    obs = next_obs

            # Save CSV files for this seed and experiment
            collector.save_seed_metrics(exp, seed)

    # 8. Compute Cross-Seed Summary metrics
    logger.info("Compiling cross-seed summary metrics...")
    summary_df = collector.compute_summary()
    summary_df.to_csv(results_dir / "summary_metrics.csv", index=False)

    # 9. Perform Significance Testing (RQ1 Welch's t-test + Wilcoxon + Cohen's d)
    logger.info("Running statistical significance testing...")
    ep_df = collector.aggregate_episodes(collector.get_steps_df())
    stats_results = run_significance_tests(ep_df, target_experiment="trained")

    # 10. Separated Plotting data preparation and figure rendering
    logger.info("Generating publication-ready figures...")
    plot_data = prepare_plot_data(collector.get_steps_df(), ep_df)
    generate_plots(plot_data, results_dir)

    # 11. Write Reports (JSON, Markdown, LaTeX)
    logger.info("Writing evaluation summaries and LaTeX tables...")
    generate_reports(collector, stats_results, summary_df, results_dir)
    logger.info("Evaluation complete! Results saved in: %s", results_dir)


def main() -> None:
    """CLI entry point for evaluation execution."""
    parser = argparse.ArgumentParser(
        description="P2P Energy Trading Evaluation CLI (Module 9)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/eval_config.yaml",
        help="Path to evaluation config YAML.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint folder to restore.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to save evaluation output results.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Comma-separated list of experiments to run (e.g."
        " trained,grid_only,random,heuristic,no_battery).",
    )
    args = parser.parse_args()

    experiments_list = None
    if args.experiments:
        experiments_list = [e.strip() for e in args.experiments.split(",")]

    try:
        run_evaluation(
            eval_config_path=args.config,
            checkpoint_path_override=args.checkpoint,
            output_dir_override=args.output_dir,
            experiments_override=experiments_list,
        )
    except Exception as e:
        logger.exception(
            "Evaluation execution encountered an unhandled exception: %s", e
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
