"""Metrics collection and aggregation for the evaluation framework.

Collects and compiles evaluation metrics exclusively from values returned by the
environment (reward, info, observations, and episode metadata). It does not
recompute market settlements, battery physics, or grid power flows.

Design reference: docs/module_9_evaluation_framework.md §4-7
"""

from __future__ import annotations

# standard library
from pathlib import Path
from typing import Any

# third party
import numpy as np
import pandas as pd

# local
from p2p_energy_trading.constants import ALL_AGENT_IDS, COLLEGE_AGENT_ID


class MetricCollector:
    """Collects and aggregates evaluation metrics from environment steps."""

    def __init__(self, results_dir: str | Path = "results") -> None:
        """Initialize the MetricCollector.

        Args:
            results_dir: Base directory to save output metrics CSV files.
        """
        self.results_dir = Path(results_dir)
        self.raw_steps: list[dict[str, Any]] = []

    def log_step(
        self,
        experiment: str,
        seed: int,
        episode_id: int,
        timestep: int,
        timestamp: pd.Timestamp,
        obs_dict: dict[str, Any],
        reward_dict: dict[str, float],
        info_dict: dict[str, Any],
        env_metadata: dict[str, Any],
    ) -> None:
        """Log metrics for a single simulation timestep.

        Extracts all variables strictly from the environment step outputs.
        """
        for aid in ALL_AGENT_IDS:
            agent_obs = obs_dict[aid]["obs"]
            agent_info = info_dict[aid]

            # 1. Retrieve metadata peaks to unnormalise energy quantities
            b_meta = env_metadata.get("buildings", {}).get(aid, {})
            p_stats = b_meta.get("profile_stats", {})
            peak_solar = p_stats.get("solar_generation_kw", {}).get("peak", 0.0)
            peak_demand = p_stats.get("demand_kw", {}).get("peak", 0.0)

            # 2. Recover unnormalised demand/solar from observations
            normalised_solar = float(agent_obs[0])
            normalised_demand = float(agent_obs[1])
            solar_kw = normalised_solar * peak_solar
            demand_kw = normalised_demand * peak_demand

            # 3. Retrieve battery SoC (index 2 in local obs vector)
            battery_soc = float(agent_obs[2])

            # 4. Determine agent type
            if aid == COLLEGE_AGENT_ID:
                agent_type = "college"
            elif aid.startswith("solar_"):
                agent_type = "solar"
            else:
                agent_type = "consumer"

            self.raw_steps.append(
                {
                    "experiment": experiment,
                    "seed": seed,
                    "episode_id": episode_id,
                    "timestep": timestep,
                    "timestamp": timestamp,
                    "agent_id": aid,
                    "agent_type": agent_type,
                    "demand_kw": demand_kw,
                    "solar_kw": solar_kw,
                    "p2p_bought_kw": agent_info.get("p2p_bought_kw", 0.0),
                    "p2p_sold_kw": agent_info.get("p2p_sold_kw", 0.0),
                    "grid_bought_kw": agent_info.get("grid_bought_kw", 0.0),
                    "grid_sold_kw": agent_info.get("grid_sold_kw", 0.0),
                    "net_cost": agent_info.get("net_cost", 0.0),
                    "voltage_violation": bool(
                        agent_info.get("voltage_violation", False)
                    ),
                    "thermal_violation": bool(
                        agent_info.get("thermal_violation", False)
                    ),
                    "battery_soc": battery_soc,
                    "reward": reward_dict.get(aid, 0.0),
                }
            )

    def get_steps_df(self) -> pd.DataFrame:
        """Convert collected raw step logs to a pandas DataFrame."""
        if not self.raw_steps:
            return pd.DataFrame()
        return pd.DataFrame(self.raw_steps)

    def save_seed_metrics(self, experiment: str, seed: int) -> None:
        """Save raw step and episode summaries for a specific experiment and seed.

        Exports them under a deterministic directory structure:
        results/seed_<seed>/<experiment>_per_step.csv and per_episode.csv
        """
        df = self.get_steps_df()
        if df.empty:
            return

        # Filter for this experiment and seed
        seed_df = df[(df["experiment"] == experiment) & (df["seed"] == seed)]
        if seed_df.empty:
            return

        # Create target directory
        seed_dir = self.results_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Write per-step CSV
        step_file = seed_dir / f"{experiment}_per_step.csv"
        seed_df.to_csv(step_file, index=False)

        # Compute per-episode aggregations
        ep_agg = self.aggregate_episodes(seed_df)
        ep_file = seed_dir / f"{experiment}_per_episode.csv"
        ep_agg.to_csv(ep_file, index=False)

    def aggregate_episodes(self, step_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-step DataFrame to per-episode metrics."""
        # Group by experiment, seed, episode_id
        grouped = step_df.groupby(["experiment", "seed", "episode_id"])

        episodes_list: list[dict[str, Any]] = []
        for (exp, seed, ep_id), group in grouped:
            # 1. Financial: Sum cost over all agents and timesteps
            total_cost = group["net_cost"].sum()

            # 2. Market metrics
            # Total cleared P2P volume is the sum of bought or sold
            # (they are equal for the microgrid)
            total_p2p_volume = group["p2p_bought_kw"].sum()

            # Max clearable volume is the sum over timesteps of
            # min(total_surplus, total_deficit)
            # Find step-level available surplus/deficit
            step_group = group.groupby("timestep")
            max_clearable_list = []
            for _, step_data in step_group:
                surpluses = (step_data["solar_kw"] - step_data["demand_kw"]).clip(
                    lower=0.0
                )
                deficits = (step_data["demand_kw"] - step_data["solar_kw"]).clip(
                    lower=0.0
                )
                # Note: battery dispatch is handled inside environment obs/physics,
                # but max clearable is based on base profiles
                max_clearable_list.append(min(surpluses.sum(), deficits.sum()))
            max_clearable = sum(max_clearable_list)

            p2p_utilisation = (
                total_p2p_volume / max_clearable if max_clearable > 0 else 1.0
            )

            # 3. Grid Violations Rate
            # Voltage violation occurs if voltage_violation is True for any
            # agent in a timestep
            # Total timesteps = length of step_group
            total_steps = len(step_group)
            voltage_violations = 0
            thermal_violations = 0
            for _, step_data in step_group:
                if step_data["voltage_violation"].any():
                    voltage_violations += 1
                if step_data["thermal_violation"].any():
                    thermal_violations += 1

            voltage_violation_rate = (
                voltage_violations / total_steps if total_steps > 0 else 0.0
            )
            thermal_violation_rate = (
                thermal_violations / total_steps if total_steps > 0 else 0.0
            )

            # 4. Battery Stats (College agent only)
            college_group = group[group["agent_id"] == COLLEGE_AGENT_ID]
            socs = college_group["battery_soc"].values
            soc_mean = socs.mean() if len(socs) > 0 else 0.5
            soc_min = socs.min() if len(socs) > 0 else 0.5
            soc_max = socs.max() if len(socs) > 0 else 0.5

            # Calculate charging/discharging throughput (kw = kwh since dt=1 hour)
            # We can compute difference in SoC to see throughput
            soc_diffs = np.diff(socs)
            charge_throughput = (
                sum(d for d in soc_diffs if d > 0) * 500.0
            )  # capacity=500
            discharge_throughput = sum(-d for d in soc_diffs if d < 0) * 500.0
            total_throughput = charge_throughput + discharge_throughput
            equivalent_cycles = total_throughput / (2.0 * 500.0)

            # 5. economic welfare (Savings over grid-only counterfactual)
            # grid_buy_rate = 13.78, grid_sell_rate = 8.10, p2p_clearing_price = 10.94
            # Buyer welfare: sum_buyers (13.78 - 10.94) * p2p_bought_kw
            # = 2.84 * sum(p2p_bought)
            # Seller welfare: sum_sellers (10.94 - 8.10) * p2p_sold_kw
            # = 2.84 * sum(p2p_sold)
            buyer_welfare = 2.84 * group["p2p_bought_kw"].sum()
            seller_welfare = 2.84 * group["p2p_sold_kw"].sum()
            campus_welfare = buyer_welfare + seller_welfare

            # 6. Grid interaction volumes
            grid_import = group["grid_bought_kw"].sum()
            grid_export = group["grid_sold_kw"].sum()

            # 7. Rewards
            mean_reward = group["reward"].mean()
            episode_return = group["reward"].sum()

            episodes_list.append(
                {
                    "experiment": exp,
                    "seed": seed,
                    "episode_id": ep_id,
                    "total_cost": total_cost,
                    "p2p_volume": total_p2p_volume,
                    "p2p_utilisation": p2p_utilisation,
                    "voltage_violation_rate": voltage_violation_rate,
                    "thermal_violation_rate": thermal_violation_rate,
                    "battery_soc_mean": soc_mean,
                    "battery_soc_min": soc_min,
                    "battery_soc_max": soc_max,
                    "battery_throughput": total_throughput,
                    "battery_cycles": equivalent_cycles,
                    "buyer_welfare": buyer_welfare,
                    "seller_welfare": seller_welfare,
                    "campus_welfare": campus_welfare,
                    "grid_import": grid_import,
                    "grid_export": grid_export,
                    "mean_reward": mean_reward,
                    "episode_return": episode_return,
                }
            )

        return pd.DataFrame(episodes_list)

    def compute_summary(self, baseline_cost_mean: float | None = None) -> pd.DataFrame:
        """Aggregate episode metrics to cross-seed statistics for each experiment.

        Computes mean, standard deviation, and min/max across seeds.
        """
        df = self.get_steps_df()
        if df.empty:
            return pd.DataFrame()

        # Step 1. Get episode metrics
        ep_df = self.aggregate_episodes(df)

        # Step 2. Get per-seed mean (mean of 20 episodes for each seed)
        seed_df = ep_df.groupby(["experiment", "seed"]).mean().reset_index()

        # Step 3. Group by experiment to compute cross-seed statistics (N=5 seeds)
        summary_list: list[dict[str, Any]] = []
        for exp, exp_data in seed_df.groupby("experiment"):
            cost_mean = exp_data["total_cost"].mean()
            cost_std = exp_data["total_cost"].std()

            p2p_mean = exp_data["p2p_volume"].mean()
            p2p_std = exp_data["p2p_volume"].std()

            volt_mean = exp_data["voltage_violation_rate"].mean()
            volt_std = exp_data["voltage_violation_rate"].std()

            therm_mean = exp_data["thermal_violation_rate"].mean()
            therm_std = exp_data["thermal_violation_rate"].std()

            util_mean = exp_data["p2p_utilisation"].mean()
            util_std = exp_data["p2p_utilisation"].std()

            welfare_mean = exp_data["campus_welfare"].mean()
            welfare_std = exp_data["campus_welfare"].std()

            import_mean = exp_data["grid_import"].mean()
            import_std = exp_data["grid_import"].std()

            export_mean = exp_data["grid_export"].mean()
            export_std = exp_data["grid_export"].std()

            ret_mean = exp_data["episode_return"].mean()
            ret_std = exp_data["episode_return"].std()

            cycles_mean = exp_data["battery_cycles"].mean()
            cycles_std = exp_data["battery_cycles"].std()

            summary_list.append(
                {
                    "experiment": exp,
                    "cost_mean": cost_mean,
                    "cost_std": cost_std,
                    "p2p_volume_mean": p2p_mean,
                    "p2p_volume_std": p2p_std,
                    "voltage_violation_rate_mean": volt_mean,
                    "voltage_violation_rate_std": volt_std,
                    "thermal_violation_rate_mean": therm_mean,
                    "thermal_violation_rate_std": therm_std,
                    "p2p_utilisation_mean": util_mean,
                    "p2p_utilisation_std": util_std,
                    "campus_welfare_mean": welfare_mean,
                    "campus_welfare_std": welfare_std,
                    "grid_import_mean": import_mean,
                    "grid_import_std": import_std,
                    "grid_export_mean": export_mean,
                    "grid_export_std": export_std,
                    "episode_return_mean": ret_mean,
                    "episode_return_std": ret_std,
                    "battery_cycles_mean": cycles_mean,
                    "battery_cycles_std": cycles_std,
                }
            )

        summary_df = pd.DataFrame(summary_list)

        # Calculate cost reduction percentage if Grid-Only baseline cost is available
        if baseline_cost_mean is not None and baseline_cost_mean > 0:
            summary_df["cost_reduction_pct"] = (
                (baseline_cost_mean - summary_df["cost_mean"])
                / baseline_cost_mean
                * 100.0
            )
        else:
            # Fall back to using "grid_only" experiment if present
            grid_only_row = summary_df[summary_df["experiment"] == "grid_only"]
            if not grid_only_row.empty:
                go_cost = float(grid_only_row["cost_mean"].iloc[0])
                if go_cost > 0:
                    summary_df["cost_reduction_pct"] = (
                        (go_cost - summary_df["cost_mean"]) / go_cost * 100.0
                    )
                else:
                    summary_df["cost_reduction_pct"] = 0.0
            else:
                summary_df["cost_reduction_pct"] = 0.0

        return summary_df
