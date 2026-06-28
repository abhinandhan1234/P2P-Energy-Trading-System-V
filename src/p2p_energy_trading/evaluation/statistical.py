"""Statistical significance testing for the evaluation framework.

Calculates Welch's t-tests, Cohen's d, Wilcoxon signed-rank tests, and 95%
confidence intervals for comparison metrics between experiments.

Design reference: docs/module_9_evaluation_framework.md §8
"""

from __future__ import annotations

# standard library
import math
from typing import Any

# third party
import numpy as np
import pandas as pd

# Check for SciPy presence to enforce strict import check
try:
    # third party
    import scipy.stats as stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def calculate_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Formula:
    d = (mean1 - mean2) / pooled_std
    where pooled_std = sqrt((std1^2 + std2^2) / 2)
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Handle zero-variance case
    if math.isclose(std1, 0.0, abs_tol=1e-8) and math.isclose(std2, 0.0, abs_tol=1e-8):
        return 0.0

    pooled_std = math.sqrt((std1**2 + std2**2) / 2.0)
    if pooled_std == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_std)


def calculate_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate the 95% confidence interval for a given 1D array of seed values.

    For small sample size (e.g. N=5 seeds), uses t-distribution with df = N - 1.
    """
    n = len(data)
    if n <= 1:
        return float(np.mean(data)), 0.0

    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))

    if not SCIPY_AVAILABLE:
        # Standard t-critical fallback for df = 4, alpha = 0.05 two-sided is 2.776
        t_critical = 2.7764 if n == 5 else 1.96
    else:
        df = n - 1
        t_critical = float(stats.t.ppf((1 + confidence) / 2.0, df))

    margin_of_error = t_critical * (std / math.sqrt(n))
    return mean, margin_of_error


def run_significance_tests(
    episode_df: pd.DataFrame,
    target_experiment: str = "trained",
) -> dict[str, dict[str, Any]]:
    """Compare a target experiment (e.g. trained) to all baseline strategies.

    Computes Welch's t-test, Wilcoxon signed-rank test, and Cohen's d on episode costs.

    Raises:
        ImportError: If SciPy is missing, preventing silent failure of thesis statistics.
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("SciPy is required for statistical evaluation.")

    # 1. Group episode costs per seed (mean cost of 20 episodes for each seed)
    seed_means = (
        episode_df.groupby(["experiment", "seed"])["total_cost"].mean().reset_index()
    )

    target_data = seed_means[seed_means["experiment"] == target_experiment][
        "total_cost"
    ].values
    if len(target_data) == 0:
        # If target experiment was not run, return empty dictionary
        return {}

    comparison_results: dict[str, dict[str, Any]] = {}

    experiments = seed_means["experiment"].unique()
    for exp in experiments:
        if exp == target_experiment:
            continue

        exp_data = seed_means[seed_means["experiment"] == exp]["total_cost"].values
        if len(exp_data) == 0:
            continue

        # 1. Welch's t-test (unequal variance)
        t_stat, p_val = stats.ttest_ind(target_data, exp_data, equal_var=False)

        # 2. Wilcoxon signed-rank test (paired samples since they share seeds/profiles)
        # Note: If size is very small or differences are constant, wilcoxon can raise warnings/errors
        try:
            wilc_stat, wilc_p = stats.wilcoxon(target_data, exp_data)
        except Exception:
            wilc_stat, wilc_p = float("nan"), float("nan")

        # 3. Cohen's d effect size
        d = calculate_cohens_d(target_data, exp_data)

        # 4. Success criteria verification
        significant = bool(p_val < 0.05)

        comparison_results[exp] = {
            "comparison": f"{target_experiment} vs {exp}",
            "welch_t_statistic": float(t_stat),
            "p_value": float(p_val),
            "wilcoxon_statistic": float(wilc_stat),
            "wilcoxon_p_value": float(wilc_p),
            "cohens_d": d,
            "statistically_significant": significant,
        }

    return comparison_results
