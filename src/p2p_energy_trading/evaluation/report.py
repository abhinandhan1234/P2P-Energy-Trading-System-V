"""Report generation for the evaluation framework.

Consumes pre-aggregated MetricCollector summaries and statistical significance tests,
outputting JSON, Markdown, and publication-ready LaTeX tables without recomputing
any metric values.

Design reference: docs/module_9_evaluation_framework.md §11
"""

from __future__ import annotations

# standard library
import json
from pathlib import Path
from typing import Any

# third party
import pandas as pd

# local
from p2p_energy_trading.evaluation.metrics import MetricCollector


def generate_reports(
    collector: MetricCollector,
    stats_results: dict[str, dict[str, Any]],
    summary_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Generate all evaluation reports (JSON, Markdown, LaTeX) in output directories."""
    output_dir = Path(output_dir)
    latex_dir = output_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)

    # Convert summary DataFrame to dictionary mapping experiment name
    # to its aggregated row
    summary_dict = summary_df.set_index("experiment").to_dict(orient="index")

    # 1. JSON Report
    json_data = {
        "summary": summary_dict,
        "significance_tests": stats_results,
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    # 2. LaTeX Table: Main Performance Results (RQ1)
    latex_main = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Primary Performance and Cost Reduction Comparison}",
        "\\label{table:main_results}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Experiment & Mean Cost (\\rupee) & Cost Red. (\\%) "
        "& p-value (Welch's t) & Effect Size (Cohen's d) \\\\",
        "\\hline",
    ]
    for exp, vals in summary_dict.items():
        cost_str = f"{vals['cost_mean']:.2f} \\pm {vals['cost_std']:.2f}"
        red_str = f"{vals.get('cost_reduction_pct', 0.0):.2f}"

        stat = stats_results.get(exp, {})
        p_val = f"{stat.get('p_value', 1.0):.4f}" if stat else "-"
        cohen_d = f"{stat.get('cohens_d', 0.0):.2f}" if stat else "-"

        latex_main.append(
            f"{exp.replace('_', ' ').capitalize()}"
            f" & {cost_str} & {red_str} & {p_val} & {cohen_d} \\\\",
        )
    latex_main.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(latex_dir / "table_main_results.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_main) + "\n")

    # 3. LaTeX Table: Grid Safety Comparison (RQ2)
    latex_grid = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{IEEE 33-Bus Grid Safety and Constraint Violation Rates}",
        "\\label{table:grid_safety}",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Experiment & Voltage Violation Rate (\\%) & Thermal Overload Rate (\\%) \\\\",
        "\\hline",
    ]
    for exp, vals in summary_dict.items():
        v_str = f"{vals['voltage_violation_rate_mean'] * 100.0:.3f}"
        t_str = f"{vals['thermal_violation_rate_mean'] * 100.0:.3f}"
        latex_grid.append(
            f"{exp.replace('_', ' ').capitalize()} & {v_str} & {t_str} \\\\"
        )
    latex_grid.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(latex_dir / "table_grid_safety.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_grid) + "\n")

    # 4. LaTeX Table: Market and Welfare Metrics
    latex_market = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{P2P Energy Trading Market cleared Volume and Utilisation}",
        "\\label{table:market_metrics}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Experiment & P2P Volume (kWh) & Utilisation (\\%) "
        "& Campus Welfare (\\rupee) \\\\",
        "\\hline",
    ]
    for exp, vals in summary_dict.items():
        p2p_str = f"{vals['p2p_volume_mean']:.2f} \\pm {vals['p2p_volume_std']:.2f}"
        util_str = f"{vals['p2p_utilisation_mean'] * 100.0:.2f}"
        wel_str = f"{vals['campus_welfare_mean']:.2f}"
        latex_market.append(
            f"{exp.replace('_', ' ').capitalize()}"
            f" & {p2p_str} & {util_str} & {wel_str} \\\\",
        )
    latex_market.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(latex_dir / "table_market_metrics.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_market) + "\n")

    # 5. LaTeX Table: Ablation Study Comparison (RQ3, RQ4)
    latex_ablation = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{Ablation Study Comparison of Battery and P2P Integration}",
        "\\label{table:ablation_results}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Experiment & Cost Savings (\\%) & P2P Volume (kWh) & Battery Cycles \\\\",
        "\\hline",
    ]
    for exp in ["trained", "no_battery", "no_p2p"]:
        if exp not in summary_dict:
            continue
        vals = summary_dict[exp]
        red_str = f"{vals.get('cost_reduction_pct', 0.0):.2f}"
        p2p_str = f"{vals['p2p_volume_mean']:.2f}"
        cycle_str = f"{vals['battery_cycles_mean']:.3f}"
        latex_ablation.append(
            f"{exp.replace('_', ' ').capitalize()}"
            f" & {red_str} & {p2p_str} & {cycle_str} \\\\",
        )
    latex_ablation.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    with open(latex_dir / "table_ablation_results.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_ablation) + "\n")

    # 6. Markdown Summary Report
    md_content = [
        "# P2P Energy Trading System - Evaluation Framework Summary Report",
        "",
        "This report documents performance benchmarks for MAPPO policies"
        " against non-learning baselines.",
        "",
        "## Performance Metrics Overview",
        "",
        "| Experiment | Mean Cost (₹) | Cost Red. vs Grid (%)"
        " | Voltage Violations (%) | P2P Utilisation (%) |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for exp, vals in summary_dict.items():
        cost_str = f"{vals['cost_mean']:.2f} ± {vals['cost_std']:.2f}"
        red_str = f"{vals.get('cost_reduction_pct', 0.0):.2f}%"
        v_str = f"{vals['voltage_violation_rate_mean'] * 100.0:.3f}%"
        u_str = f"{vals['p2p_utilisation_mean'] * 100.0:.2f}%"
        md_content.append(
            f"| {exp.replace('_', ' ').capitalize()}"
            f" | {cost_str} | {red_str} | {v_str} | {u_str} |",
        )

    md_content.extend(
        [
            "",
            "## Statistical Significance Analysis",
            "",
            " Welch's t-test comparing the trained policy against other"
            " baseline policies on total cost (alpha = 0.05):",
            "",
        ]
    )
    for exp, stat in stats_results.items():
        sig_str = (
            "**Significant**"
            if stat["statistically_significant"]
            else "Not Significant"
        )
        md_content.append(
            f"- **vs {exp.replace('_', ' ').capitalize()}**:"
            f" p = {stat['p_value']:.5f}, "
            f"Cohen's d = {stat['cohens_d']:.3f} ({sig_str})",
        )

    md_content.extend(["", "## Verification Success Thresholds", ""])

    # Check Success Thresholds
    trained_vals = summary_dict.get("trained")
    if trained_vals:
        cost_red = trained_vals.get("cost_reduction_pct", 0.0)
        v_violation = trained_vals.get("voltage_violation_rate_mean", 0.0) * 100.0
        t_violation = trained_vals.get("thermal_violation_rate_mean", 0.0) * 100.0
        p2p_util = trained_vals.get("p2p_utilisation_mean", 0.0) * 100.0

        md_content.append(
            f"- **Cost reduction (>= 10%)**: {cost_red:.2f}%"
            f" - {'PASSED' if cost_red >= 10.0 else 'FAILED'}",
        )
        md_content.append(
            f"- **Voltage Safety (< 1%)**: {v_violation:.3f}%"
            f" - {'PASSED' if v_violation < 1.0 else 'FAILED'}",
        )
        md_content.append(
            f"- **Thermal Safety (< 1%)**: {t_violation:.3f}%"
            f" - {'PASSED' if t_violation < 1.0 else 'FAILED'}",
        )
        md_content.append(
            f"- **P2P Utilisation (> 60%)**: {p2p_util:.2f}%"
            f" - {'PASSED' if p2p_util > 60.0 else 'FAILED'}",
        )

    with open(output_dir / "evaluation_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_content) + "\n")
