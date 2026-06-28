"""Evaluation Framework Package — Module 9.

Orchestrates benchmarking of trained policies against baseline controllers (Grid-Only,
Random, and Heuristic), compiles detailed metrics (financial, safety, and battery),
calculates statistical significances, and generates plots and LaTeX tables.

Design reference: docs/module_9_evaluation_framework.md
"""

from __future__ import annotations

# local
from p2p_energy_trading.evaluation.baselines import (
    BaselinePolicy,
    GridOnlyBaseline,
    HeuristicBaseline,
    RandomBaseline,
)
from p2p_energy_trading.evaluation.evaluate import main
from p2p_energy_trading.evaluation.metrics import MetricCollector
from p2p_energy_trading.evaluation.plotting import generate_plots, prepare_plot_data
from p2p_energy_trading.evaluation.report import generate_reports
from p2p_energy_trading.evaluation.statistical import run_significance_tests

__all__ = [
    "BaselinePolicy",
    "GridOnlyBaseline",
    "RandomBaseline",
    "HeuristicBaseline",
    "MetricCollector",
    "run_significance_tests",
    "prepare_plot_data",
    "generate_plots",
    "generate_reports",
    "main",
]
