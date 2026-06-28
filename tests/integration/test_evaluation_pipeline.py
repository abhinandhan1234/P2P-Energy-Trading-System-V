"""Integration tests for the P2P Energy Trading evaluation framework.

Design reference: docs/module_9_evaluation_framework.md §12
"""

from __future__ import annotations

# standard library
import json
from pathlib import Path

# third party
import pandas as pd
import yaml

# local
from p2p_energy_trading.evaluation.evaluate import run_evaluation


def test_evaluation_pipeline_execution(tmp_path: Path) -> None:
    """Run a short evaluation pipeline integration run and verify all artifacts.

    Verifies summary_metrics.csv, evaluation_summary.json, plots, and latex files.
    """
    # 1. Create a mock eval_config.yaml in tmp_path
    mock_eval_cfg = {
        "training_config": "config/training_config.yaml",
        "checkpoint_path": "checkpoints/best_model",
        "evaluation": {
            "num_seeds": 2,
            "seeds": [42, 123],
            "num_episodes_per_seed": 2,
            # We use small start hours to make it fast
            "eval_episode_starts": [0, 168],
            "results_dir": str(tmp_path / "results"),
            "experiments": ["grid_only", "random", "heuristic"],
        },
    }

    cfg_file = tmp_path / "eval_config.yaml"
    with open(cfg_file, "w") as f:
        yaml.dump(mock_eval_cfg, f)

    # 2. Run the evaluation pipeline (non-learning baselines only, no Ray required)
    run_evaluation(
        eval_config_path=cfg_file,
        output_dir_override=str(tmp_path / "results"),
        experiments_override=["grid_only", "random", "heuristic"],
    )

    results_dir = tmp_path / "results"

    # 3. Assert deterministic folder structure and per-seed files exist
    assert (results_dir / "seed_42").exists()
    assert (results_dir / "seed_123").exists()
    assert (results_dir / "seed_42" / "grid_only_per_step.csv").exists()
    assert (results_dir / "seed_42" / "grid_only_per_episode.csv").exists()

    # 4. Assert summary_metrics.csv is created with correct columns
    summary_file = results_dir / "summary_metrics.csv"
    assert summary_file.exists()

    summary_df = pd.read_csv(summary_file)
    expected_cols = [
        "experiment",
        "cost_mean",
        "cost_std",
        "p2p_volume_mean",
        "p2p_volume_std",
        "voltage_violation_rate_mean",
        "voltage_violation_rate_std",
        "p2p_utilisation_mean",
        "campus_welfare_mean",
        "grid_import_mean",
        "grid_export_mean",
    ]
    for col in expected_cols:
        assert col in summary_df.columns

    assert set(summary_df["experiment"]) == {"grid_only", "random", "heuristic"}

    # 5. Assert evaluation_summary.json is created and has valid structure
    summary_json = results_dir / "evaluation_summary.json"
    assert summary_json.exists()

    with open(summary_json) as f:
        json_data = json.load(f)

    assert "summary" in json_data
    assert "significance_tests" in json_data
    assert "grid_only" in json_data["summary"]
    assert "random" in json_data["summary"]
    assert "heuristic" in json_data["summary"]

    # 6. Assert plots directory exists and contains key rendered figures
    plots_dir = results_dir / "plots"
    assert plots_dir.exists()
    assert (plots_dir / "fig_cost_comparison.png").exists()
    assert (plots_dir / "fig_cost_comparison.pdf").exists()
    assert (plots_dir / "fig_p2p_volume.png").exists()
    assert (plots_dir / "fig_violation_rates.png").exists()
    assert (plots_dir / "fig_battery_soc_profile.png").exists()

    # 7. Assert latex tables directory exists and contains tex files
    latex_dir = results_dir / "latex"
    assert latex_dir.exists()
    assert (latex_dir / "table_main_results.tex").exists()
    assert (latex_dir / "table_grid_safety.tex").exists()
    assert (latex_dir / "table_market_metrics.tex").exists()
    assert (latex_dir / "table_ablation_results.tex").exists()

    # Verify latex table contents
    with open(latex_dir / "table_main_results.tex") as f:
        latex_content = f.read()
        assert "Experiment" in latex_content
        assert "Grid only" in latex_content
        assert "Random" in latex_content
        assert "Heuristic" in latex_content
