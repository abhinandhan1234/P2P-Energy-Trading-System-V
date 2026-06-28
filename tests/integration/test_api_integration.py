"""Integration tests for Module 10 API Layer.

Mocks subprocess invocation to keep tests fast, verifying directory sandboxes,
config validations, state transitions, log parsing, results queries, and concurrency limits.

Design reference: docs/module_12_repository_structure.md §7
"""

from __future__ import annotations

# standard library
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# third party
import pytest

# local
from p2p_energy_trading.api import (
    P2PExperimentAPI,
    ResourceError,
    TrainingRequest,
)


@pytest.fixture
def temp_experiments_dir(tmp_path: Path) -> Path:
    """Fixture providing a clean temporary experiments directory."""
    d = tmp_path / "experiments"
    d.mkdir()
    return d


@pytest.fixture
def mock_training_config(tmp_path: Path) -> Path:
    """Fixture creating a minimal valid configuration file."""
    config_path = tmp_path / "test_training_config.yaml"
    config_data = {
        "environment": {
            "episode_length": 168,
            "grid_buy_rate": 15.0,
            "grid_sell_rate": 5.0,
            "profile_data_dir": "data/processed",
        },
        "ppo": {
            "lr": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_param": 0.2,
            "entropy_coeff": 0.01,
            "vf_loss_coeff": 1.0,
            "train_batch_size_per_learner": 100,
            "sgd_minibatch_size": 10,
            "actor": {"hidden_layers": [64, 64]},
            "critic": {"hidden_layers": [128, 128]},
        },
        "curriculum": {"stages": {"debug": {}, "training": {}}},
        "hardware": {"num_env_runners": 1, "num_gpus_per_learner_worker": 0},
    }
    with open(config_path, "w", encoding="utf-8") as f:
        # third party
        import yaml

        yaml.safe_dump(config_data, f)
    return config_path


def test_api_concurrency_guard_and_lifecycle(
    temp_experiments_dir: Path, mock_training_config: Path
) -> None:
    """Test start, status, concurrency rejection, and self-healing process tracking."""
    api = P2PExperimentAPI(base_dir=temp_experiments_dir)

    # 1. Mock Popen to simulate a running training process
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.poll.return_value = None  # Process is running

    with (
        patch(
            "p2p_energy_trading.api.services.training_service._is_process_alive",
            return_value=True,
        ),
        patch(
            "p2p_energy_trading.api.experiment_api._is_process_alive",
            return_value=True,
        ),
    ):
        with patch("subprocess.Popen", return_value=mock_proc):
            req = TrainingRequest(
                config_path=str(mock_training_config),
                experiment_name="My Test Run",
                max_iterations=10,
            )
            record1 = api.start_training(req)

            assert record1.state == "RUNNING"
            assert record1.pid == 12345
            assert record1.experiment_name == "My Test Run"

            # Verify sandbox folders exist
            exp_dir = temp_experiments_dir / record1.experiment_id
            assert exp_dir.exists()
            assert (exp_dir / "checkpoints").exists()
            assert (exp_dir / "logs").exists()
            assert (exp_dir / "results").exists()

            # 2. Attempt starting a second training run -> should fail due to concurrency lock
            with pytest.raises(ResourceError) as exc_info:
                api.start_training(req)
            assert "concurrency limit reached" in str(exc_info.value).lower()

    # 3. Simulate process dying outside API by patching process liveness check to False
    with (
        patch(
            "p2p_energy_trading.api.services.training_service._is_process_alive",
            return_value=False,
        ),
        patch(
            "p2p_energy_trading.api.experiment_api._is_process_alive",
            return_value=False,
        ),
    ):
        # Starting training now should self-heal (transition record1 to FAILED)
        # and start the new process
        mock_proc2 = MagicMock()
        mock_proc2.pid = 12346
        mock_proc2.poll.return_value = None

        req2 = TrainingRequest(
            config_path=str(mock_training_config),
            experiment_name="My Test Run 2",
            max_iterations=10,
            seed=999,
        )
        with patch("subprocess.Popen", return_value=mock_proc2):
            record2 = api.start_training(req2)
            assert record2.state == "RUNNING"
            assert record2.pid == 12346

            # Check that record1 was moved to FAILED
            updated_record1 = api.get_experiment(record1.experiment_id)
            assert updated_record1.state == "FAILED"


def test_status_log_parsing(
    temp_experiments_dir: Path, mock_training_config: Path
) -> None:
    """Verify log parser parses iteration stats and best checkpoints correctly."""
    api = P2PExperimentAPI(base_dir=temp_experiments_dir)

    mock_proc = MagicMock()
    mock_proc.pid = 98765
    mock_proc.poll.return_value = None

    with (
        patch(
            "p2p_energy_trading.api.services.training_service._is_process_alive",
            return_value=True,
        ),
        patch(
            "p2p_energy_trading.api.experiment_api._is_process_alive",
            return_value=True,
        ),
    ):
        with patch("subprocess.Popen", return_value=mock_proc):
            req = TrainingRequest(config_path=str(mock_training_config))
            record = api.start_training(req)

            # Write mock console output logs
            log_file = Path(record.log_dir) / "training.log"
            mock_log_content = (
                "2026-06-27 12:00:00 | INFO | p2p_energy_trading | Initializing...\n"
                "\n[Iter 0005 | Stage: debug | Phase: 1 | Steps: 0.05M]\n"
                "  Reward: college=1.20  solar=2.30  consumer=0.80  mean=1.50\n"
                "  Market: P2P_vol=120.0kWh  util=0.75  campus_cost=₹1,200\n"
                "  Grid:   violations=0.0  min_V=0.980  max_loading=0.15\n"
                "  Battery: SoC_mean=0.55  cycles=0.0\n"
                "  Training: loss=0.1234  entropy=0.820  KL=0.0012\n"
            )
            with open(log_file, "w", encoding="utf-8") as lf:
                lf.write(mock_log_content)

            # Write mock best model metadata
            best_dir = Path(record.checkpoint_dir) / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            best_meta = {"mean_eval_reward": 4.56, "grid_violation_rate": 0.01}
            with open(
                best_dir / "best_model_metadata.json", "w", encoding="utf-8"
            ) as f:
                json.dump(best_meta, f)

            # Query status
            status = api.get_status(record.experiment_id)

            assert status.current_iteration == 5
            assert status.current_stage == "debug"
            assert status.agent_steps == 50000
            assert status.best_reward == 4.56

            metrics = status.metrics_summary
            assert metrics is not None
            assert metrics["mean_reward"] == 1.50
            assert metrics["p2p_volume"] == 120.0
            assert metrics["loss_total"] == 0.1234
            assert metrics["entropy"] == 0.820


def test_results_metrics_and_figures_retrieval(
    temp_experiments_dir: Path, mock_training_config: Path
) -> None:
    """Test retrieving summary metrics, listing plots, and compiling download ZIPs."""
    api = P2PExperimentAPI(base_dir=temp_experiments_dir)

    mock_proc = MagicMock()
    mock_proc.pid = 11111

    with (
        patch(
            "p2p_energy_trading.api.services.training_service._is_process_alive",
            return_value=True,
        ),
        patch(
            "p2p_energy_trading.api.experiment_api._is_process_alive",
            return_value=True,
        ),
    ):
        with patch("subprocess.Popen", return_value=mock_proc):
            req = TrainingRequest(config_path=str(mock_training_config))
            record = api.start_training(req)

            results_dir = Path(record.results_dir)

            # 1. Write mock summary metrics CSV
            summary_csv = results_dir / "summary_metrics.csv"
            df_content = (
                "experiment,seed,mean_reward,violations\n"
                "trained,42,2.34,0.01\n"
                "grid_only,42,1.20,0.05\n"
            )
            with open(summary_csv, "w", encoding="utf-8") as f:
                f.write(df_content)

            # 2. Write mock plots
            plots_dir = results_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            open(plots_dir / "fig_cost_comparison.png", "w").close()
            open(plots_dir / "fig_cost_comparison.pdf", "w").close()
            open(plots_dir / "fig_reward_curves.png", "w").close()

            # 3. Query metrics
            metrics_res = api.results.get_metrics(record.experiment_id, level="summary")
            assert metrics_res.row_count == 2
            assert metrics_res.columns == [
                "experiment",
                "seed",
                "mean_reward",
                "violations",
            ]
            assert metrics_res.data[0]["mean_reward"] == 2.34

            # 4. Query figures
            figs = api.results.get_figures(record.experiment_id)
            assert len(figs) == 2
            fig_names = [f.name for f in figs]
            assert "fig_cost_comparison" in fig_names
            assert "fig_reward_curves" in fig_names

            cost_fig = next(f for f in figs if f.name == "fig_cost_comparison")
            assert (
                cost_fig.description == "Total campus energy cost comparison (95% CI)"
            )
            assert cost_fig.path_pdf.endswith("fig_cost_comparison.pdf")

            # 5. Download zip archive
            zip_path_str = api.results.download(
                record.experiment_id, artifact_type="results"
            )
            zip_path = Path(zip_path_str)
            assert zip_path.exists()
            assert zip_path.suffix == ".zip"

            # Verify ZIP contains summary_metrics.csv
            with zipfile.ZipFile(zip_path, "r") as z:
                namelist = z.namelist()
            assert "results/summary_metrics.csv" in namelist
            # Verify plots folder was excluded according to download filter rules
            assert not any("plots" in name for name in namelist)


def test_config_service_comparison(
    temp_experiments_dir: Path, mock_training_config: Path, tmp_path: Path
) -> None:
    """Verify ConfigService identifies safe vs breaking configuration diffs."""
    api = P2PExperimentAPI(base_dir=temp_experiments_dir)

    # Copy base config to create config_b
    config_b_path = tmp_path / "config_b.yaml"
    shutil.copy(mock_training_config, config_b_path)

    # Modify parameters in config_b:
    # 1. safe change: lr, gamma
    # 2. breaking change: critic network architecture
    # third party
    import yaml

    with open(config_b_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["ppo"]["lr"] = 5e-4  # safe change
    cfg["ppo"]["critic"]["hidden_layers"] = [256, 256]  # breaking change

    with open(config_b_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    res = api.config.compare(str(mock_training_config), str(config_b_path))
    assert res.identical is False
    assert len(res.differences) == 2

    # Check safe difference
    safe_fields = [d.field for d in res.safe_differences]
    assert "ppo.lr" in safe_fields

    # Check breaking difference
    breaking_fields = [d.field for d in res.breaking_differences]
    assert "ppo.critic.hidden_layers" in breaking_fields
