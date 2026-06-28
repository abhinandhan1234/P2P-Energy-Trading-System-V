"""Results Service for Module 10 API Layer.

Retrieves and aggregates CSV metric logs, headlessly rendered plots, report documents,
and LaTeX tables from completed experiment directories. Supports on-demand zip archiving.

Design reference: docs/module_10_api_layer.md §5
"""

from __future__ import annotations

# standard library
import datetime
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

# third party
import pandas as pd

# local
from p2p_energy_trading.api.models import FigureInfo, MetricsResult, ReportResult
from p2p_energy_trading.exceptions import ExperimentNotFoundError

logger = logging.getLogger(__name__)

FIGURE_DESCRIPTIONS = {
    "fig_cost_comparison": "Total campus energy cost comparison (95% CI)",
    "fig_reward_curves": "Training convergence reward progress curves",
    "fig_p2p_volume": "Total P2P energy trading volume (95% CI)",
    "fig_violation_rates": "IEEE 33-Bus voltage/thermal constraints violation rate bar chart",
    "fig_battery_soc_profile": "College battery 24-hour State-of-Charge (SoC) profiles",
    "fig_battery_dispatch_heatmap": "Trained policy college battery dispatch profile heatmap",
    "fig_campus_welfare": "Economic campus buyer and seller welfare breakdown stack bar",
    "fig_cost_reduction_ablation": "Campus cost savings percentage across experiments",
    "fig_agent_type_rewards": "Mean reward comparisons divided by agent type",
    "fig_renewable_utilisation": "Renewable local self-consumption and P2P utilization rates",
    "fig_grid_import_export": "Grid power import vs export volumes comparisons",
    "fig_market_volume_timeseries": "P2P market cleared volume hourly timeline profile",
    "fig_voltage_profile": "Campus network voltage ranges bounding limits",
    "fig_ablation_summary": "Multi-metric ablation studies performance radar chart",
}


class ResultsService:
    """Provides unified access to experiment output metrics, plots, and document archives."""

    def __init__(self, base_dir: Path) -> None:
        """Initialize the results service.

        Args:
            base_dir: Registry root directory experiments/.
        """
        self.base_dir = base_dir
        self.downloads_dir = self.base_dir / "downloads"
        self.downloads_dir.mkdir(exist_ok=True)

    def get_metrics(
        self,
        experiment_id: str,
        level: str = "summary",
        metrics: list[str] | None = None,
    ) -> MetricsResult:
        """Retrieve metrics from a completed experiment, aggregating seed logs if requested."""
        exp_dir = self.base_dir / experiment_id
        if not exp_dir.exists():
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

        results_dir = exp_dir / "results"

        # 1. Read summary metrics
        if level == "summary":
            summary_file = results_dir / "summary_metrics.csv"
            if not summary_file.exists():
                raise FileNotFoundError(
                    f"Summary metrics not found for experiment '{experiment_id}'"
                )
            df = pd.read_csv(summary_file)

        # 2. Aggregate per-episode metrics across all seeds
        elif level == "per_episode":
            dfs = []
            for seed_dir in results_dir.glob("seed_*"):
                ep_file = seed_dir / "per_episode_metrics.csv"
                if ep_file.exists():
                    dfs.append(pd.read_csv(ep_file))
            if not dfs:
                raise FileNotFoundError(
                    f"No episode metrics found for experiment '{experiment_id}'"
                )
            df = pd.concat(dfs, ignore_index=True)

        # 3. Aggregate per-step metrics across all seeds
        elif level == "per_step":
            dfs = []
            for seed_dir in results_dir.glob("seed_*"):
                step_file = seed_dir / "per_step_metrics.csv"
                if step_file.exists():
                    dfs.append(pd.read_csv(step_file))
            if not dfs:
                raise FileNotFoundError(
                    f"No step metrics found for experiment '{experiment_id}'"
                )
            df = pd.concat(dfs, ignore_index=True)
        else:
            raise ValueError(f"Unknown metric aggregation level: '{level}'")

        # Apply filtering if requested
        if metrics:
            # Preserve critical index/key columns
            meta_cols = [
                c
                for c in ["experiment", "seed", "episode_id", "timestep", "agent_id"]
                if c in df.columns
            ]
            filter_cols = meta_cols + [m for m in metrics if m in df.columns]
            df = df[filter_cols]

        data_records = df.to_dict(orient="records")

        return MetricsResult(
            experiment_id=experiment_id,
            level=level,
            data=data_records,
            columns=list(df.columns),
            row_count=len(df),
            generated_at=datetime.datetime.now().isoformat(),
        )

    def get_figures(self, experiment_id: str) -> list[FigureInfo]:
        """List and locate Matplotlib plots generated during evaluation."""
        exp_dir = self.base_dir / experiment_id
        if not exp_dir.exists():
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

        plots_dir = exp_dir / "results" / "plots"
        if not plots_dir.exists():
            return []

        figures = []
        # Find unique figure basenames by looking at PNGs
        for png_file in plots_dir.glob("*.png"):
            basename = png_file.stem
            pdf_file = plots_dir / f"{basename}.pdf"

            description = FIGURE_DESCRIPTIONS.get(
                basename, "Evaluation performance plot"
            )
            stat = png_file.stat()

            info = FigureInfo(
                name=basename,
                path_png=str(png_file),
                path_pdf=str(pdf_file) if pdf_file.exists() else "",
                description=description,
                generated_at=datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                size_bytes=stat.st_size,
            )
            figures.append(info)

        return figures

    def get_report(self, experiment_id: str, format: str = "markdown") -> ReportResult:
        """Retrieve the evaluation report in JSON, Markdown, or LaTeX format."""
        exp_dir = self.base_dir / experiment_id
        if not exp_dir.exists():
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

        results_dir = exp_dir / "results"
        content: str | dict[str, Any] = ""

        # 1. Parse JSON summary
        if format == "json":
            json_file = results_dir / "evaluation_summary.json"
            if not json_file.exists():
                raise FileNotFoundError(
                    f"JSON summary report not found for experiment '{experiment_id}'"
                )
            with open(json_file, encoding="utf-8") as f:
                content = json.load(f)

        # 2. Read Markdown summary
        elif format == "markdown":
            md_file = results_dir / "evaluation_report.md"
            if not md_file.exists():
                raise FileNotFoundError(
                    f"Markdown report file not found for experiment '{experiment_id}'"
                )
            with open(md_file, encoding="utf-8") as f:
                content = f.read()

        # 3. Read LaTeX reports
        elif format == "latex":
            latex_dir = results_dir / "latex"
            tex_contents = {}
            if latex_dir.exists():
                for tex_file in latex_dir.glob("*.tex"):
                    with open(tex_file, encoding="utf-8") as f:
                        tex_contents[tex_file.name] = f.read()
            content = tex_contents
        else:
            raise ValueError(f"Unknown report format: '{format}'")

        # Identify LaTeX tables and figure paths for indexing
        tables = [str(p) for p in results_dir.glob("latex/*.tex")]
        figures = [str(p) for p in results_dir.glob("plots/*.png")]

        return ReportResult(
            experiment_id=experiment_id,
            format=format,
            content=content,
            tables=tables,
            figures=figures,
            generated_at=datetime.datetime.now().isoformat(),
        )

    def download(self, experiment_id: str, artifact_type: str = "all") -> str:
        """Compress experiment files into a zip archive on-demand."""
        exp_dir = self.base_dir / experiment_id
        if not exp_dir.exists():
            raise ExperimentNotFoundError(f"Experiment '{experiment_id}' not found.")

        zip_filename = f"{experiment_id}_{artifact_type}.zip"
        zip_path = self.downloads_dir / zip_filename

        # Filter directories to compress based on type
        paths_to_compress = []
        if artifact_type == "all":
            paths_to_compress = [exp_dir]
        elif artifact_type == "checkpoints" and (exp_dir / "checkpoints").exists():
            paths_to_compress = [exp_dir / "checkpoints"]
        elif artifact_type == "results" and (exp_dir / "results").exists():
            # Include results excluding plots
            paths_to_compress = [
                p for p in (exp_dir / "results").iterdir() if p.name != "plots"
            ]
        elif artifact_type == "figures" and (exp_dir / "results" / "plots").exists():
            paths_to_compress = [exp_dir / "results" / "plots"]
        elif artifact_type == "logs" and (exp_dir / "logs").exists():
            paths_to_compress = [exp_dir / "logs"]
        elif artifact_type == "config" and (exp_dir / "config").exists():
            paths_to_compress = [exp_dir / "config"]
        else:
            raise ValueError(
                f"Unknown or empty artifact download type: '{artifact_type}'"
            )

        # Compile zip file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for source_path in paths_to_compress:
                if source_path.is_file():
                    zipf.write(
                        source_path, arcname=str(source_path.relative_to(exp_dir))
                    )
                else:
                    for root, _, files in os.walk(source_path):
                        for file in files:
                            file_path = Path(root) / file
                            # Calculate path relative to experiment root to structure zip neatly
                            rel_path = file_path.relative_to(exp_dir)
                            zipf.write(file_path, arcname=str(rel_path))

        logger.info("Generated zip archive for %s: %s", experiment_id, zip_path)
        return str(zip_path)
