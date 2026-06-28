"""Metadata generation for Module 1.

Generates metadata.json describing the complete profile portfolio:
peak values, scaling factors, date ranges, and generation parameters
for each of the 21 buildings.

Design reference: docs/module_1_profile_generator.md §Output Structure
"""

from __future__ import annotations

# standard library
import json
import logging
from pathlib import Path
from typing import Any

# third party
import pandas as pd

# local
from p2p_energy_trading.constants import (
    INTERNAL_COL_DEMAND,
    INTERNAL_COL_SOLAR,
    INTERNAL_COL_TIMESTAMP,
)
from p2p_energy_trading.exceptions import ProfileGenerationError
from p2p_energy_trading.modules.profile_generator.generator import BuildingProfile

logger = logging.getLogger(__name__)

METADATA_FILENAME: str = "metadata.json"


def build_metadata(
    profiles: list[BuildingProfile],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build the metadata dictionary for the complete profile portfolio.

    Computes per-building statistics (peak demand, peak solar, mean demand,
    total solar energy, date range, row count) and records the generation
    parameters (load_scale, solar_scale, shift_hours, seed) for each building.

    Args:
        profiles: List of all 21 BuildingProfile objects (college + solar + consumer).
        output_dir: Directory where metadata.json will be written.

    Returns:
        Metadata dictionary. Also written to output_dir/metadata.json.

    Raises:
        ProfileGenerationError: If a profile DataFrame is missing required
            columns or if the output directory cannot be created.
    """
    output_path = Path(output_dir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise ProfileGenerationError(
            f"Cannot create output directory '{output_path}': {e}"
        ) from e

    buildings: dict[str, Any] = {}
    for profile in profiles:
        df = profile.df
        _check_profile_df(df, profile.building_id)

        demand = df[INTERNAL_COL_DEMAND]
        solar = df[INTERNAL_COL_SOLAR]
        ts = pd.to_datetime(df[INTERNAL_COL_TIMESTAMP])

        buildings[profile.building_id] = {
            "building_id": profile.building_id,
            "building_type": profile.building_type,
            "generation_params": {
                "load_scale": round(profile.load_scale, 6),
                "solar_scale": round(profile.solar_scale, 6),
                "shift_hours": profile.shift_hours,
                "seed": profile.seed,
            },
            "profile_stats": {
                "num_samples": int(len(df)),
                "date_range": {
                    "start": ts.min().isoformat(),
                    "end": ts.max().isoformat(),
                },
                "demand_kw": {
                    "peak": float(demand.max()),
                    "mean": float(demand.mean()),
                    "min": float(demand.min()),
                },
                "solar_generation_kw": {
                    "peak": float(solar.max()),
                    "mean": float(solar.mean()),
                    "total_energy_kwh": float(solar.sum()),  # hourly data: sum = kWh
                },
            },
        }

    # Portfolio-level summary
    total_buildings = len(profiles)
    solar_count = sum(1 for p in profiles if p.building_type == "solar")
    consumer_count = sum(1 for p in profiles if p.building_type == "consumer")
    college_count = sum(1 for p in profiles if p.building_type == "college")

    metadata: dict[str, Any] = {
        "version": "1.0",
        "portfolio_summary": {
            "total_buildings": total_buildings,
            "college_buildings": college_count,
            "solar_buildings": solar_count,
            "consumer_buildings": consumer_count,
        },
        "buildings": buildings,
    }

    # Write to disk
    metadata_file = output_path / METADATA_FILENAME
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except OSError as e:
        raise ProfileGenerationError(
            f"Failed to write metadata to '{metadata_file}': {e}"
        ) from e

    logger.info("Metadata written: %s (%d buildings)", metadata_file, total_buildings)
    return metadata


def load_metadata(output_dir: str | Path) -> dict[str, Any]:
    """Load previously generated metadata.json from the output directory.

    Args:
        output_dir: Directory containing metadata.json.

    Returns:
        Metadata dictionary.

    Raises:
        ProfileGenerationError: If metadata.json does not exist or is malformed.
    """
    metadata_file = Path(output_dir) / METADATA_FILENAME
    if not metadata_file.exists():
        raise ProfileGenerationError(
            f"metadata.json not found at '{metadata_file}'. "
            "Run the profile generator first."
        )

    try:
        with open(metadata_file, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ProfileGenerationError(
            f"Failed to load metadata from '{metadata_file}': {e}"
        ) from e


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_profile_df(df: pd.DataFrame, building_id: str) -> None:
    """Check that the profile DataFrame has the required columns."""
    required = [INTERNAL_COL_TIMESTAMP, INTERNAL_COL_DEMAND, INTERNAL_COL_SOLAR]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ProfileGenerationError(
            f"Building '{building_id}' profile is missing columns: {missing}"
        )
