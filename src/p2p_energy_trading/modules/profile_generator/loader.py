"""Raw CSV loading and validation for Module 1.

Loads the complete KL S VDIT hourly market dataset and validates
required columns and data quality.

Design reference: docs/module_1_profile_generator.md
"""

from __future__ import annotations

# standard library
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
    RAW_CSV_COLUMN_DEMAND,
    RAW_CSV_COLUMN_SOLAR,
    RAW_CSV_COLUMN_TIMESTAMP,
    RAW_DATA_FILENAME,
)
from p2p_energy_trading.exceptions import ProfileGenerationError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    RAW_CSV_COLUMN_TIMESTAMP,
    RAW_CSV_COLUMN_DEMAND,
    RAW_CSV_COLUMN_SOLAR,
]

EXPECTED_DTYPES = {
    RAW_CSV_COLUMN_TIMESTAMP: "object",
    RAW_CSV_COLUMN_DEMAND: "float64",
    RAW_CSV_COLUMN_SOLAR: "float64",
}


def load_raw_data(
    data_dir: str | Path = "data/raw",
    filename: str = RAW_DATA_FILENAME,
) -> pd.DataFrame:
    """Load and validate the raw KL S VDIT hourly market CSV.

    Loads the complete dataset (all years) without truncation.
    Validates required columns, data types, and continuity.

    Args:
        data_dir: Directory containing the raw CSV file.
        filename: Name of the CSV file to load.

    Returns:
        DataFrame with validated data and internal column names.

    Raises:
        ProfileGenerationError: If file not found, columns missing,
            or data validation fails.
    """
    file_path = Path(data_dir) / filename

    if not file_path.exists():
        raise ProfileGenerationError(
            f"Raw data file not found: {file_path}. "
            f"Please place {filename} in {data_dir}/"
        )

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ProfileGenerationError(f"Failed to read CSV {file_path}: {e}") from e

    logger.info("Loaded raw CSV: %s (%d rows)", file_path, len(df))

    _validate_columns(df)
    _validate_dtypes(df)
    _validate_no_missing(df)
    _validate_no_negative(df)

    df = df.rename(
        columns={
            RAW_CSV_COLUMN_TIMESTAMP: INTERNAL_COL_TIMESTAMP,
            RAW_CSV_COLUMN_DEMAND: INTERNAL_COL_DEMAND,
            RAW_CSV_COLUMN_SOLAR: INTERNAL_COL_SOLAR,
        }
    )

    df[INTERNAL_COL_TIMESTAMP] = pd.to_datetime(df[INTERNAL_COL_TIMESTAMP])
    df = df.sort_values(INTERNAL_COL_TIMESTAMP).reset_index(drop=True)

    _validate_timestamp_continuity(df)

    logger.info(
        "Raw data validated: %d samples, %s to %s",
        len(df),
        df[INTERNAL_COL_TIMESTAMP].min().date(),
        df[INTERNAL_COL_TIMESTAMP].max().date(),
    )

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Validate that all required columns are present."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ProfileGenerationError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _validate_dtypes(df: pd.DataFrame) -> None:
    """Validate that columns have expected data types."""
    for col, expected_dtype in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        actual_dtype = str(df[col].dtype)
        if expected_dtype not in actual_dtype and not (
            expected_dtype == "float64" and "float" in actual_dtype
        ):
            raise ProfileGenerationError(
                f"Column '{col}' has dtype '{actual_dtype}',"
                f" expected '{expected_dtype}'"
            )


def _validate_no_missing(df: pd.DataFrame) -> None:
    """Validate that required columns have no missing values."""
    for col in REQUIRED_COLUMNS:
        if col in df.columns and df[col].isna().any():
            missing_count = df[col].isna().sum()
            raise ProfileGenerationError(
                f"Column '{col}' has {missing_count} missing values"
            )


def _validate_no_negative(df: pd.DataFrame) -> None:
    """Validate that demand and solar are non-negative."""
    for col in [RAW_CSV_COLUMN_DEMAND, RAW_CSV_COLUMN_SOLAR]:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                raise ProfileGenerationError(
                    f"Column '{col}' has {negative_count} negative values"
                )


def _validate_timestamp_continuity(df: pd.DataFrame) -> None:
    """Validate that timestamps are hourly and continuous."""
    if INTERNAL_COL_TIMESTAMP not in df.columns:
        return

    time_diffs = df[INTERNAL_COL_TIMESTAMP].diff().dropna()
    expected_freq = pd.Timedelta(hours=1)

    non_hourly = time_diffs[time_diffs != expected_freq]
    if len(non_hourly) > 0:
        raise ProfileGenerationError(
            f"Timestamp continuity broken: {len(non_hourly)} gaps found. "
            f"Expected hourly frequency. First gap: {non_hourly.iloc[0]}"
        )

    duplicate_count = df[INTERNAL_COL_TIMESTAMP].duplicated().sum()
    if duplicate_count > 0:
        raise ProfileGenerationError(
            f"Timestamp column has {duplicate_count} duplicate values"
        )


def get_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate summary statistics for the loaded data.

    Args:
        df: Validated DataFrame with internal column names.

    Returns:
        Dictionary with summary statistics.
    """
    return {
        "total_samples": len(df),
        "date_range": {
            "start": df[INTERNAL_COL_TIMESTAMP].min().isoformat(),
            "end": df[INTERNAL_COL_TIMESTAMP].max().isoformat(),
        },
        "demand_kw": {
            "min": float(df[INTERNAL_COL_DEMAND].min()),
            "max": float(df[INTERNAL_COL_DEMAND].max()),
            "mean": float(df[INTERNAL_COL_DEMAND].mean()),
            "std": float(df[INTERNAL_COL_DEMAND].std()),
        },
        "solar_generation_kw": {
            "min": float(df[INTERNAL_COL_SOLAR].min()),
            "max": float(df[INTERNAL_COL_SOLAR].max()),
            "mean": float(df[INTERNAL_COL_SOLAR].mean()),
            "std": float(df[INTERNAL_COL_SOLAR].std()),
        },
        "years_covered": sorted(df[INTERNAL_COL_TIMESTAMP].dt.year.unique().tolist()),
    }
