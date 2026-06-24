"""Portfolio validation rules for Module 1.

Validates individual building profiles and the complete 21-building
portfolio against the hard rules defined in the design specification.

Design reference: docs/module_1_profile_generator.md §Validation Rules
"""

from __future__ import annotations

import logging

import pandas as pd

from p2p_energy_trading.constants import (
    INTERNAL_COL_DEMAND,
    INTERNAL_COL_SOLAR,
    INTERNAL_COL_TIMESTAMP,
    NUM_COLLEGE,
    NUM_CONSUMER,
    NUM_SOLAR,
)
from p2p_energy_trading.exceptions import ProfileGenerationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected portfolio composition
# ---------------------------------------------------------------------------

EXPECTED_TOTAL: int = NUM_COLLEGE + NUM_SOLAR + NUM_CONSUMER  # 21
EXPECTED_SOLAR: int = NUM_SOLAR   # 15
EXPECTED_CONSUMER: int = NUM_CONSUMER  # 5
EXPECTED_COLLEGE: int = NUM_COLLEGE  # 1


def validate_profile(df: pd.DataFrame, building_id: str) -> None:
    """Validate a single building profile against hard rules.

    Hard rules (module_1_profile_generator.md §Validation Rules):
    - No negative demand values
    - No negative solar generation values
    - Timestamp continuity (hourly, no gaps, no duplicates)
    - No missing values in any column

    Args:
        df: DataFrame with internal column names (timestamp, demand_kw,
            solar_generation_kw).
        building_id: Identifier used in error messages.

    Raises:
        ProfileGenerationError: If any hard rule is violated.
    """
    _validate_required_columns(df, building_id)
    _validate_no_missing_values(df, building_id)
    _validate_no_negative_demand(df, building_id)
    _validate_no_negative_solar(df, building_id)
    _validate_timestamp_continuity(df, building_id)
    logger.debug("Profile validated: %s (%d samples)", building_id, len(df))


def validate_portfolio(
    profiles: dict[str, pd.DataFrame],
) -> None:
    """Validate the complete 21-building portfolio.

    Checks:
    - Exactly 21 buildings total
    - Exactly 15 solar buildings
    - Exactly 5 consumer buildings
    - Exactly 1 college building
    - Each individual profile passes hard rules

    Args:
        profiles: Mapping of building_id to its validated DataFrame.
            Keys follow naming: 'college', 'solar_01'…'solar_15',
            'consumer_01'…'consumer_05'.

    Raises:
        ProfileGenerationError: If portfolio composition is wrong or any
            individual profile fails hard rules.
    """
    _validate_portfolio_composition(profiles)

    for building_id, df in profiles.items():
        validate_profile(df, building_id)

    logger.info(
        "Portfolio validated: %d buildings (%d solar, %d consumer, %d college)",
        len(profiles),
        sum(1 for k in profiles if k.startswith("solar_")),
        sum(1 for k in profiles if k.startswith("consumer_")),
        sum(1 for k in profiles if k == "college"),
    )


# ---------------------------------------------------------------------------
# Private validators
# ---------------------------------------------------------------------------

def _validate_required_columns(df: pd.DataFrame, building_id: str) -> None:
    """Check that all required columns are present."""
    required = [INTERNAL_COL_TIMESTAMP, INTERNAL_COL_DEMAND, INTERNAL_COL_SOLAR]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ProfileGenerationError(
            f"Building '{building_id}': missing required columns {missing}. "
            f"Available: {list(df.columns)}"
        )


def _validate_no_missing_values(df: pd.DataFrame, building_id: str) -> None:
    """Check that no column has missing values."""
    for col in [INTERNAL_COL_TIMESTAMP, INTERNAL_COL_DEMAND, INTERNAL_COL_SOLAR]:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                raise ProfileGenerationError(
                    f"Building '{building_id}': column '{col}' has "
                    f"{n_missing} missing values"
                )


def _validate_no_negative_demand(df: pd.DataFrame, building_id: str) -> None:
    """Check that demand_kw has no negative values (hard rule)."""
    if INTERNAL_COL_DEMAND not in df.columns:
        return
    n_neg = (df[INTERNAL_COL_DEMAND] < 0).sum()
    if n_neg > 0:
        min_val = df[INTERNAL_COL_DEMAND].min()
        raise ProfileGenerationError(
            f"Building '{building_id}': {n_neg} negative demand values "
            f"(min={min_val:.4f} kW). Hard rule violation."
        )


def _validate_no_negative_solar(df: pd.DataFrame, building_id: str) -> None:
    """Check that solar_generation_kw has no negative values (hard rule)."""
    if INTERNAL_COL_SOLAR not in df.columns:
        return
    n_neg = (df[INTERNAL_COL_SOLAR] < 0).sum()
    if n_neg > 0:
        min_val = df[INTERNAL_COL_SOLAR].min()
        raise ProfileGenerationError(
            f"Building '{building_id}': {n_neg} negative solar values "
            f"(min={min_val:.4f} kW). Hard rule violation."
        )


def _validate_timestamp_continuity(df: pd.DataFrame, building_id: str) -> None:
    """Check that timestamps are hourly, continuous, and duplicate-free."""
    if INTERNAL_COL_TIMESTAMP not in df.columns or len(df) < 2:
        return

    ts = pd.to_datetime(df[INTERNAL_COL_TIMESTAMP])
    expected_freq = pd.Timedelta(hours=1)

    time_diffs = ts.diff().dropna()
    non_hourly = time_diffs[time_diffs != expected_freq]
    if len(non_hourly) > 0:
        raise ProfileGenerationError(
            f"Building '{building_id}': timestamp continuity broken — "
            f"{len(non_hourly)} non-hourly gaps. "
            f"First gap at index {non_hourly.index[0]}: {non_hourly.iloc[0]}"
        )

    n_dups = ts.duplicated().sum()
    if n_dups > 0:
        raise ProfileGenerationError(
            f"Building '{building_id}': {n_dups} duplicate timestamps"
        )


def _validate_portfolio_composition(profiles: dict[str, pd.DataFrame]) -> None:
    """Check that the portfolio has exactly the required building counts."""
    solar_ids = [k for k in profiles if k.startswith("solar_")]
    consumer_ids = [k for k in profiles if k.startswith("consumer_")]
    college_ids = [k for k in profiles if k == "college"]

    total = len(profiles)
    n_solar = len(solar_ids)
    n_consumer = len(consumer_ids)
    n_college = len(college_ids)

    errors: list[str] = []

    if total != EXPECTED_TOTAL:
        errors.append(
            f"Expected {EXPECTED_TOTAL} buildings total, got {total}"
        )
    if n_solar != EXPECTED_SOLAR:
        errors.append(
            f"Expected {EXPECTED_SOLAR} solar buildings, got {n_solar}"
        )
    if n_consumer != EXPECTED_CONSUMER:
        errors.append(
            f"Expected {EXPECTED_CONSUMER} consumer buildings, got {n_consumer}"
        )
    if n_college != EXPECTED_COLLEGE:
        errors.append(
            f"Expected {EXPECTED_COLLEGE} college building, got {n_college}"
        )

    if errors:
        raise ProfileGenerationError(
            "Portfolio composition validation failed: " + "; ".join(errors)
        )
