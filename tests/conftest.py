"""Shared test fixtures for the P2P Energy Trading System test suite.

Provides reusable pytest fixtures for unit and integration tests.
All fixtures use fixed seeds for deterministic results.

Reference: docs/module_12_repository_structure.md §7 (Testing Standards)
"""

from __future__ import annotations

# third party
import numpy as np
import pandas as pd
import pytest

# local
from p2p_energy_trading.constants import (
    DEFAULT_SEED,
    INTERNAL_COL_DEMAND,
    INTERNAL_COL_SOLAR,
    INTERNAL_COL_TIMESTAMP,
)

# ---------------------------------------------------------------------------
# Seed configuration
# ---------------------------------------------------------------------------

FIXTURE_SEED: int = DEFAULT_SEED  # 42


# ---------------------------------------------------------------------------
# Raw data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_timestamps() -> pd.DatetimeIndex:
    """Generate 8760 hourly timestamps for one full year (2023).

    Returns:
        DatetimeIndex with 8760 hourly timestamps starting 2023-01-01 00:00.
    """
    return pd.date_range(start="2023-01-01", periods=8760, freq="h")


@pytest.fixture
def minimal_timestamps() -> pd.DatetimeIndex:
    """Generate 168 hourly timestamps (one week) for fast unit tests.

    Returns:
        DatetimeIndex with 168 hourly timestamps starting 2023-01-01 00:00.
    """
    return pd.date_range(start="2023-01-01", periods=168, freq="h")


@pytest.fixture
def college_raw_df(sample_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Minimal validated college DataFrame using internal column names.

    Generates a synthetic but physically plausible dataset:
    - demand_kw: daily sinusoidal pattern (50-200 kW range)
    - solar_generation_kw: midday peak (0-100 kW range)

    Args:
        sample_timestamps: Fixture providing 8760 hourly timestamps.

    Returns:
        DataFrame with columns: timestamp, demand_kw, solar_generation_kw.
    """
    rng = np.random.default_rng(FIXTURE_SEED)
    n = len(sample_timestamps)

    # Diurnal demand pattern: peaks around hour 14
    hours = np.arange(n) % 24
    demand_base = 100.0 + 80.0 * np.sin(np.pi * (hours - 6) / 12)
    demand_base = np.clip(demand_base, 20.0, None)
    demand_noise = rng.normal(0.0, 5.0, n)
    demand = np.clip(demand_base + demand_noise, 0.0, None)

    # Solar: only during daylight hours (6-18)
    solar = np.where(
        (hours >= 6) & (hours <= 18),
        60.0 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0.0, 3.0, n),
        0.0,
    )
    solar = np.clip(solar, 0.0, None)

    return pd.DataFrame(
        {
            INTERNAL_COL_TIMESTAMP: sample_timestamps,
            INTERNAL_COL_DEMAND: demand.astype(np.float64),
            INTERNAL_COL_SOLAR: solar.astype(np.float64),
        }
    )


@pytest.fixture
def small_college_df(minimal_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Small (168-sample) college DataFrame for fast unit tests.

    Args:
        minimal_timestamps: Fixture providing 168 hourly timestamps.

    Returns:
        DataFrame with columns: timestamp, demand_kw, solar_generation_kw.
    """
    rng = np.random.default_rng(FIXTURE_SEED)
    n = len(minimal_timestamps)

    hours = np.arange(n) % 24
    demand = np.clip(
        100.0 + 50.0 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0.0, 3.0, n),
        0.0,
        None,
    )
    solar = np.where(
        (hours >= 6) & (hours <= 18),
        np.clip(
            40.0 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0.0, 2.0, n), 0.0, None
        ),
        0.0,
    )

    return pd.DataFrame(
        {
            INTERNAL_COL_TIMESTAMP: minimal_timestamps,
            INTERNAL_COL_DEMAND: demand.astype(np.float64),
            INTERNAL_COL_SOLAR: solar.astype(np.float64),
        }
    )


@pytest.fixture
def raw_csv_df(sample_timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic DataFrame in raw CSV column format (before renaming).

    Mimics the structure of kls_vdit_hourly_market.csv including
    the raw column names. Used for testing loader.py.

    Args:
        sample_timestamps: Fixture providing 8760 hourly timestamps.

    Returns:
        DataFrame with columns: Timestamp, Campus_Demand_kW, College_Solar_kW.
    """
    # local
    from p2p_energy_trading.constants import (
        RAW_CSV_COLUMN_DEMAND,
        RAW_CSV_COLUMN_SOLAR,
        RAW_CSV_COLUMN_TIMESTAMP,
    )

    rng = np.random.default_rng(FIXTURE_SEED)
    n = len(sample_timestamps)
    hours = np.arange(n) % 24

    demand = np.clip(
        100.0 + 80.0 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0.0, 5.0, n),
        0.0,
        None,
    )
    solar = np.where(
        (hours >= 6) & (hours <= 18),
        np.clip(
            60.0 * np.sin(np.pi * (hours - 6) / 12) + rng.normal(0.0, 3.0, n), 0.0, None
        ),
        0.0,
    )

    return pd.DataFrame(
        {
            RAW_CSV_COLUMN_TIMESTAMP: sample_timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            RAW_CSV_COLUMN_DEMAND: demand.astype(np.float64),
            RAW_CSV_COLUMN_SOLAR: solar.astype(np.float64),
        }
    )


# ---------------------------------------------------------------------------
# Profile portfolio fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_portfolio(small_college_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Complete 21-building portfolio using 168-sample profiles.

    Generates a minimal valid portfolio for fast portfolio-level tests.
    Uses fixed seed for determinism.

    Args:
        small_college_df: 168-sample college DataFrame.

    Returns:
        Dict mapping building_id to DataFrame (21 entries).
    """
    # local
    from p2p_energy_trading.modules.profile_generator.generator import (
        generate_college_profile,
        generate_consumer_profiles,
        generate_solar_profiles,
    )

    college = generate_college_profile(small_college_df)
    solar = generate_solar_profiles(small_college_df, base_seed=FIXTURE_SEED)
    consumer = generate_consumer_profiles(
        small_college_df, base_seed=FIXTURE_SEED + 100
    )

    return {p.building_id: p.df for p in [college] + solar + consumer}
