"""Unit tests for Module 1 — Profile Generator.

Tests cover:
- Loader: column validation, dtype validation, missing/negative data,
  timestamp continuity
- Generator: load scaling, solar scaling, temporal shifting,
  solar generation, weekend effects
- Validator: hard rules (no negatives, continuity), portfolio composition
- Metadata: build_metadata output structure
- Exception hierarchy: ProfileGenerationError inherits from P2PEnergyTradingError

All tests are deterministic (fixed seeds), fast (<1s each), require no GPU, no Ray.

Design reference: docs/module_1_profile_generator.md §Testing
Reference: docs/module_12_repository_structure.md §7
"""

from __future__ import annotations

# third party
import numpy as np
import pandas as pd
import pytest

# local
from p2p_energy_trading.constants import (
    INTERNAL_COL_DEMAND,
    INTERNAL_COL_SOLAR,
    INTERNAL_COL_TIMESTAMP,
    NUM_CONSUMER,
    NUM_SOLAR,
    RAW_CSV_COLUMN_DEMAND,
    RAW_CSV_COLUMN_SOLAR,
    RAW_CSV_COLUMN_TIMESTAMP,
)
from p2p_energy_trading.exceptions import P2PEnergyTradingError, ProfileGenerationError

# ===========================================================================
# Section 1: Exception Hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    """Verify that ProfileGenerationError is in the approved hierarchy."""

    def test_profile_generation_error_is_p2p_error(self) -> None:
        """ProfileGenerationError must inherit from P2PEnergyTradingError."""
        err = ProfileGenerationError("test")
        assert isinstance(err, P2PEnergyTradingError), (
            "ProfileGenerationError must inherit from P2PEnergyTradingError"
        )

    def test_profile_generation_error_is_exception(self) -> None:
        """ProfileGenerationError must be catchable as a base Exception."""
        err = ProfileGenerationError("test")
        assert isinstance(err, Exception)

    def test_profile_generation_error_message(self) -> None:
        """ProfileGenerationError must preserve the message string."""
        msg = "Something went wrong"
        err = ProfileGenerationError(msg)
        assert str(err) == msg


# ===========================================================================
# Section 2: Loader — _validate_columns
# ===========================================================================


class TestLoaderColumnValidation:
    """Tests for loader._validate_columns."""

    def test_valid_columns_pass(self, raw_csv_df: pd.DataFrame) -> None:
        """DataFrame with all required columns should not raise."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_columns,
        )

        _validate_columns(raw_csv_df)  # should not raise

    def test_missing_timestamp_column_raises(self, raw_csv_df: pd.DataFrame) -> None:
        """Missing Timestamp column must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_columns,
        )

        df = raw_csv_df.drop(columns=[RAW_CSV_COLUMN_TIMESTAMP])
        with pytest.raises(ProfileGenerationError, match="Missing required columns"):
            _validate_columns(df)

    def test_missing_demand_column_raises(self, raw_csv_df: pd.DataFrame) -> None:
        """Missing Campus_Demand_kW column must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_columns,
        )

        df = raw_csv_df.drop(columns=[RAW_CSV_COLUMN_DEMAND])
        with pytest.raises(ProfileGenerationError, match="Missing required columns"):
            _validate_columns(df)

    def test_missing_solar_column_raises(self, raw_csv_df: pd.DataFrame) -> None:
        """Missing College_Solar_kW column must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_columns,
        )

        df = raw_csv_df.drop(columns=[RAW_CSV_COLUMN_SOLAR])
        with pytest.raises(ProfileGenerationError, match="Missing required columns"):
            _validate_columns(df)

    def test_extra_columns_pass(self, raw_csv_df: pd.DataFrame) -> None:
        """Extra columns should not cause validation to fail."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_columns,
        )

        df = raw_csv_df.copy()
        df["Neighbor_Hotel_kW"] = 0.0
        _validate_columns(df)  # should not raise


# ===========================================================================
# Section 3: Loader — _validate_no_negative
# ===========================================================================


class TestLoaderNegativeValidation:
    """Tests for loader._validate_no_negative."""

    def test_non_negative_data_passes(self, raw_csv_df: pd.DataFrame) -> None:
        """All-positive demand and solar should not raise."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_no_negative,
        )

        _validate_no_negative(raw_csv_df)  # should not raise

    def test_negative_demand_raises(self, raw_csv_df: pd.DataFrame) -> None:
        """Negative demand values must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_no_negative,
        )

        df = raw_csv_df.copy()
        df.loc[0, RAW_CSV_COLUMN_DEMAND] = -5.0
        with pytest.raises(ProfileGenerationError, match="negative values"):
            _validate_no_negative(df)

    def test_negative_solar_raises(self, raw_csv_df: pd.DataFrame) -> None:
        """Negative solar values must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_no_negative,
        )

        df = raw_csv_df.copy()
        df.loc[10, RAW_CSV_COLUMN_SOLAR] = -1.0
        with pytest.raises(ProfileGenerationError, match="negative values"):
            _validate_no_negative(df)

    def test_zero_values_pass(self, raw_csv_df: pd.DataFrame) -> None:
        """Zero demand or solar values are valid (not negative)."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_no_negative,
        )

        df = raw_csv_df.copy()
        df.loc[0, RAW_CSV_COLUMN_DEMAND] = 0.0
        df.loc[1, RAW_CSV_COLUMN_SOLAR] = 0.0
        _validate_no_negative(df)  # should not raise


# ===========================================================================
# Section 4: Loader — _validate_no_missing
# ===========================================================================


class TestLoaderMissingValidation:
    """Tests for loader._validate_no_missing."""

    def test_complete_data_passes(self, raw_csv_df: pd.DataFrame) -> None:
        """DataFrame with no NaNs should not raise."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_no_missing,
        )

        _validate_no_missing(raw_csv_df)  # should not raise

    def test_nan_demand_raises(self, raw_csv_df: pd.DataFrame) -> None:
        """NaN in demand column must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_no_missing,
        )

        df = raw_csv_df.copy()
        df.loc[5, RAW_CSV_COLUMN_DEMAND] = np.nan
        with pytest.raises(ProfileGenerationError, match="missing values"):
            _validate_no_missing(df)

    def test_nan_solar_raises(self, raw_csv_df: pd.DataFrame) -> None:
        """NaN in solar column must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_no_missing,
        )

        df = raw_csv_df.copy()
        df.loc[3, RAW_CSV_COLUMN_SOLAR] = np.nan
        with pytest.raises(ProfileGenerationError, match="missing values"):
            _validate_no_missing(df)


# ===========================================================================
# Section 5: Loader — _validate_timestamp_continuity
# ===========================================================================


class TestLoaderTimestampContinuity:
    """Tests for loader._validate_timestamp_continuity."""

    def test_hourly_continuous_timestamps_pass(
        self, college_raw_df: pd.DataFrame
    ) -> None:
        """Perfectly hourly timestamps should not raise."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_timestamp_continuity,
        )

        _validate_timestamp_continuity(college_raw_df)  # should not raise

    def test_gap_in_timestamps_raises(self, college_raw_df: pd.DataFrame) -> None:
        """A gap in the timestamp sequence must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_timestamp_continuity,
        )

        df = college_raw_df.copy()
        # Drop row 5 to create a 2-hour gap
        df = df.drop(index=5).reset_index(drop=True)
        with pytest.raises(ProfileGenerationError, match="continuity broken"):
            _validate_timestamp_continuity(df)

    def test_duplicate_timestamps_raises(self, college_raw_df: pd.DataFrame) -> None:
        """Duplicate timestamps must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            _validate_timestamp_continuity,
        )

        df = college_raw_df.copy()
        duplicate_row = df.iloc[[0]].copy()
        df = pd.concat([df, duplicate_row], ignore_index=True)
        df = df.sort_values(INTERNAL_COL_TIMESTAMP).reset_index(drop=True)
        with pytest.raises(ProfileGenerationError):
            _validate_timestamp_continuity(df)


# ===========================================================================
# Section 6: Loader — load_raw_data (file not found)
# ===========================================================================


class TestLoaderFileHandling:
    """Tests for loader.load_raw_data file-level handling."""

    def test_missing_file_raises(self, tmp_path: object) -> None:
        """Missing CSV file must raise ProfileGenerationError with clear message."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import load_raw_data

        with pytest.raises(ProfileGenerationError, match="not found"):
            load_raw_data(data_dir=str(tmp_path), filename="nonexistent.csv")

    def test_load_raw_data_returns_internal_column_names(
        self, tmp_path: object, raw_csv_df: pd.DataFrame
    ) -> None:
        """load_raw_data must rename columns to internal names."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            RAW_DATA_FILENAME,
            load_raw_data,
        )

        csv_path = str(tmp_path) + f"/{RAW_DATA_FILENAME}"
        raw_csv_df.to_csv(csv_path, index=False)

        result = load_raw_data(data_dir=str(tmp_path), filename=RAW_DATA_FILENAME)

        assert INTERNAL_COL_TIMESTAMP in result.columns
        assert INTERNAL_COL_DEMAND in result.columns
        assert INTERNAL_COL_SOLAR in result.columns

    def test_load_raw_data_preserves_all_rows(
        self, tmp_path: object, raw_csv_df: pd.DataFrame
    ) -> None:
        """load_raw_data must load the complete dataset without truncation."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import (
            RAW_DATA_FILENAME,
            load_raw_data,
        )

        csv_path = str(tmp_path) + f"/{RAW_DATA_FILENAME}"
        raw_csv_df.to_csv(csv_path, index=False)

        result = load_raw_data(data_dir=str(tmp_path), filename=RAW_DATA_FILENAME)
        assert len(result) == len(raw_csv_df)


# ===========================================================================
# Section 7: Generator — load scaling
# ===========================================================================


class TestGeneratorLoadScaling:
    """Tests for generator profile scaling factors."""

    def test_solar_load_scale_within_bounds(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Solar profile demand must be 30%-80% of college peak demand."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            SOLAR_LOAD_SCALE_MAX,
            SOLAR_LOAD_SCALE_MIN,
            generate_solar_profiles,
        )

        profiles = generate_solar_profiles(small_college_df, base_seed=42)
        college_peak = small_college_df[INTERNAL_COL_DEMAND].max()

        for p in profiles:
            assert SOLAR_LOAD_SCALE_MIN <= p.load_scale <= SOLAR_LOAD_SCALE_MAX, (
                f"{p.building_id}: load_scale={p.load_scale:.3f} out of bounds"
            )
            solar_peak = p.df[INTERNAL_COL_DEMAND].max()
            # Peak demand should be approximately within expected range
            # (weekend/occupancy effects may push it slightly below)
            assert solar_peak <= college_peak * SOLAR_LOAD_SCALE_MAX * 1.40, (
                f"{p.building_id}: peak demand {solar_peak:.1f} exceeds expected bound"
            )

    def test_consumer_load_scale_within_bounds(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Consumer profile demand must be 10%-30% of college peak demand."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            CONSUMER_LOAD_SCALE_MAX,
            CONSUMER_LOAD_SCALE_MIN,
            generate_consumer_profiles,
        )

        profiles = generate_consumer_profiles(small_college_df, base_seed=100)

        for p in profiles:
            assert CONSUMER_LOAD_SCALE_MIN <= p.load_scale <= CONSUMER_LOAD_SCALE_MAX, (
                f"{p.building_id}: load_scale={p.load_scale:.3f} out of bounds"
            )

    def test_solar_profiles_have_different_scales(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Different solar buildings must have different (non-identical) scales."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_solar_profiles,
        )

        profiles = generate_solar_profiles(small_college_df, base_seed=42)
        scales = [p.load_scale for p in profiles]
        # At minimum 5 distinct values among 15 profiles
        assert len(set(round(s, 4) for s in scales)) >= 5


# ===========================================================================
# Section 8: Generator — solar generation
# ===========================================================================


class TestGeneratorSolarGeneration:
    """Tests for solar generation values in profiles."""

    def test_solar_profiles_have_nonzero_solar(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Solar buildings must have positive solar generation (inherited from
        college)."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_solar_profiles,
        )

        profiles = generate_solar_profiles(small_college_df, base_seed=42)
        for p in profiles:
            assert p.df[INTERNAL_COL_SOLAR].max() > 0.0, (
                f"{p.building_id}: solar generation is all zero"
            )

    def test_consumer_profiles_have_zero_solar(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Consumer buildings must have zero solar generation throughout."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_consumer_profiles,
        )

        profiles = generate_consumer_profiles(small_college_df, base_seed=100)
        for p in profiles:
            assert p.df[INTERNAL_COL_SOLAR].sum() == 0.0, (
                f"{p.building_id}: solar_generation_kw should be all zero"
            )
            assert p.solar_scale == 0.0

    def test_solar_generation_nonnegative(self, small_college_df: pd.DataFrame) -> None:
        """Solar generation must never be negative (hard rule)."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_solar_profiles,
        )

        profiles = generate_solar_profiles(small_college_df, base_seed=42)
        for p in profiles:
            assert (p.df[INTERNAL_COL_SOLAR] >= 0).all(), (
                f"{p.building_id}: negative solar values found"
            )

    def test_demand_nonnegative_in_all_profiles(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Demand must never be negative in any generated profile."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_consumer_profiles,
            generate_solar_profiles,
        )

        for p in generate_solar_profiles(small_college_df, base_seed=42):
            assert (p.df[INTERNAL_COL_DEMAND] >= 0).all(), (
                f"{p.building_id}: negative demand values found"
            )
        for p in generate_consumer_profiles(small_college_df, base_seed=100):
            assert (p.df[INTERNAL_COL_DEMAND] >= 0).all(), (
                f"{p.building_id}: negative demand values found"
            )


# ===========================================================================
# Section 9: Generator — temporal shift
# ===========================================================================


class TestGeneratorTemporalShift:
    """Tests for temporal shift parameter in generated profiles."""

    def test_shift_hours_within_bounds(self, small_college_df: pd.DataFrame) -> None:
        """Temporal shift must be within +-3 hours (as specified)."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            MAX_SHIFT_HOURS,
            generate_solar_profiles,
        )

        profiles = generate_solar_profiles(small_college_df, base_seed=42)
        for p in profiles:
            assert -MAX_SHIFT_HOURS <= p.shift_hours <= MAX_SHIFT_HOURS, (
                f"{p.building_id}: shift_hours={p.shift_hours} out of bounds"
            )

    def test_profiles_are_not_identical(self, small_college_df: pd.DataFrame) -> None:
        """Different solar buildings must have different demand patterns."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_solar_profiles,
        )

        profiles = generate_solar_profiles(small_college_df, base_seed=42)
        demands = [p.df[INTERNAL_COL_DEMAND].values for p in profiles]
        # No two profiles should be exactly identical
        for i in range(len(demands)):
            for j in range(i + 1, len(demands)):
                assert not np.allclose(demands[i], demands[j], atol=1e-6), (
                    f"solar_{i + 1:02d} and solar_{j + 1:02d} are identical"
                )

    def test_timestamp_count_preserved_after_shift(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Circular shift must not change the number of samples."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_solar_profiles,
        )

        n_input = len(small_college_df)
        profiles = generate_solar_profiles(small_college_df, base_seed=42)
        for p in profiles:
            assert len(p.df) == n_input, (
                f"{p.building_id}: sample count changed from {n_input} to {len(p.df)}"
            )


# ===========================================================================
# Section 10: Generator — reproducibility
# ===========================================================================


class TestGeneratorReproducibility:
    """Tests that generation with the same seed produces identical results."""

    def test_solar_profiles_reproducible_with_fixed_seed(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Same seed must produce identical solar profiles."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_solar_profiles,
        )

        profiles_a = generate_solar_profiles(small_college_df, base_seed=42)
        profiles_b = generate_solar_profiles(small_college_df, base_seed=42)

        for a, b in zip(profiles_a, profiles_b, strict=True):
            assert a.building_id == b.building_id
            pd.testing.assert_frame_equal(a.df, b.df)

    def test_consumer_profiles_reproducible_with_fixed_seed(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Same seed must produce identical consumer profiles."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_consumer_profiles,
        )

        profiles_a = generate_consumer_profiles(small_college_df, base_seed=100)
        profiles_b = generate_consumer_profiles(small_college_df, base_seed=100)

        for a, b in zip(profiles_a, profiles_b, strict=True):
            assert a.building_id == b.building_id
            pd.testing.assert_frame_equal(a.df, b.df)

    def test_different_seeds_produce_different_profiles(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Different seeds must produce different profiles."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_solar_profiles,
        )

        profiles_a = generate_solar_profiles(small_college_df, base_seed=42)
        profiles_b = generate_solar_profiles(small_college_df, base_seed=99)

        # At least one profile should differ
        any_different = any(
            not np.allclose(
                a.df[INTERNAL_COL_DEMAND].values, b.df[INTERNAL_COL_DEMAND].values
            )
            for a, b in zip(profiles_a, profiles_b, strict=True)
        )
        assert any_different, "Different seeds produced identical profiles"


# ===========================================================================
# Section 11: Validator — single profile
# ===========================================================================


class TestValidatorSingleProfile:
    """Tests for validator.validate_profile."""

    def test_valid_profile_passes(self, small_college_df: pd.DataFrame) -> None:
        """A valid college profile should not raise."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_profile,
        )

        validate_profile(small_college_df, "college")  # should not raise

    def test_profile_with_negative_demand_raises(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Profile with negative demand must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_profile,
        )

        df = small_college_df.copy()
        df.loc[0, INTERNAL_COL_DEMAND] = -1.0
        with pytest.raises(ProfileGenerationError, match="negative demand"):
            validate_profile(df, "college")

    def test_profile_with_negative_solar_raises(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Profile with negative solar must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_profile,
        )

        df = small_college_df.copy()
        df.loc[5, INTERNAL_COL_SOLAR] = -0.5
        with pytest.raises(ProfileGenerationError, match="negative solar"):
            validate_profile(df, "solar_01")

    def test_profile_with_missing_values_raises(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Profile with NaN values must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_profile,
        )

        df = small_college_df.copy()
        df.loc[2, INTERNAL_COL_DEMAND] = np.nan
        with pytest.raises(ProfileGenerationError, match="missing values"):
            validate_profile(df, "college")

    def test_profile_with_missing_column_raises(
        self, small_college_df: pd.DataFrame
    ) -> None:
        """Profile missing a required column must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_profile,
        )

        df = small_college_df.drop(columns=[INTERNAL_COL_SOLAR])
        with pytest.raises(ProfileGenerationError, match="missing required columns"):
            validate_profile(df, "solar_01")


# ===========================================================================
# Section 12: Validator — portfolio composition
# ===========================================================================


class TestValidatorPortfolio:
    """Tests for validator.validate_portfolio."""

    def test_valid_portfolio_passes(self, minimal_portfolio: dict) -> None:
        """Complete 21-building portfolio must pass validation."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_portfolio,
        )

        validate_portfolio(minimal_portfolio)  # should not raise

    def test_portfolio_missing_college_raises(self, minimal_portfolio: dict) -> None:
        """Portfolio without college building must raise ProfileGenerationError."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_portfolio,
        )

        portfolio = {k: v for k, v in minimal_portfolio.items() if k != "college"}
        with pytest.raises(ProfileGenerationError, match="college"):
            validate_portfolio(portfolio)

    def test_portfolio_wrong_solar_count_raises(self, minimal_portfolio: dict) -> None:
        """Portfolio with fewer than 15 solar buildings must raise."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_portfolio,
        )

        portfolio = {k: v for k, v in minimal_portfolio.items() if k != "solar_15"}
        with pytest.raises(ProfileGenerationError, match="solar"):
            validate_portfolio(portfolio)

    def test_portfolio_wrong_consumer_count_raises(
        self, minimal_portfolio: dict
    ) -> None:
        """Portfolio with fewer than 5 consumer buildings must raise."""
        # local
        from p2p_energy_trading.modules.profile_generator.validator import (
            validate_portfolio,
        )

        portfolio = {k: v for k, v in minimal_portfolio.items() if k != "consumer_05"}
        with pytest.raises(ProfileGenerationError, match="consumer"):
            validate_portfolio(portfolio)

    def test_portfolio_has_correct_total_count(self, minimal_portfolio: dict) -> None:
        """Portfolio must contain exactly 21 buildings."""
        assert len(minimal_portfolio) == 21, (
            f"Expected 21 buildings, got {len(minimal_portfolio)}"
        )

    def test_portfolio_has_correct_solar_count(self, minimal_portfolio: dict) -> None:
        """Portfolio must contain exactly 15 solar buildings."""
        solar = [k for k in minimal_portfolio if k.startswith("solar_")]
        assert len(solar) == NUM_SOLAR, (
            f"Expected {NUM_SOLAR} solar buildings, got {len(solar)}"
        )

    def test_portfolio_has_correct_consumer_count(
        self, minimal_portfolio: dict
    ) -> None:
        """Portfolio must contain exactly 5 consumer buildings."""
        consumers = [k for k in minimal_portfolio if k.startswith("consumer_")]
        assert len(consumers) == NUM_CONSUMER, (
            f"Expected {NUM_CONSUMER} consumer buildings, got {len(consumers)}"
        )


# ===========================================================================
# Section 13: Metadata
# ===========================================================================


class TestMetadata:
    """Tests for metadata.build_metadata output structure."""

    def test_metadata_contains_all_buildings(
        self, minimal_portfolio: dict, tmp_path: object
    ) -> None:
        """Metadata must contain an entry for each building."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_college_profile,
            generate_consumer_profiles,
            generate_solar_profiles,
        )
        from p2p_energy_trading.modules.profile_generator.metadata import build_metadata

        _ = list(minimal_portfolio.values())[0]  # reuse fixture df shape
        # Build actual BuildingProfile objects
        college_df_real = pd.DataFrame(
            {
                INTERNAL_COL_TIMESTAMP: pd.date_range(
                    "2023-01-01", periods=168, freq="h"
                ),
                INTERNAL_COL_DEMAND: np.ones(168) * 100.0,
                INTERNAL_COL_SOLAR: np.ones(168) * 20.0,
            }
        )
        college_p = generate_college_profile(college_df_real)
        solar_ps = generate_solar_profiles(college_df_real, base_seed=42)
        consumer_ps = generate_consumer_profiles(college_df_real, base_seed=142)
        all_profiles = [college_p] + solar_ps + consumer_ps

        metadata = build_metadata(all_profiles, output_dir=str(tmp_path))

        assert "buildings" in metadata
        assert len(metadata["buildings"]) == 21

    def test_metadata_portfolio_summary(self, tmp_path: object) -> None:
        """Metadata portfolio_summary must report correct building counts."""
        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_college_profile,
            generate_consumer_profiles,
            generate_solar_profiles,
        )
        from p2p_energy_trading.modules.profile_generator.metadata import build_metadata

        df = pd.DataFrame(
            {
                INTERNAL_COL_TIMESTAMP: pd.date_range(
                    "2023-01-01", periods=168, freq="h"
                ),
                INTERNAL_COL_DEMAND: np.ones(168) * 100.0,
                INTERNAL_COL_SOLAR: np.ones(168) * 20.0,
            }
        )
        all_profiles = (
            [generate_college_profile(df)]
            + generate_solar_profiles(df, base_seed=42)
            + generate_consumer_profiles(df, base_seed=142)
        )

        metadata = build_metadata(all_profiles, output_dir=str(tmp_path))

        summary = metadata["portfolio_summary"]
        assert summary["total_buildings"] == 21
        assert summary["solar_buildings"] == 15
        assert summary["consumer_buildings"] == 5
        assert summary["college_buildings"] == 1

    def test_metadata_json_file_written(self, tmp_path: object) -> None:
        """build_metadata must write metadata.json to the output directory."""
        # standard library
        import json
        from pathlib import Path

        # local
        from p2p_energy_trading.modules.profile_generator.generator import (
            generate_college_profile,
            generate_consumer_profiles,
            generate_solar_profiles,
        )
        from p2p_energy_trading.modules.profile_generator.metadata import build_metadata

        df = pd.DataFrame(
            {
                INTERNAL_COL_TIMESTAMP: pd.date_range(
                    "2023-01-01", periods=168, freq="h"
                ),
                INTERNAL_COL_DEMAND: np.ones(168) * 100.0,
                INTERNAL_COL_SOLAR: np.ones(168) * 20.0,
            }
        )
        all_profiles = (
            [generate_college_profile(df)]
            + generate_solar_profiles(df, base_seed=42)
            + generate_consumer_profiles(df, base_seed=142)
        )

        build_metadata(all_profiles, output_dir=str(tmp_path))

        metadata_file = Path(str(tmp_path)) / "metadata.json"
        assert metadata_file.exists(), "metadata.json was not created"
        with open(metadata_file, encoding="utf-8") as f:
            loaded = json.load(f)
        assert "buildings" in loaded


# ===========================================================================
# Section 14: get_data_summary
# ===========================================================================


class TestGetDataSummary:
    """Tests for loader.get_data_summary."""

    def test_summary_has_expected_keys(self, college_raw_df: pd.DataFrame) -> None:
        """get_data_summary must return all expected top-level keys."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import get_data_summary

        summary = get_data_summary(college_raw_df)
        assert "total_samples" in summary
        assert "date_range" in summary
        assert "demand_kw" in summary
        assert "solar_generation_kw" in summary
        assert "years_covered" in summary

    def test_summary_total_samples_matches(self, college_raw_df: pd.DataFrame) -> None:
        """total_samples must equal the DataFrame length."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import get_data_summary

        summary = get_data_summary(college_raw_df)
        assert summary["total_samples"] == len(college_raw_df)

    def test_summary_demand_stats_are_finite(
        self, college_raw_df: pd.DataFrame
    ) -> None:
        """All demand statistics must be finite floats."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import get_data_summary

        summary = get_data_summary(college_raw_df)
        for key, val in summary["demand_kw"].items():
            assert np.isfinite(val), f"demand_kw.{key} is not finite: {val}"

    def test_summary_solar_max_nonnegative(self, college_raw_df: pd.DataFrame) -> None:
        """Solar max value must be non-negative."""
        # local
        from p2p_energy_trading.modules.profile_generator.loader import get_data_summary

        summary = get_data_summary(college_raw_df)
        assert summary["solar_generation_kw"]["max"] >= 0.0
