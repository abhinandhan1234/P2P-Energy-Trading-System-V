"""CLI entry point for Module 1 profile generation.

Generates all 21 building profiles (1 college, 15 solar, 5 consumer)
from the raw CSV dataset and writes them as parquet files to the
processed data directory, along with a metadata.json summary.

Usage:
    python -m p2p_energy_trading.modules.profile_generator.run
    python -m p2p_energy_trading.modules.profile_generator.run \\
        --data-dir data/raw --output-dir data/processed --seed 42

Design reference: docs/module_1_profile_generator.md §Output Structure
"""

from __future__ import annotations

# standard library
import argparse
import logging
import sys
from pathlib import Path

# local
from p2p_energy_trading.constants import DEFAULT_SEED
from p2p_energy_trading.exceptions import ProfileGenerationError
from p2p_energy_trading.modules.profile_generator.generator import (
    BuildingProfile,
    generate_college_profile,
    generate_consumer_profiles,
    generate_solar_profiles,
)
from p2p_energy_trading.modules.profile_generator.loader import load_raw_data
from p2p_energy_trading.modules.profile_generator.metadata import build_metadata
from p2p_energy_trading.modules.profile_generator.validator import validate_portfolio
from p2p_energy_trading.utils.logging import setup_logging
from p2p_energy_trading.utils.seeding import set_numpy_seed

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expected output file names — from module_1_profile_generator.md §Output
# ---------------------------------------------------------------------------

OUTPUT_COLLEGE = "college.parquet"
OUTPUT_SOLAR_TEMPLATE = "solar_{:02d}.parquet"
OUTPUT_CONSUMER_TEMPLATE = "consumer_{:02d}.parquet"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with data_dir, output_dir, seed, verbose.
    """
    parser = argparse.ArgumentParser(
        prog="profile_generator",
        description=(
            "Generate 21 synthetic building energy profiles "
            "(1 college, 15 solar, 5 consumer) from the raw campus dataset."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing the raw CSV file (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write parquet files and metadata.json (default: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args(argv)


def run_generation(
    data_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/processed",
    seed: int = DEFAULT_SEED,
) -> list[BuildingProfile]:
    """Generate all 21 building profiles and write outputs.

    Orchestrates the full Module 1 pipeline:
    1. Load and validate raw CSV
    2. Generate college, solar, and consumer profiles
    3. Validate each profile and the complete portfolio
    4. Write parquet files to output_dir
    5. Write metadata.json

    Args:
        data_dir: Directory containing the raw CSV.
        output_dir: Directory for parquet output files.
        seed: Random seed for reproducibility.

    Returns:
        List of all 21 BuildingProfile objects.

    Raises:
        ProfileGenerationError: If any step fails.
    """
    set_numpy_seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load raw data
    logger.info("Loading raw data from '%s'", data_dir)
    college_df = load_raw_data(data_dir=data_dir)
    logger.info("Raw data loaded: %d samples", len(college_df))

    # Step 2: Generate profiles
    logger.info("Generating college profile...")
    college_profile = generate_college_profile(college_df)

    logger.info("Generating 15 solar building profiles (seed=%d)...", seed)
    solar_profiles = generate_solar_profiles(college_df, base_seed=seed)

    logger.info("Generating 5 consumer building profiles (seed=%d)...", seed + 100)
    consumer_profiles = generate_consumer_profiles(college_df, base_seed=seed + 100)

    all_profiles: list[BuildingProfile] = (
        [college_profile] + solar_profiles + consumer_profiles
    )

    # Step 3: Validate portfolio
    logger.info("Validating portfolio (%d buildings)...", len(all_profiles))
    profile_dict = {p.building_id: p.df for p in all_profiles}
    validate_portfolio(profile_dict)
    logger.info("Portfolio validation passed")

    # Step 4: Write parquet files
    logger.info("Writing parquet files to '%s'...", output_path)
    _write_parquet_files(all_profiles, output_path)

    # Step 5: Write metadata
    logger.info("Writing metadata.json...")
    build_metadata(all_profiles, output_path)

    logger.info(
        "Module 1 complete: %d parquet files + metadata.json written to '%s'",
        len(all_profiles),
        output_path,
    )
    return all_profiles


def _write_parquet_files(
    profiles: list[BuildingProfile],
    output_dir: Path,
) -> None:
    """Write all building profiles as parquet files.

    Args:
        profiles: List of BuildingProfile objects to serialise.
        output_dir: Target directory (must exist).

    Raises:
        ProfileGenerationError: If a file cannot be written.
    """
    for profile in profiles:
        filename = _get_output_filename(profile)
        file_path = output_dir / filename
        try:
            profile.df.to_parquet(file_path, index=False)
            logger.debug("Written: %s", file_path)
        except Exception as e:
            raise ProfileGenerationError(
                f"Failed to write parquet for '{profile.building_id}' "
                f"to '{file_path}': {e}"
            ) from e


def _get_output_filename(profile: BuildingProfile) -> str:
    """Return the parquet filename for a given building profile.

    Args:
        profile: BuildingProfile with building_id and building_type.

    Returns:
        Filename string, e.g. 'college.parquet', 'solar_01.parquet'.

    Raises:
        ProfileGenerationError: If the building_id is unrecognised.
    """
    if profile.building_type == "college":
        return OUTPUT_COLLEGE
    elif profile.building_type == "solar":
        # Extract index from building_id 'solar_NN'
        idx = int(profile.building_id.split("_")[1])
        return OUTPUT_SOLAR_TEMPLATE.format(idx)
    elif profile.building_type == "consumer":
        idx = int(profile.building_id.split("_")[1])
        return OUTPUT_CONSUMER_TEMPLATE.format(idx)
    else:
        raise ProfileGenerationError(
            f"Unknown building_type '{profile.building_type}' "
            f"for building_id '{profile.building_id}'"
        )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    args = parse_args(argv)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    try:
        run_generation(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        return 0
    except ProfileGenerationError as e:
        logger.error("Profile generation failed: %s", e)
        return 1
    except Exception as e:
        logger.exception("Unexpected error during profile generation: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
