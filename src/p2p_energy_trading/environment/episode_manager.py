"""Episode Manager for the P2P Energy Trading Environment.

Pre-loads profile data, manages offset sampling, and handles episode slicing.
Design reference: docs/module_6_multiagent_environment.md
"""

from __future__ import annotations

# standard library
import json
import logging
from pathlib import Path
from typing import Any

# third party
import numpy as np
import pandas as pd

# local
from p2p_energy_trading.constants import (
    ALL_AGENT_IDS,
    COLLEGE_AGENT_ID,
    CONSUMER_AGENT_IDS,
    INTERNAL_COL_DEMAND,
    INTERNAL_COL_SOLAR,
    INTERNAL_COL_TIMESTAMP,
    PROFILE_REQUIRED_COLUMNS,
    SOLAR_AGENT_IDS,
)
from p2p_energy_trading.exceptions import ProfileGenerationError

logger = logging.getLogger(__name__)


class EpisodeManager:
    """Manages energy profile loading, memory-caching, and slicing for episodes."""

    def __init__(self, data_dir: str | Path, episode_length: int) -> None:
        """Initialize the EpisodeManager.

        Args:
            data_dir: Path to directory containing Parquet profiles and metadata.json.
            episode_length: Length of an episode in hours.
        """
        self._data_dir = Path(data_dir)
        self._episode_length = episode_length
        self._raw_profiles: dict[str, pd.DataFrame] = {}
        self._metadata: dict[str, Any] = {}

    @property
    def metadata(self) -> dict[str, Any]:
        """Return the loaded portfolio metadata dictionary."""
        return self._metadata

    @property
    def raw_profiles(self) -> dict[str, pd.DataFrame]:
        """Return the in-memory cached yearly raw profiles."""
        return self._raw_profiles

    def load_profiles(self) -> None:
        """Pre-load all Parquet profile dataframes and metadata.json into memory.

        Raises:
            FileNotFoundError: If any profile or metadata file is missing.
            ValueError: If dataframe format/columns are invalid.
        """
        metadata_path = self._data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at expected path: '{metadata_path}'"
            )

        try:
            with open(metadata_path, encoding="utf-8") as f:
                self._metadata = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ProfileGenerationError(
                f"Failed to load or parse metadata.json at '{metadata_path}': {e}"
            ) from e

        # Pre-load building profiles
        for aid in ALL_AGENT_IDS:
            if aid == COLLEGE_AGENT_ID:
                filename = "college.parquet"
            elif aid in SOLAR_AGENT_IDS:
                # Suffix mapping: e.g. solar_01 -> solar_01.parquet
                filename = f"{aid}.parquet"
            elif aid in CONSUMER_AGENT_IDS:
                filename = f"{aid}.parquet"
            else:
                raise ValueError(f"Unknown agent type for ID: {aid}")

            file_path = self._data_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Profile Parquet file not found at expected path: '{file_path}'"
                )

            try:
                df = pd.read_parquet(file_path)
            except (OSError, ValueError) as e:
                raise ProfileGenerationError(
                    f"Failed to read Parquet file for agent '{aid}'"
                    f" at '{file_path}': {e}"
                ) from e

            # Verification of columns
            for col in PROFILE_REQUIRED_COLUMNS:
                if col not in df.columns:
                    raise ValueError(
                        f"Profile Parquet for '{aid}' is missing"
                        f" required column: '{col}'"
                    )

            # Coerce timestamps and convert columns to float
            df[INTERNAL_COL_TIMESTAMP] = pd.to_datetime(df[INTERNAL_COL_TIMESTAMP])
            df[INTERNAL_COL_DEMAND] = df[INTERNAL_COL_DEMAND].astype(np.float64)
            df[INTERNAL_COL_SOLAR] = df[INTERNAL_COL_SOLAR].astype(np.float64)

            self._raw_profiles[aid] = df

        # Verify length consistency
        lengths = {aid: len(df) for aid, df in self._raw_profiles.items()}
        first_aid = ALL_AGENT_IDS[0]
        expected_len = lengths[first_aid]
        mismatches = [
            f"{aid} = {length} rows"
            for aid, length in lengths.items()
            if length != expected_len
        ]
        if mismatches:
            msg = "\n".join([f"{first_aid} = {expected_len} rows"] + mismatches)
            raise ValueError(f"Profile length mismatch:\n{msg}")

        logger.info(
            "Successfully cached %d agent profiles in memory.", len(self._raw_profiles)
        )

    def reset(
        self,
        seed: int | None = None,
        is_eval: bool = False,
        eval_start_hour: int = 0,
    ) -> tuple[int, dict[str, pd.DataFrame]]:
        """Slice profiles for a new episode.

        Determines start offset, performs slicing of length (episode_length + 1).

        Args:
            seed: Optional seed for start offset random sampling (training mode only).
            is_eval: True for deterministic evaluation mode.
            eval_start_hour: Fixed start index in evaluation mode.

        Returns:
            Tuple containing:
            - int: The starting row index offset.
            - dict[str, pd.DataFrame]: Sliced dataframes per agent.
        """
        # Determine total length of raw data
        first_agent = ALL_AGENT_IDS[0]
        total_hours = len(self._raw_profiles[first_agent])

        # Validate episode length fits in raw data
        current_episode_length = self._episode_length
        if total_hours <= current_episode_length:
            raise ValueError(
                f"Configured episode_length ({current_episode_length})"
                f" exceeds available profile length ({total_hours})."
            )

        # 1. Determine start index offset
        if is_eval:
            start_idx = max(
                0, min(total_hours - current_episode_length - 1, eval_start_hour)
            )
        else:
            # Sample start index stochastically
            rng = np.random.default_rng(seed)
            max_start = total_hours - current_episode_length - 1
            start_idx = int(rng.integers(0, max_start + 1))

        # 2. Slice dataframes
        sliced_profiles: dict[str, pd.DataFrame] = {}
        slice_len = current_episode_length + 1

        if start_idx + slice_len > total_hours:
            raise ValueError(
                f"Requested slice (start: {start_idx}, length: {slice_len})"
                f" exceeds dataframe bounds ({total_hours})."
            )

        for aid in ALL_AGENT_IDS:
            df = self._raw_profiles[aid]
            sliced_df = df.iloc[start_idx : start_idx + slice_len].copy()
            if len(sliced_df) != slice_len:
                raise ValueError(
                    f"Sliced profile length ({len(sliced_df)}) for {aid}"
                    f" does not match requested slice length ({slice_len})."
                )
            # Reset index to clean 0-indexed for step lookups
            sliced_df.reset_index(drop=True, inplace=True)
            sliced_profiles[aid] = sliced_df

        logger.debug(
            "Reset episode window: start=%d, length=%d, slice_rows=%d.",
            start_idx,
            current_episode_length,
            slice_len,
        )

        return start_idx, sliced_profiles
