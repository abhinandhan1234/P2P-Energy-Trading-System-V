"""Project-wide constants, type aliases, and fixed configuration values.

This module defines all immutable constants used across the P2P Energy Trading
System. HESCOM tariff defaults defined here are fallback-only; runtime behaviour
must always use values from configuration (YAML -> env_config).

Reference: docs/module_12_repository_structure.md, docs/implementation_authority.md
"""

from __future__ import annotations

# standard library
from typing import Any

# third party
import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

AgentId = str
PolicyId = str
ObsDict = dict[AgentId, dict[str, NDArray[np.float32]]]
ActionDict = dict[AgentId, NDArray[np.float32]]
RewardDict = dict[AgentId, float]
TerminatedDict = dict[AgentId | str, bool]  # includes "__all__"
TruncatedDict = dict[AgentId | str, bool]  # includes "__all__"
InfoDict = dict[AgentId, dict[str, Any]]

# ---------------------------------------------------------------------------
# System dimensions (Module 4, Module 6 — DO NOT CHANGE)
# ---------------------------------------------------------------------------

NUM_AGENTS: int = 21
NUM_COLLEGE: int = 1
NUM_SOLAR: int = 15
NUM_CONSUMER: int = 5

ACTOR_OBS_DIM: int = 23
CRITIC_STATE_DIM: int = 243
ACTION_DIM: int = 3

NUM_LOCAL_FEATURES: int = 11  # per-agent local obs dimensions in critic

# ---------------------------------------------------------------------------
# Episode defaults (Module 6 §7)
# ---------------------------------------------------------------------------

EPISODE_LENGTH_DEBUG: int = 24  # 1 day
EPISODE_LENGTH_TRAINING: int = 168  # 1 week
EPISODE_LENGTH_EVALUATION: int = 720  # ~1 month

DEFAULT_EPISODE_LENGTH: int = EPISODE_LENGTH_TRAINING

# ---------------------------------------------------------------------------
# Agent identifiers (Module 6 §2 — DO NOT CHANGE)
# ---------------------------------------------------------------------------

COLLEGE_AGENT_ID: AgentId = "college"
SOLAR_AGENT_IDS: list[AgentId] = [f"solar_{i:02d}" for i in range(1, 16)]
CONSUMER_AGENT_IDS: list[AgentId] = [f"consumer_{i:02d}" for i in range(1, 6)]
ALL_AGENT_IDS: list[AgentId] = [COLLEGE_AGENT_ID] + SOLAR_AGENT_IDS + CONSUMER_AGENT_IDS

assert len(ALL_AGENT_IDS) == NUM_AGENTS, (
    f"Agent count mismatch: {len(ALL_AGENT_IDS)} != {NUM_AGENTS}"
)

# ---------------------------------------------------------------------------
# Policy identifiers (Module 6 §2 — DO NOT CHANGE)
# ---------------------------------------------------------------------------

POLICY_COLLEGE: PolicyId = "policy_college"
POLICY_SOLAR: PolicyId = "policy_solar"
POLICY_CONSUMER: PolicyId = "policy_consumer"

ALL_POLICY_IDS: list[PolicyId] = [POLICY_COLLEGE, POLICY_SOLAR, POLICY_CONSUMER]

AGENT_TO_POLICY: dict[AgentId, PolicyId] = {
    COLLEGE_AGENT_ID: POLICY_COLLEGE,
    **{aid: POLICY_SOLAR for aid in SOLAR_AGENT_IDS},
    **{aid: POLICY_CONSUMER for aid in CONSUMER_AGENT_IDS},
}

# ---------------------------------------------------------------------------
# Bus mapping (Module 2 §Bus Mapping — DO NOT CHANGE)
# ---------------------------------------------------------------------------

SLACK_BUS: int = 0  # PandaPower 0-indexed; Bus 1 in IEEE notation
COLLEGE_BUS: int = 6  # Bus 7 in IEEE notation

AGENT_TO_BUS: dict[AgentId, int] = {
    COLLEGE_AGENT_ID: COLLEGE_BUS,
    **{f"solar_{i:02d}": 6 + i for i in range(1, 16)},  # Buses 7-21 (0-indexed)
    **{f"consumer_{i:02d}": 21 + i for i in range(1, 6)},  # Buses 22-26 (0-indexed)
}

# ---------------------------------------------------------------------------
# Network constants (Module 2)
# ---------------------------------------------------------------------------

NUM_BUSES: int = 33
SLACK_VOLTAGE_PU: float = 1.02
PRIMARY_VOLTAGE_KV: float = 12.66
SECONDARY_VOLTAGE_KV: float = 0.4

SUBSTATION_MVA: float = 5.0
COLLEGE_TRANSFORMER_KVA: float = 500.0
SOLAR_TRANSFORMER_KVA: float = 100.0
CONSUMER_TRANSFORMER_KVA: float = 50.0

GRID_IMPORT_EXPORT_LIMIT_KW: float = 2000.0  # +/-2 MW

# ---------------------------------------------------------------------------
# Voltage and thermal limits (Module 2 §Constraints)
# ---------------------------------------------------------------------------

VOLTAGE_MIN_PU: float = 0.95
VOLTAGE_MAX_PU: float = 1.05
VOLTAGE_CATASTROPHIC_LOW_PU: float = 0.80
VOLTAGE_CATASTROPHIC_HIGH_PU: float = 1.20
LINE_LOADING_MAX_PERCENT: float = 100.0
TRANSFORMER_LOADING_MAX_PERCENT: float = 100.0

# ---------------------------------------------------------------------------
# Battery constants (Module 2 §Battery Model — DO NOT CHANGE)
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_KWH: float = 500.0
BATTERY_POWER_KW: float = 250.0
BATTERY_EFFICIENCY: float = 0.90  # round-trip
BATTERY_SOC_MIN: float = 0.10
BATTERY_SOC_MAX: float = 0.95
BATTERY_INITIAL_SOC_EVAL: float = 0.50
BATTERY_INITIAL_SOC_TRAIN_LOW: float = 0.30
BATTERY_INITIAL_SOC_TRAIN_HIGH: float = 0.70
BATTERY_MIN_DISPATCH_KW: float = 25.0  # 10% of power rating

# ---------------------------------------------------------------------------
# HESCOM tariff defaults (FALLBACK ONLY — runtime must use config values)
# Reference: docs/implementation_authority.md §Configuration Requirements
# ---------------------------------------------------------------------------

DEFAULT_GRID_BUY_RATE: float = 8.15  # Rs/kWh — agent buys from grid
DEFAULT_GRID_SELL_RATE: float = 3.56  # Rs/kWh — agent sells to grid
MAX_GRID_RATE: float = 10.0  # Rs/kWh — normalisation upper bound for grid rates

# ---------------------------------------------------------------------------
# Reward constants (Module 5 — DO NOT CHANGE equations)
# ---------------------------------------------------------------------------

REWARD_CLIP_MIN: float = -10.0
REWARD_CLIP_MAX: float = 10.0
EPSILON: float = 1e-8

# Reward weights — Phase 1 (Exploration)
REWARD_W_P2P: float = 0.1
REWARD_W_SELF: float = 0.05
REWARD_W_VOLTAGE_PHASE1: float = 2.0
REWARD_W_THERMAL_PHASE1: float = 2.0
REWARD_W_TRANSFORMER_PHASE1: float = 2.0
REWARD_W_SOC: float = 1.0
REWARD_W_CYCLING: float = 0.5
REWARD_W_STORAGE: float = 0.05
REWARD_W_IMPORT: float = 0.05

# Reward weights — Phase 2 (Constraint-aware)
REWARD_W_VOLTAGE_PHASE2: float = 5.0
REWARD_W_THERMAL_PHASE2: float = 5.0
REWARD_W_TRANSFORMER_PHASE2: float = 5.0

# ---------------------------------------------------------------------------
# Training defaults (Module 8)
# ---------------------------------------------------------------------------

DEFAULT_SEED: int = 42

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

RAW_DATA_FILENAME: str = "kls_vdit_hourly_market.csv"
RAW_CSV_COLUMN_TIMESTAMP: str = "Timestamp"
RAW_CSV_COLUMN_SOLAR: str = "College_Solar_kW"
RAW_CSV_COLUMN_DEMAND: str = "Campus_Demand_kW"

INTERNAL_COL_TIMESTAMP: str = "timestamp"
INTERNAL_COL_DEMAND: str = "demand_kw"
INTERNAL_COL_SOLAR: str = "solar_generation_kw"

PROFILE_REQUIRED_COLUMNS: list[str] = [
    INTERNAL_COL_TIMESTAMP,
    INTERNAL_COL_DEMAND,
    INTERNAL_COL_SOLAR,
]

# ---------------------------------------------------------------------------
# Environment registration name (Module 6 §Implementation Notes)
# ---------------------------------------------------------------------------

ENV_NAME: str = "p2p_energy_trading"

# ---------------------------------------------------------------------------
# Default safe action (Module 6 §4 — no trade, battery idle)
# ---------------------------------------------------------------------------

DEFAULT_SAFE_ACTION: list[float] = [0.0, 0.0, 0.5]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

# ---------------------------------------------------------------------------
# NaN thresholds (Module 6 §9)
# ---------------------------------------------------------------------------

MAX_NAN_ACTIONS_PER_STEP: int = 5
MAX_NAN_PER_EPISODE: int = 50

# ---------------------------------------------------------------------------
# PandaPower retry settings (Module 6 §9)
# ---------------------------------------------------------------------------

POWERFLOW_MAX_RETRIES: int = 3
POWERFLOW_DEFAULT_TOLERANCE: float = 1e-8
POWERFLOW_RELAXED_TOLERANCE: float = 1e-6

# ---------------------------------------------------------------------------
# Energy balance tolerance (Module 3 §6)
# ---------------------------------------------------------------------------

ENERGY_BALANCE_TOLERANCE_KW: float = 0.01
