"""Unit tests for baseline policies in the P2P Energy Trading evaluation framework.

Design reference: docs/module_9_evaluation_framework.md §12
"""

from __future__ import annotations

# third party
import numpy as np

# local
from p2p_energy_trading.constants import COLLEGE_AGENT_ID
from p2p_energy_trading.evaluation.baselines import (
    GridOnlyBaseline,
    HeuristicBaseline,
    RandomBaseline,
)


def test_grid_only_baseline() -> None:
    """Verify that GridOnlyBaseline always returns [0.0, 0.0, 0.5]."""
    policy = GridOnlyBaseline()
    obs = np.zeros(23, dtype=np.float32)

    # Test for different agents
    act_college = policy.compute_actions(obs, COLLEGE_AGENT_ID)
    act_solar = policy.compute_actions(obs, "solar_01")
    act_consumer = policy.compute_actions(obs, "consumer_01")

    assert np.allclose(act_college, [0.0, 0.0, 0.5])
    assert np.allclose(act_solar, [0.0, 0.0, 0.5])
    assert np.allclose(act_consumer, [0.0, 0.0, 0.5])


def test_random_baseline() -> None:
    """Verify that RandomBaseline returns actions within [0.0, 1.0]."""
    policy = RandomBaseline(seed=42)
    obs = np.zeros(23, dtype=np.float32)

    for _ in range(10):
        act = policy.compute_actions(obs, "solar_01")
        assert act.shape == (3,)
        assert np.all(act >= 0.0)
        assert np.all(act <= 1.0)


def test_heuristic_baseline_college() -> None:
    """Verify HeuristicBaseline college rules (battery charge/discharge/idle)."""
    policy = HeuristicBaseline(peak_demand=1000.0)

    # Base observation vector (23 dims)
    # Index 2: SoC
    # Index 3: surplus_norm
    # Index 4: deficit_norm

    # 1. College has surplus, SoC = 0.50 (should charge)
    obs = np.zeros(23, dtype=np.float32)
    obs[2] = 0.50  # soc
    obs[3] = 0.20  # surplus
    act = policy.compute_actions(obs, COLLEGE_AGENT_ID)
    assert np.allclose(act, [0.0, 1.0, 1.0])  # [buy=0, sell=1, charge=1.0]

    # 2. College has deficit, SoC = 0.50 (should discharge)
    obs = np.zeros(23, dtype=np.float32)
    obs[1] = 0.80  # demand (makes unnormalised demand > 250kW)
    obs[2] = 0.50  # soc
    obs[4] = 0.30  # deficit
    act = policy.compute_actions(obs, COLLEGE_AGENT_ID)
    assert np.allclose(act, [1.0, 0.0, 0.0])  # [buy=1, sell=0, discharge=0.0]

    # 3. College has surplus, but SoC is already full >= 0.90 (should idle)
    obs = np.zeros(23, dtype=np.float32)
    obs[2] = 0.92  # full soc
    obs[3] = 0.10  # surplus
    act = policy.compute_actions(obs, COLLEGE_AGENT_ID)
    assert np.allclose(act, [0.0, 1.0, 0.5])  # [buy=0, sell=1, idle=0.5]

    # 4. College has deficit, but SoC is already empty <= 0.15 (should idle)
    obs = np.zeros(23, dtype=np.float32)
    obs[1] = 0.20  # demand
    obs[2] = 0.10  # empty soc
    obs[4] = 0.15  # deficit
    act = policy.compute_actions(obs, COLLEGE_AGENT_ID)
    assert np.allclose(act, [1.0, 0.0, 0.5])  # [buy=1, sell=0, idle=0.5]


def test_heuristic_baseline_solar() -> None:
    """Verify HeuristicBaseline solar agent rules."""
    policy = HeuristicBaseline()

    # Solar agent has surplus, battery is idle
    obs = np.zeros(23, dtype=np.float32)
    obs[2] = 0.0  # battery SoC (ignored for solar)
    obs[3] = 0.5  # surplus
    act = policy.compute_actions(obs, "solar_02")
    assert np.allclose(act, [0.0, 1.0, 0.5])  # [buy=0, sell=1, battery=0.5]


def test_heuristic_baseline_consumer() -> None:
    """Verify HeuristicBaseline consumer agent rules."""
    policy = HeuristicBaseline()

    # Consumer agent has deficit, always buy full deficit, never sell, battery idle
    obs = np.zeros(23, dtype=np.float32)
    obs[4] = 0.8  # deficit
    act = policy.compute_actions(obs, "consumer_03")
    assert np.allclose(act, [1.0, 0.0, 0.5])  # [buy=1, sell=0, battery=0.5]
