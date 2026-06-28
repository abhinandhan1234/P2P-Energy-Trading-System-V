"""PandaPower Network Builder for the IEEE 33-bus P2P microgrid.

This module constructs the physical distribution system representation, including
primary feeder lines, transformers, buses, and agent-specific load/generator links.

Design reference: docs/module_2_pandapower_network.md
"""

from __future__ import annotations

# standard library
import logging

# third party
import pandapower as pp

# local
from p2p_energy_trading.constants import (
    AGENT_TO_BUS,
    ALL_AGENT_IDS,
    COLLEGE_AGENT_ID,
    COLLEGE_TRANSFORMER_KVA,
    CONSUMER_TRANSFORMER_KVA,
    NUM_BUSES,
    PRIMARY_VOLTAGE_KV,
    SECONDARY_VOLTAGE_KV,
    SLACK_VOLTAGE_PU,
    SOLAR_TRANSFORMER_KVA,
    SUBSTATION_MVA,
)
from p2p_energy_trading.exceptions import PowerFlowError

logger = logging.getLogger(__name__)

# Module-level cached maps for deterministic element indices
_LOAD_INDEX_MAP: dict[str, int] = {aid: idx for idx, aid in enumerate(ALL_AGENT_IDS)}
_SGEN_INDEX_MAP: dict[str, int] = {}

# Build the sgen index map deterministically based on agent order
sgen_count = 0
for aid in ALL_AGENT_IDS:
    if aid == COLLEGE_AGENT_ID or aid.startswith("solar_"):
        _SGEN_INDEX_MAP[aid] = sgen_count
        sgen_count += 1


def build_network() -> pp.pandapowerNet:
    """Create the IEEE 33-bus P2P microgrid network.

    Returns a fully wired pandapowerNet with:
    - 33 primary buses (0-indexed, at 12.66 kV)
    - 1 external grid bus (at 110 kV)
    - 21 secondary buses (at 0.4 kV, one per agent)
    - Substation transformer: 5 MVA, 110/12.66 kV connecting external grid to Bus 0
    - College transformer: 500 kVA connecting Bus 6 to college secondary
    - 15 solar transformers: 100 kVA each connecting primary to secondary
    - 5 consumer transformers: 50 kVA each connecting primary to secondary
    - All 32 IEEE 33-bus distribution lines
    - Placeholder loads at all 21 agent secondary buses
    - Placeholder sgens (static generators) at college + solar secondary buses
    """
    net = pp.create_empty_network(name="IEEE_33_Bus_P2P_Microgrid")

    # 1. Create 33 primary buses (indices 0 to 32)
    for i in range(NUM_BUSES):
        pp.create_bus(
            net,
            vn_kv=PRIMARY_VOLTAGE_KV,
            name=f"Primary_Bus_{i + 1}",
        )

    # 2. Create the external utility grid bus (nominal 110 kV)
    utility_hv_bus = pp.create_bus(
        net,
        vn_kv=110.0,
        name="Utility_HV_Bus",
    )

    # 3. Create the external grid connection (slack bus) at the HV side
    pp.create_ext_grid(
        net,
        bus=utility_hv_bus,
        vm_pu=SLACK_VOLTAGE_PU,
        name="Utility_Slack_Grid",
    )

    # 4. Create the substation transformer (5 MVA, 110 kV to 12.66 kV)
    # Standard parameters: vk=6%, vkr=1%, pfe=5 kW, i0=0.2%
    pp.create_transformer_from_parameters(
        net,
        hv_bus=utility_hv_bus,
        lv_bus=0,
        sn_mva=SUBSTATION_MVA,
        vn_hv_kv=110.0,
        vn_lv_kv=PRIMARY_VOLTAGE_KV,
        vkr_percent=1.0,
        vk_percent=6.0,
        pfe_kw=SUBSTATION_MVA * 1000.0 * 0.001,
        i0_percent=0.2,
        name="substation_transformer",
    )

    # 5. Create 32 IEEE 33-bus distribution lines
    # Length is 1 km, c_nf_per_km is 0, max_i_ka is 0.4
    lines_data = [
        (0, 1, 0.0922, 0.0470),
        (1, 2, 0.4930, 0.2511),
        (2, 3, 0.3660, 0.1864),
        (3, 4, 0.3811, 0.1941),
        (4, 5, 0.8190, 0.7070),
        (5, 6, 0.1872, 0.6188),
        (6, 7, 0.7114, 0.2351),
        (7, 8, 1.0300, 0.7400),
        (8, 9, 1.0440, 0.7400),
        (9, 10, 0.1966, 0.0650),
        (10, 11, 0.3744, 0.1238),
        (11, 12, 1.4680, 1.1550),
        (12, 13, 0.5416, 0.7129),
        (13, 14, 0.5910, 0.5260),
        (14, 15, 0.7463, 0.5450),
        (15, 16, 1.2890, 1.7210),
        (16, 17, 0.7320, 0.5740),
        (1, 18, 0.1640, 0.1565),
        (18, 19, 1.5042, 1.3554),
        (19, 20, 0.4095, 0.4784),
        (20, 21, 0.7089, 0.9373),
        (2, 22, 0.4512, 0.3083),
        (22, 23, 0.8980, 0.7091),
        (23, 24, 0.8960, 0.7011),
        (5, 25, 0.2030, 0.1034),
        (25, 26, 0.2842, 0.1447),
        (26, 27, 1.0590, 0.9337),
        (27, 28, 0.8042, 0.7006),
        (28, 29, 0.5075, 0.2585),
        (29, 30, 0.9744, 0.9630),
        (30, 31, 0.3105, 0.3619),
        (31, 32, 0.3410, 0.5302),
    ]

    for idx, (from_b, to_b, r, x) in enumerate(lines_data):
        pp.create_line_from_parameters(
            net,
            from_bus=from_b,
            to_bus=to_b,
            length_km=1.0,
            r_ohm_per_km=r,
            x_ohm_per_km=x,
            c_nf_per_km=0.0,
            max_i_ka=0.4,
            name=f"Line_{idx + 1}",
        )

    # 6. Create Agent secondary buses, transformers, loads, and sgens
    for aid in ALL_AGENT_IDS:
        primary_bus = AGENT_TO_BUS[aid]

        # Determine transformer capacity (kVA) based on agent type
        if aid == COLLEGE_AGENT_ID:
            kva = COLLEGE_TRANSFORMER_KVA
        elif aid.startswith("solar_"):
            kva = SOLAR_TRANSFORMER_KVA
        else:
            kva = CONSUMER_TRANSFORMER_KVA

        # Create secondary bus (0.4 kV)
        sec_bus = pp.create_bus(
            net,
            vn_kv=SECONDARY_VOLTAGE_KV,
            name=f"{aid}_Secondary_Bus",
        )

        # Create step-down transformer
        pp.create_transformer_from_parameters(
            net,
            hv_bus=primary_bus,
            lv_bus=sec_bus,
            sn_mva=kva / 1000.0,
            vn_hv_kv=PRIMARY_VOLTAGE_KV,
            vn_lv_kv=SECONDARY_VOLTAGE_KV,
            vkr_percent=1.0,
            vk_percent=4.0,
            pfe_kw=kva * 0.001,  # 0.1% iron loss scaling
            i0_percent=0.2,
            name=f"{aid}_transformer",
        )

        # Create load at secondary bus (initially 0 kW)
        pp.create_load(
            net,
            bus=sec_bus,
            p_mw=0.0,
            q_mvar=0.0,
            name=f"{aid}_load",
        )

        # Create static generator (if solar-enabled agent)
        if aid == COLLEGE_AGENT_ID or aid.startswith("solar_"):
            pp.create_sgen(
                net,
                bus=sec_bus,
                p_mw=0.0,
                q_mvar=0.0,
                name=f"{aid}_sgen",
            )

    logger.info("Successfully constructed PandaPower IEEE 33-bus network")
    return net


def get_agent_bus_index(agent_id: str) -> int:
    """Return the 0-indexed PandaPower primary bus index for an agent.

    Args:
        agent_id: String identifier of the agent.

    Returns:
        The 0-indexed primary bus index.
    """
    if agent_id not in AGENT_TO_BUS:
        raise PowerFlowError(f"Agent '{agent_id}' has no assigned primary bus")
    return AGENT_TO_BUS[agent_id]


def get_load_index(agent_id: str) -> int:
    """Return the PandaPower load element index for an agent.

    Args:
        agent_id: String identifier of the agent.

    Returns:
        The load element index.
    """
    if agent_id not in _LOAD_INDEX_MAP:
        raise PowerFlowError(f"Agent '{agent_id}' has no assigned load index")
    return _LOAD_INDEX_MAP[agent_id]


def get_sgen_index(agent_id: str) -> int:
    """Return the PandaPower sgen element index for an agent.

    Args:
        agent_id: String identifier of the agent.

    Returns:
        The sgen element index.

    Raises:
        PowerFlowError: If the agent is a consumer agent and has no sgen.
    """
    if agent_id not in _SGEN_INDEX_MAP:
        raise PowerFlowError(
            f"Agent '{agent_id}' has no assigned static generator (sgen)"
        )
    return _SGEN_INDEX_MAP[agent_id]
