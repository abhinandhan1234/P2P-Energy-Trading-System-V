"""Multi-Agent Environment for P2P Energy Trading.

Subclasses RLlib's MultiAgentEnv and coordinates building profiles, market settling, stateful
battery model progression, PandaPower power flows, observations compilation, and reward calculations.

Design reference: docs/module_6_multiagent_environment.md
"""

from __future__ import annotations

# standard library
import dataclasses
import importlib
import logging
import math
from typing import Any

# third party
import gymnasium
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# standard library
import copy

# local
from p2p_energy_trading.constants import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    ALL_AGENT_IDS,
    BATTERY_INITIAL_SOC_EVAL,
    BATTERY_INITIAL_SOC_TRAIN_HIGH,
    BATTERY_INITIAL_SOC_TRAIN_LOW,
    BATTERY_SOC_MAX,
    BATTERY_SOC_MIN,
    COLLEGE_AGENT_ID,
    CRITIC_STATE_DIM,
    DEFAULT_EPISODE_LENGTH,
    DEFAULT_GRID_BUY_RATE,
    DEFAULT_GRID_SELL_RATE,
    DEFAULT_SAFE_ACTION,
    DEFAULT_SEED,
    INTERNAL_COL_DEMAND,
    INTERNAL_COL_SOLAR,
    INTERNAL_COL_TIMESTAMP,
    MAX_NAN_ACTIONS_PER_STEP,
    MAX_NAN_PER_EPISODE,
    POWERFLOW_DEFAULT_TOLERANCE,
    POWERFLOW_MAX_RETRIES,
    VOLTAGE_CATASTROPHIC_HIGH_PU,
    VOLTAGE_CATASTROPHIC_LOW_PU,
    ActionDict,
    InfoDict,
    ObsDict,
    RewardDict,
    TerminatedDict,
    TruncatedDict,
)
from p2p_energy_trading.environment.action_handler import ActionHandler
from p2p_energy_trading.environment.agent_registry import AgentRegistry
from p2p_energy_trading.environment.compatibility import MultiAgentEnv
from p2p_energy_trading.environment.episode_manager import EpisodeManager
from p2p_energy_trading.exceptions import (
    ConfigValidationError,
    MarketClearingError,
    PowerFlowError,
)
from p2p_energy_trading.modules.market.models import MarketState, SettlementRecord
from p2p_energy_trading.modules.market.settlement import process_settlements
from p2p_energy_trading.modules.network.battery import BatteryModel
from p2p_energy_trading.modules.network.constraints import check_constraints
from p2p_energy_trading.modules.network.network_builder import build_network
from p2p_energy_trading.modules.network.powerflow import (
    PowerFlowResult,
    run_power_flow,
    update_network_loads,
)
from p2p_energy_trading.modules.observation.builder import build_observations
from p2p_energy_trading.modules.reward.aggregator import compute_all_rewards


class P2PEnergyTradingEnv(MultiAgentEnv):
    """Custom MultiAgentEnv subclass for the P2P Energy Trading campus microgrid.

    Coordinates 21 agents (1 college prosumer with battery, 15 solar prosumers,
    5 pure consumer buildings) on an IEEE 33-bus radial distribution network.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the P2P Energy Trading Environment.

        Args:
            config: Environment configuration dictionary.

        Raises:
            ConfigValidationError: If configuration fails validation checks.
            FileNotFoundError: If profile parquet files or metadata.json are missing.
        """
        # Save raw config dict or empty fallback
        self.config = copy.deepcopy(config or {})

        # 1. Parse and validate configuration parameters
        self.episode_length = int(
            self.config.get("episode_length", DEFAULT_EPISODE_LENGTH)
        )
        if not (24 <= self.episode_length <= 8760):
            raise ConfigValidationError(
                f"Configured 'episode_length' ({self.episode_length}) is outside allowed range [24, 8760]."
            )

        self.pandapower_bypass = bool(self.config.get("pandapower_bypass", False))
        self.eval_mode = bool(self.config.get("eval_mode", False))
        self.grid_buy_rate = float(
            self.config.get("grid_buy_rate", DEFAULT_GRID_BUY_RATE)
        )
        self.grid_sell_rate = float(
            self.config.get("grid_sell_rate", DEFAULT_GRID_SELL_RATE)
        )
        self.data_dir = self.config.get("data_dir", "data/processed")

        self.curriculum_transition_step = int(
            self.config.get("curriculum_transition_step", 100000)
        )
        self.powerflow_max_retries = int(
            self.config.get("powerflow_max_retries", POWERFLOW_MAX_RETRIES)
        )
        self.powerflow_tolerance = float(
            self.config.get("powerflow_tolerance", POWERFLOW_DEFAULT_TOLERANCE)
        )
        self.max_nan_actions = int(
            self.config.get("max_nan_actions", MAX_NAN_ACTIONS_PER_STEP)
        )
        self.max_nan_episode = int(
            self.config.get("max_nan_episode", MAX_NAN_PER_EPISODE)
        )

        # Check buy/sell tariff consistency
        if self.grid_buy_rate <= self.grid_sell_rate:
            raise ConfigValidationError(
                f"Configured 'grid_buy_rate' ({self.grid_buy_rate}) must be strictly greater than "
                f"'grid_sell_rate' ({self.grid_sell_rate}) to prevent arbitrage."
            )

        # 2. Instantiate helper modules

        self.agent_registry = AgentRegistry()
        self.action_handler = ActionHandler()
        self.episode_manager = EpisodeManager(self.data_dir, self.episode_length)
        self.episode_manager.load_profiles()

        # 3. Instantiate stateful battery model
        self.battery_model = BatteryModel()

        # 4. Initialize network structure (PandaPower IEEE 33-bus model)
        self.net = build_network()

        # 5. Define observation and action spaces (required for MultiAgentEnv)
        self._agent_ids = set(ALL_AGENT_IDS)
        self.observation_space = gymnasium.spaces.Dict(
            {
                "obs": gymnasium.spaces.Box(
                    low=-1.0, high=1.0, shape=(ACTOR_OBS_DIM,), dtype=np.float32
                ),
                "state": gymnasium.spaces.Box(
                    low=-1.0, high=1.0, shape=(CRITIC_STATE_DIM,), dtype=np.float32
                ),
            }
        )
        self.action_space = gymnasium.spaces.Box(
            low=0.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32
        )

        self.observation_spaces = {
            aid: self.observation_space for aid in self._agent_ids
        }
        self.action_spaces = {aid: self.action_space for aid in self._agent_ids}

        # 6. Initialize global step counter and state attributes
        self.total_env_steps = 0
        self.curriculum_phase = 1
        self.episode_start_hour = 0
        self.current_timestep = 0
        self.last_actions: ActionDict = {}
        self.prev_battery_dispatch_kw = 0.0

        # Cache standard default action array and default market state
        self._default_action = np.asarray(DEFAULT_SAFE_ACTION, dtype=np.float32)
        self._default_market = MarketState(
            p2p_clearing_price=(self.grid_buy_rate + self.grid_sell_rate) / 2.0,
            total_p2p_volume=0.0,
            p2p_utilisation_ratio=0.5,
            grid_import_total=0.0,
            grid_export_total=0.0,
            voltage_violation=False,
            thermal_violation=False,
            curtailment_applied=False,
            total_bids=0.0,
            total_offers=0.0,
        )

        # Cache imported PandaPower module
        self._pandapower = None
        self._pandapower_attempted = False

        # Cache standard normalization constant max_possible_cost (College Peak Demand * Grid Buy Tariff)
        college_meta = self.episode_manager.metadata.get("buildings", {}).get(
            COLLEGE_AGENT_ID, {}
        )
        peak_demand_college_kw = (
            college_meta.get("profile_stats", {})
            .get(INTERNAL_COL_DEMAND, {})
            .get("peak", 0.0)
        )
        self.max_possible_cost = float(peak_demand_college_kw * self.grid_buy_rate)

        super().__init__()

        logger.info(
            "P2PEnergyTradingEnv initialized. eval_mode=%s, bypass=%s, buy=%s, sell=%s.",
            self.eval_mode,
            self.pandapower_bypass,
            self.grid_buy_rate,
            self.grid_sell_rate,
        )

    def seed(self, seed: int | None = None) -> list[int]:
        """Set the random seed for the environment.

        Args:
            seed: Seed integer.

        Returns:
            List containing the active seed.
        """
        if seed is None:
            seed = DEFAULT_SEED
        np.random.seed(seed)
        # standard library
        import random

        random.seed(seed)
        return [seed]

    def _get_profiles_at_timestep(
        self,
        timestep: int,
    ) -> tuple[dict[str, float], dict[str, float], pd.Timestamp]:
        """Extract demand and solar profiles for all agents at a given timestep.

        Args:
            timestep: The profile index to access.

        Returns:
            Tuple containing:
            - dict[str, float]: Demands mapping agent ID to value.
            - dict[str, float]: Solar generation mapping agent ID to value.
            - pd.Timestamp: The timestamp of the college agent profile.

        Raises:
            IndexError: If the requested timestep is out of range of the available profiles.
        """
        profile_len = len(self.episode_profiles[COLLEGE_AGENT_ID])
        if timestep < 0 or timestep >= profile_len:
            raise IndexError(
                f"Requested timestep {timestep} is out of bounds for available profiles of length {profile_len}."
            )

        demands = {
            aid: float(self.episode_profiles[aid].at[timestep, INTERNAL_COL_DEMAND])
            for aid in ALL_AGENT_IDS
        }
        solars = {
            aid: float(self.episode_profiles[aid].at[timestep, INTERNAL_COL_SOLAR])
            for aid in ALL_AGENT_IDS
        }
        ts = self.episode_profiles[COLLEGE_AGENT_ID].at[
            timestep, INTERNAL_COL_TIMESTAMP
        ]
        return demands, solars, ts

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsDict, InfoDict]:
        """Reset the environment for a new episode.

        Args:
            seed: Optional seed for random offset and training SoC selection.
            options: Optional dictionaries containing extra startup variables.

        Returns:
            Tuple containing:
            - obs_dict: Initial observation dictionary.
            - info_dict: Metadata dictionary containing starting offset and step indices.
        """
        if seed is not None:
            self.seed(seed)

        # 1. Reset battery SoC (Evaluation = 0.50, Training = random [0.3, 0.7])
        if self.eval_mode:
            init_soc = BATTERY_INITIAL_SOC_EVAL
        else:
            init_soc = float(
                np.random.uniform(
                    BATTERY_INITIAL_SOC_TRAIN_LOW, BATTERY_INITIAL_SOC_TRAIN_HIGH
                )
            )

        if not BATTERY_SOC_MIN <= init_soc <= BATTERY_SOC_MAX:
            raise ValueError(
                f"Initial SoC {init_soc} is outside allowed bounds [{BATTERY_SOC_MIN}, {BATTERY_SOC_MAX}]."
            )

        self.battery_model.reset(initial_soc=init_soc)

        # 2. Reset episode manager slice offset
        eval_start_hour = int(self.config.get("eval_start_hour", 0))
        if options is not None and "eval_start_hour" in options:
            eval_start_hour = int(options["eval_start_hour"])

        start_hour, sliced_profiles = self.episode_manager.reset(
            seed=seed, is_eval=self.eval_mode, eval_start_hour=eval_start_hour
        )
        self.episode_start_hour = start_hour
        self.episode_profiles = sliced_profiles

        self.current_timestep = 0
        self.prev_battery_dispatch_kw = 0.0
        self.nan_episode_count = 0
        self._pandapower = None
        self._pandapower_attempted = False

        self.last_actions = {aid: self._default_action.copy() for aid in ALL_AGENT_IDS}

        demands_t0, solar_t0, timestamp_t0 = self._get_profiles_at_timestep(0)

        # Default fallback values for grid and market clearing states (copied to protect the cache)
        if dataclasses.is_dataclass(self._default_market):
            default_market = dataclasses.replace(self._default_market)
        elif hasattr(self._default_market, "copy"):
            default_market = self._default_market.copy()
        else:
            default_market = copy.copy(self._default_market)

        obs_dict = build_observations(
            demands_kw=demands_t0,
            solar_kw=solar_t0,
            battery_state=self.battery_model.get_state(),
            last_actions=self.last_actions,
            grid_result=None,
            market_state=default_market,
            timestamp=timestamp_t0,
            grid_buy_rate=self.grid_buy_rate,
            grid_sell_rate=self.grid_sell_rate,
            metadata=self.episode_manager.metadata,
        )

        info_dict = {
            aid: {
                "timestep": 0,
                "episode_start_hour": self.episode_start_hour,
            }
            for aid in ALL_AGENT_IDS
        }

        return obs_dict, info_dict

    @staticmethod
    def _create_agent_bool_dict(value: bool) -> dict[str, bool]:
        """Create a dictionary with ALL_AGENT_IDS as keys and a boolean value."""
        return {aid: value for aid in ALL_AGENT_IDS}

    def _validate_actions(
        self, action_dict: ActionDict
    ) -> tuple[ActionDict, TruncatedDict]:
        """Validate and pre-clip raw action fractions to [0.0, 1.0].

        Additionally monitors NaN thresholds and updates self.nan_episode_count.
        If thresholds are exceeded, flags truncation.
        """
        cleaned_actions, nan_count = self.action_handler.validate_and_scale(action_dict)
        self.nan_episode_count += nan_count

        truncated_dict = self._create_agent_bool_dict(False)
        truncated_dict["__all__"] = False

        if (
            nan_count > self.max_nan_actions
            or self.nan_episode_count > self.max_nan_episode
        ):
            logger.error(
                "NaN action threshold exceeded (step NaNs: %d, episode NaNs: %d). "
                "Truncating episode rollout.",
                nan_count,
                self.nan_episode_count,
            )
            truncated_dict["__all__"] = True

        return cleaned_actions, truncated_dict

    def _process_market(
        self,
        demands_kw: dict[str, float],
        solar_kw: dict[str, float],
        actions: ActionDict,
    ) -> tuple[dict[str, SettlementRecord], MarketState]:
        """Execute market settlements clearing."""
        return process_settlements(
            demands_kw=demands_kw,
            solar_kw=solar_kw,
            actions=actions,
            battery_soc=self.battery_model.soc,
            grid_buy_rate=self.grid_buy_rate,
            grid_sell_rate=self.grid_sell_rate,
        )

    def _execute_battery(self, college_action_a2: float) -> float:
        """Execute one battery step and verify dispatch consistency.

        Ensures mathematical consistency between expected dispatch from settlements
        and actual battery model step output to prevent physical constraint violations.
        """
        expected_dispatch = self.battery_model.predict_dispatch(
            college_action_a2, dt=1.0
        )
        battery_dispatch_kw = self.battery_model.step(college_action_a2, dt=1.0)

        if not math.isclose(battery_dispatch_kw, expected_dispatch, abs_tol=1e-8):
            raise RuntimeError(
                f"Battery dispatch consistency check failed: model={battery_dispatch_kw:.4f} kW, "
                f"expected (settlement)={expected_dispatch:.4f} kW."
            )
        return battery_dispatch_kw

    def _run_powerflow(
        self,
        demands_kw: dict[str, float],
        solar_kw: dict[str, float],
        battery_dispatch_kw: float,
        truncated_dict: TruncatedDict,
    ) -> tuple[PowerFlowResult | None, bool, bool, bool, TruncatedDict]:
        """Execute PandaPower power flow and check voltage/thermal violations.

        Caches the PandaPower module dynamically to optimize startup performance and avoid
        unnecessary dependency loading during bypass runs. Note that transformer_violation
        is retained here for future grid constraints integration and downstream API stability.
        """
        power_flow_result = None
        voltage_violation = False
        thermal_violation = False
        # Retained for future integration/downstream evaluation metrics and API stability
        transformer_violation = False

        # TODO: If future modules (such as Module 7) introduce additional responsibilities,
        # the PandaPower lazy-loading/lazy-import logic should be extracted into a dedicated
        # _get_pandapower() helper to keep _run_powerflow() focused.
        if not self.pandapower_bypass and not truncated_dict["__all__"]:
            if not self._pandapower_attempted:
                self._pandapower_attempted = True
                try:
                    self._pandapower = importlib.import_module("pandapower")
                except ImportError as e:
                    logger.error(
                        "PandaPower module is not installed or unavailable: %s. "
                        "Truncating episode rollout.",
                        e,
                    )

            if self._pandapower is None:
                truncated_dict["__all__"] = True
            else:
                # Dynamically retrieve LoadflowNotConverged from pandapower to maintain version compatibility
                # and avoid import-time resolution errors if the library is stubbed/mocked.
                loadflow_error = getattr(
                    self._pandapower, "LoadflowNotConverged", Exception
                )
                try:
                    update_network_loads(
                        self.net, demands_kw, solar_kw, battery_dispatch_kw
                    )

                    power_flow_result = run_power_flow(
                        self.net,
                        max_retries=self.powerflow_max_retries,
                        tolerance=self.powerflow_tolerance,
                    )

                    violations = check_constraints(power_flow_result)
                    voltage_violation = violations.voltage_violation
                    thermal_violation = violations.thermal_violation
                    transformer_violation = violations.transformer_violation

                    if (
                        violations.voltage_min_pu < VOLTAGE_CATASTROPHIC_LOW_PU
                        or violations.voltage_max_pu > VOLTAGE_CATASTROPHIC_HIGH_PU
                    ):
                        logger.warning(
                            "Catastrophic voltage violation detected! "
                            "Min: %.3f, Max: %.3f. Truncating episode rollout.",
                            violations.voltage_min_pu,
                            violations.voltage_max_pu,
                        )
                        truncated_dict["__all__"] = True

                except (PowerFlowError, loadflow_error) as e:
                    logger.error(
                        "PandaPower solver failed to converge: %s. "
                        "Truncating episode rollout.",
                        e,
                    )
                    truncated_dict["__all__"] = True

        return (
            power_flow_result,
            voltage_violation,
            thermal_violation,
            transformer_violation,
            truncated_dict,
        )

    def _build_terminal_flags(
        self,
        truncated_dict: TruncatedDict,
    ) -> tuple[bool, bool, TerminatedDict, TruncatedDict]:
        """Determine and build episode termination and truncation dictionaries.

        Ensures that normal termination boundaries are checked explicitly based on profile
        length rather than relying on IndexError exceptions.
        """
        is_terminated = (self.current_timestep + 1) >= self.episode_length or (
            self.current_timestep + 1
        ) >= len(self.episode_profiles[COLLEGE_AGENT_ID])
        is_truncated = truncated_dict["__all__"]

        terminated_dict = self._create_agent_bool_dict(is_terminated)
        terminated_dict["__all__"] = is_terminated

        # Propagate truncation to all agents
        for aid in ALL_AGENT_IDS:
            truncated_dict[aid] = is_truncated
        truncated_dict["__all__"] = is_truncated

        return is_terminated, is_truncated, terminated_dict, truncated_dict

    def _build_next_observation(
        self,
        is_terminated: bool,
        is_truncated: bool,
        demands_t: dict[str, float],
        solar_t: dict[str, float],
        timestamp_t: pd.Timestamp,
        cleaned_actions: ActionDict,
        power_flow_result: PowerFlowResult | None,
        updated_market_state: MarketState,
    ) -> ObsDict:
        """Compile actor and critic observations for the next timestep.

        Reuses the final transition state when the episode terminates or truncates to
        avoid accessing out-of-range profile indices.
        """
        # Reusing the final transition state at episode end prevents out-of-bounds profile queries
        # and eliminates exception-driven indexing logic.
        if is_terminated or is_truncated:
            demands_next = demands_t
            solar_next = solar_t
            timestamp_next = timestamp_t
        else:
            demands_next, solar_next, timestamp_next = self._get_profiles_at_timestep(
                self.current_timestep + 1
            )

        return build_observations(
            demands_kw=demands_next,
            solar_kw=solar_next,
            battery_state=self.battery_model.get_state(),
            last_actions=cleaned_actions,
            grid_result=power_flow_result,
            market_state=updated_market_state,
            timestamp=timestamp_next,
            grid_buy_rate=self.grid_buy_rate,
            grid_sell_rate=self.grid_sell_rate,
            metadata=self.episode_manager.metadata,
        )

    def _build_step_info(
        self,
        settlements: dict[str, SettlementRecord],
        voltage_violation: bool,
        thermal_violation: bool,
    ) -> InfoDict:
        """Construct the step metadata mapping for all agents."""
        info_dict: InfoDict = {}
        for aid in ALL_AGENT_IDS:
            info_dict[aid] = {
                "timestep": self.current_timestep,
                "episode_start_hour": self.episode_start_hour,
                "net_cost": float(settlements[aid].net_cost),
                "p2p_sold_kw": float(settlements[aid].p2p_sold_kw),
                "p2p_bought_kw": float(settlements[aid].p2p_bought_kw),
                "grid_sold_kw": float(settlements[aid].grid_sold_kw),
                "grid_bought_kw": float(settlements[aid].grid_bought_kw),
                "voltage_violation": voltage_violation,
                "thermal_violation": thermal_violation,
            }
        return info_dict

    # The execution order below is part of the environment contract. Future modifications
    # should preserve this ordering unless the environment semantics intentionally change.
    #
    # Environment execution order:
    #
    # 1. Validate actions
    # 2. Read current profiles
    # 3. Process market settlements
    # 4. Execute battery dispatch
    # 5. Run PandaPower
    # 6. Compute rewards
    # 7. Determine termination/truncation
    # 8. Build observations
    # 9. Return Gymnasium outputs
    def step(
        self,
        action_dict: ActionDict,
    ) -> tuple[
        ObsDict,
        RewardDict,
        TerminatedDict,
        TruncatedDict,
        InfoDict,
    ]:
        """Advance the environment by one timestep.

        Args:
            action_dict: Dict mapping active agent IDs to action vectors.

        Returns:
            RLlib return tuple containing next observations, rewards, terminations, truncations, and metadata.
        """
        # Preconditions check: timestep must be within episode limit
        if self.current_timestep >= self.episode_length:
            raise RuntimeError(
                f"step() called after episode has terminated. current_timestep={self.current_timestep}"
            )

        demands_t, solar_t, timestamp_t = self._get_profiles_at_timestep(
            self.current_timestep
        )

        cleaned_actions, truncated_dict = self._validate_actions(action_dict)

        try:
            settlements, market_state = self._process_market(
                demands_t, solar_t, cleaned_actions
            )
        except MarketClearingError as e:
            logger.warning(
                "Market clearing failed with agent actions: %s. "
                "Falling back to default safe actions.",
                e,
            )
            cleaned_actions = {
                aid: self._default_action.copy() for aid in ALL_AGENT_IDS
            }
            settlements, market_state = self._process_market(
                demands_t, solar_t, cleaned_actions
            )

        college_action_a2 = float(cleaned_actions[COLLEGE_AGENT_ID][2])
        battery_dispatch_kw = self._execute_battery(college_action_a2)

        # _transformer_violation is unpacked to keep signature stable, though not currently used in rewards or info
        (
            power_flow_result,
            voltage_violation,
            thermal_violation,
            _transformer_violation,
            truncated_dict,
        ) = self._run_powerflow(demands_t, solar_t, battery_dispatch_kw, truncated_dict)

        updated_market_state = dataclasses.replace(
            market_state,
            voltage_violation=voltage_violation,
            thermal_violation=thermal_violation,
        )

        self._check_curriculum_phase()
        reward_dict = compute_all_rewards(
            settlements=settlements,
            demands_kw=demands_t,
            solar_kw=solar_t,
            battery_dispatch_kw=battery_dispatch_kw,
            battery_soc=self.battery_model.soc,
            prev_battery_dispatch_kw=self.prev_battery_dispatch_kw,
            power_flow_result=power_flow_result,
            max_possible_cost=self.max_possible_cost,
            curriculum_phase=self.curriculum_phase,
        )

        is_terminated, is_truncated, terminated_dict, truncated_dict = (
            self._build_terminal_flags(truncated_dict)
        )

        obs_dict = self._build_next_observation(
            is_terminated=is_terminated,
            is_truncated=is_truncated,
            demands_t=demands_t,
            solar_t=solar_t,
            timestamp_t=timestamp_t,
            cleaned_actions=cleaned_actions,
            power_flow_result=power_flow_result,
            updated_market_state=updated_market_state,
        )

        self.current_timestep += 1
        self.total_env_steps += 1

        info_dict = self._build_step_info(
            settlements, voltage_violation, thermal_violation
        )

        self.last_actions = cleaned_actions
        self.prev_battery_dispatch_kw = battery_dispatch_kw

        return obs_dict, reward_dict, terminated_dict, truncated_dict, info_dict

    def close(self) -> None:
        """Clean up environment resources."""
        logger.info("Closing P2PEnergyTradingEnv.")

    def _check_curriculum_phase(self) -> None:
        """Update the curriculum phase based on cumulative environment step counts."""
        if self.total_env_steps >= self.curriculum_transition_step:
            self.curriculum_phase = 2
        else:
            self.curriculum_phase = 1
