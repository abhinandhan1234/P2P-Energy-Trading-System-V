"""Unit and Integration tests for P2PEnergyTradingEnv (Module 6).

Verifies Gymnasium interface, spaces, reset, steps, bypass, seeding,
determinism, and fail-safe recoveries.
Design reference: docs/module_6_multiagent_environment.md
"""

from __future__ import annotations

# standard library
import math
from unittest.mock import patch

# third party
import numpy as np
import pytest

# local
from p2p_energy_trading.constants import (
    ACTION_DIM,
    ACTOR_OBS_DIM,
    ALL_AGENT_IDS,
    COLLEGE_AGENT_ID,
    CRITIC_STATE_DIM,
    ActionDict,
)
from p2p_energy_trading.environment.env import P2PEnergyTradingEnv
from p2p_energy_trading.exceptions import ConfigValidationError, PowerFlowError
from p2p_energy_trading.modules.network.powerflow import PowerFlowResult


@pytest.fixture
def base_config() -> dict:
    """Base config dictionary for the environment, using bypass for fast execution."""
    return {
        "episode_length": 24,
        "pandapower_bypass": True,
        "grid_buy_rate": 8.15,
        "grid_sell_rate": 3.56,
        "data_dir": "data/processed",
        "eval_mode": True,
        "seed": 42,
    }


@pytest.fixture
def default_actions() -> ActionDict:
    """Default idle actions mapping for all agents."""
    return {aid: np.array([0.0, 0.0, 0.5], dtype=np.float32) for aid in ALL_AGENT_IDS}


@pytest.fixture
def nominal_pf_result() -> PowerFlowResult:
    """Default nominal power flow result for mocking."""
    return PowerFlowResult(
        converged=True,
        bus_vm_pu={i: 1.0 for i in range(33)},
        line_loading_pct={i: 10.0 for i in range(32)},
        trafo_loading_pct={0: 20.0},
        p_grid_kw=50.0,
    )


@pytest.fixture
def mock_violations_factory():
    """Factory to create a mocked check_constraints violations object."""
    # standard library
    from unittest.mock import MagicMock

    def _create_violations(
        voltage_violation=False,
        thermal_violation=False,
        voltage_min=1.0,
        voltage_max=1.0,
    ):
        violations = MagicMock()
        violations.voltage_violation = voltage_violation
        violations.thermal_violation = thermal_violation
        violations.transformer_violation = False
        violations.voltage_min_pu = voltage_min
        violations.voltage_max_pu = voltage_max
        return violations

    return _create_violations


class TestEnvironmentInit:
    """Verify that environment initializes correctly and validates config schema."""

    def test_spaces_and_agents(self, base_config):
        env = P2PEnergyTradingEnv(base_config)
        assert len(env._agent_ids) == 21
        assert env.observation_space.spaces["obs"].shape == (ACTOR_OBS_DIM,)
        assert env.observation_space.spaces["state"].shape == (CRITIC_STATE_DIM,)
        assert env.action_space.shape == (ACTION_DIM,)
        assert env.grid_buy_rate == 8.15
        assert env.grid_sell_rate == 3.56
        env.close()

    def test_invalid_episode_length(self, base_config):
        config = base_config.copy()
        config["episode_length"] = 12  # Below 24 limit
        with pytest.raises(ConfigValidationError, match="episode_length"):
            P2PEnergyTradingEnv(config)

    def test_invalid_grid_rates(self, base_config):
        config = base_config.copy()
        config["grid_buy_rate"] = 3.0
        config["grid_sell_rate"] = 4.0  # Buy <= Sell (arbitrage)
        with pytest.raises(ConfigValidationError, match="Configured 'grid_buy_rate'"):
            P2PEnergyTradingEnv(config)


class TestEnvironmentReset:
    """Verify the environment reset workflow."""

    def test_reset_outputs(self, base_config):
        env = P2PEnergyTradingEnv(base_config)
        obs, info = env.reset()

        assert len(obs) == 21
        assert set(obs.keys()) == set(ALL_AGENT_IDS)
        assert len(info) == 21

        for aid in ALL_AGENT_IDS:
            assert aid in obs
            assert aid in info
            assert obs[aid]["obs"].shape == (ACTOR_OBS_DIM,)
            assert obs[aid]["state"].shape == (CRITIC_STATE_DIM,)
            assert info[aid]["timestep"] == 0
            assert info[aid]["episode_start_hour"] == 0

        # In evaluation mode, reset battery SoC should be exactly 0.50
        assert env.battery_model.soc == pytest.approx(0.50)
        env.close()

    def test_stochastic_reset_soc(self, base_config):
        config = base_config.copy()
        config["eval_mode"] = False  # Training mode enables random SoC
        env = P2PEnergyTradingEnv(config)

        # Track initial SoCs across resets to confirm they vary in [0.3, 0.7]
        socs = []
        for seed in range(5):
            env.reset(seed=seed)
            soc = env.battery_model.soc
            assert 0.3 <= soc <= 0.7
            socs.append(soc)

        # Verify they are not all identical
        assert len(set(socs)) > 1
        env.close()


class TestEnvironmentStep:
    """Verify environment step progression and outputs."""

    def test_nominal_step(self, base_config, default_actions):
        env = P2PEnergyTradingEnv(base_config)
        env.reset()

        # Step actions matching idle state
        obs, rew, term, trunc, info = env.step(default_actions)

        assert env.current_timestep == 1
        assert len(obs) == 21
        assert set(obs.keys()) == set(ALL_AGENT_IDS)
        assert len(rew) == 21
        assert len(term) == 22
        assert len(trunc) == 22
        assert len(info) == 21

        # Check all terminated/truncated are False
        assert not term["__all__"]
        assert not trunc["__all__"]

        for aid in ALL_AGENT_IDS:
            assert not term[aid]
            assert not trunc[aid]
            assert isinstance(rew[aid], float)
            assert -10.0 <= rew[aid] <= 10.0
        env.close()

    def test_episode_termination(self, base_config, default_actions):
        env = P2PEnergyTradingEnv(base_config)
        env.reset()

        # Step through all 24 hours of base config episode length
        for _ in range(23):
            obs, rew, term, trunc, info = env.step(default_actions)
            assert not term["__all__"]

        # Final 24th step
        obs, rew, term, trunc, info = env.step(default_actions)
        assert term["__all__"]
        for aid in ALL_AGENT_IDS:
            assert term[aid]
        env.close()


class TestEnvironmentSeedingAndDeterminism:
    """Verify environment seeding and trajectory reproducibility."""

    def test_determinism(self, base_config):
        env1 = P2PEnergyTradingEnv(base_config)
        env2 = P2PEnergyTradingEnv(base_config)

        # Run resets with same seed
        obs1, info1 = env1.reset(seed=101)
        obs2, info2 = env2.reset(seed=101)

        for aid in ALL_AGENT_IDS:
            np.testing.assert_allclose(obs1[aid]["obs"], obs2[aid]["obs"])
            np.testing.assert_allclose(obs1[aid]["state"], obs2[aid]["state"])

        actions = {
            aid: np.array([0.2, 0.3, 0.6], dtype=np.float32) for aid in ALL_AGENT_IDS
        }

        # Compare 5 sequential steps
        for _ in range(5):
            obs1, rew1, term1, trunc1, info1 = env1.step(actions)
            obs2, rew2, term2, trunc2, info2 = env2.step(actions)

            for aid in ALL_AGENT_IDS:
                np.testing.assert_allclose(obs1[aid]["obs"], obs2[aid]["obs"])
                np.testing.assert_allclose(obs1[aid]["state"], obs2[aid]["state"])
                assert rew1[aid] == pytest.approx(rew2[aid])
                assert term1[aid] == term2[aid]
                assert trunc1[aid] == trunc2[aid]

        env1.close()
        env2.close()


class TestEnvironmentFailureScenarios:
    """Verify fail-safe mechanisms for invalid actions and network failures."""

    def test_action_nan_recovery(self, base_config, default_actions):
        env = P2PEnergyTradingEnv(base_config)
        env.reset()

        actions = default_actions.copy()
        # Inject NaN action for College
        actions[COLLEGE_AGENT_ID] = np.array([np.nan, 0.0, 0.5], dtype=np.float32)

        # Step should recover using default safe action and not crash
        obs, rew, term, trunc, info = env.step(actions)
        assert env.nan_episode_count == 1
        assert not trunc["__all__"]
        env.close()

    def test_excessive_nan_truncation(self, base_config, default_actions):
        config = base_config.copy()
        config["max_nan_actions"] = 2
        env = P2PEnergyTradingEnv(config)
        env.reset()

        actions = default_actions.copy()
        # Inject 3 NaN actions
        actions["solar_01"] = np.array([np.nan, 0.0, 0.5], dtype=np.float32)
        actions["solar_02"] = np.array([0.0, np.nan, 0.5], dtype=np.float32)
        actions["solar_03"] = np.array([0.0, 0.0, np.nan], dtype=np.float32)

        obs, rew, term, trunc, info = env.step(actions)
        assert trunc["__all__"]
        for aid in ALL_AGENT_IDS:
            assert trunc[aid]
        env.close()

    @patch("p2p_energy_trading.environment.env.run_power_flow")
    def test_pandapower_divergence_truncation(
        self, mock_pf, base_config, default_actions
    ):
        # Disable bypass so power flow is run
        config = base_config.copy()
        config["pandapower_bypass"] = False

        env = P2PEnergyTradingEnv(config)
        env.reset()

        # Mock powerflow run to raise convergence exception
        mock_pf.side_effect = PowerFlowError("Diverged")

        obs, rew, term, trunc, info = env.step(default_actions)

        # Convergence failure must trigger episode truncation
        assert trunc["__all__"]
        for aid in ALL_AGENT_IDS:
            assert trunc[aid]
        env.close()

    def test_catastrophic_voltage_truncation(self, base_config, default_actions):
        # Disable bypass
        config = base_config.copy()
        config["pandapower_bypass"] = False

        env = P2PEnergyTradingEnv(config)
        env.reset()

        # Create a mock PowerFlowResult with extreme voltage of 1.25 p.u. (limit is 1.20)
        extreme_pf_result = PowerFlowResult(
            converged=True,
            bus_vm_pu={i: 1.25 for i in range(33)},
            line_loading_pct={i: 10.0 for i in range(32)},
            trafo_loading_pct={0: 20.0},
            p_grid_kw=50.0,
        )

        with patch(
            "p2p_energy_trading.environment.env.run_power_flow",
            return_value=extreme_pf_result,
        ):
            obs, rew, term, trunc, info = env.step(default_actions)

            # Catastrophic voltage must trigger truncation
            assert trunc["__all__"]
        env.close()

    def test_battery_consistency_runtime_error(self, base_config, default_actions):
        env = P2PEnergyTradingEnv(base_config)
        env.reset()

        # Force a consistency failure by mocking battery step to return a divergent value
        with patch.object(env.battery_model, "step", return_value=100.0):
            with pytest.raises(
                RuntimeError, match="Battery dispatch consistency check failed"
            ):
                env.step(default_actions)
        env.close()

    def test_defensive_bounds_checking(self, base_config, default_actions):
        env = P2PEnergyTradingEnv(base_config)
        env.reset()

        # Simulate profile exhaustion by making episode_length larger than profile length
        profile_len = len(env.episode_profiles[COLLEGE_AGENT_ID])
        env.episode_length = profile_len + 10
        # Set current_timestep to the last valid index, so the next step increments out of bounds
        env.current_timestep = profile_len - 1

        obs, rew, term, trunc, info = env.step(default_actions)
        assert term["__all__"]
        env.close()


class TestFallbackMultiAgentEnv:
    """Verify fallback MultiAgentEnv class attributes and backward compatibility."""

    def test_fallback_attributes(self, base_config):
        env = P2PEnergyTradingEnv(base_config)

        assert hasattr(env, "possible_agents")
        assert hasattr(env, "agents")
        assert hasattr(env, "observation_spaces")
        assert hasattr(env, "action_spaces")

        assert isinstance(env.possible_agents, list)
        assert len(env.possible_agents) == 21
        assert "college" in env.possible_agents

        assert isinstance(env.agents, list)
        assert len(env.agents) == 21

        assert isinstance(env.observation_spaces, dict)
        assert len(env.observation_spaces) == 21

        assert isinstance(env.action_spaces, dict)
        assert len(env.action_spaces) == 21

    def test_fallback_subclass_defined_spaces(self):
        """Verify that subclass-defined observation_spaces and action_spaces are preserved."""
        # local
        from p2p_energy_trading.environment.compatibility import MultiAgentEnv

        class SubEnv(MultiAgentEnv):
            def __init__(self):
                self.observation_spaces = {"agent_1": "custom_obs"}
                self.action_spaces = {"agent_1": "custom_act"}
                super().__init__()

        sub = SubEnv()
        assert sub.observation_spaces == {"agent_1": "custom_obs"}
        assert sub.action_spaces == {"agent_1": "custom_act"}


class TestPass3EngineeringCorrections:
    """Verify Pass 3 and Pass 4 engineering corrections for Battery, Terminal Obs, Lazy Import, and Caching."""

    def test_step_delegates_to_predict_dispatch(self):
        """Verify BatteryModel.step() internally calls predict_dispatch() exactly once,
        updates SoC correctly, and returns expected dispatch."""
        # local
        from p2p_energy_trading.modules.network.battery import BatteryModel

        battery = BatteryModel(initial_soc=0.5)

        initial_soc = battery.soc
        socs_during_predict = []

        # Save original predict_dispatch method to call inside the spy
        original_predict = battery.predict_dispatch

        def spy_predict(action_charge_fraction, dt=1.0):
            socs_during_predict.append(battery.soc)
            # Return a controlled mock value (e.g. 50.0) instead of actual prediction
            # to make sure step() uses exactly the return value of predict_dispatch
            return 50.0

        with patch.object(
            battery, "predict_dispatch", side_effect=spy_predict
        ) as mock_predict:
            disp = battery.step(action_charge_fraction=0.1, dt=1.0)

            # 1. predict_dispatch is called exactly once by step()
            mock_predict.assert_called_once_with(0.1, 1.0)

            # 2. step() performs all SoC updates after receiving the predicted dispatch
            # (i.e. during the predict call, SoC is still at initial value)
            assert socs_during_predict == [initial_soc]
            # Verify that SoC is indeed updated afterwards
            expected_soc_after = initial_soc - (50.0 / (math.sqrt(0.90) * 1.0)) / 500.0
            assert battery.soc == pytest.approx(expected_soc_after)

            # 3. The dispatch returned by step() is identical to the value returned by predict_dispatch()
            assert disp == 50.0

    def test_battery_predict_dispatch_scenarios(self):
        """Verify predict_dispatch() matches step() output across charging, discharging, idle, clipping, and thresholds."""
        # local
        from p2p_energy_trading.modules.network.battery import BatteryModel

        scenarios = [0.0, 0.1, 0.5, 0.9, 1.0, -0.2, 1.2, 0.49]
        for action in scenarios:
            for init_soc in [0.3, 0.5, 0.8]:
                b_step = BatteryModel(initial_soc=init_soc)
                expected_disp = b_step.step(action, dt=1.0)

                b_predict = BatteryModel(initial_soc=init_soc)
                pred_disp = b_predict.predict_dispatch(action, dt=1.0)

                assert math.isclose(pred_disp, expected_disp, abs_tol=1e-8), (
                    f"Divergence for action {action} at SoC {init_soc}: predict={pred_disp}, step={expected_disp}"
                )

    def test_terminal_observation_never_requests_invalid_profile(
        self, base_config, default_actions
    ):
        """Verify no IndexError occurs at episode termination, no invalid dataframe access, and clean termination."""
        env = P2PEnergyTradingEnv(base_config)
        env.reset()

        for step_idx in range(23):
            obs, rew, term, trunc, info = env.step(default_actions)
            assert not term["__all__"]
            assert not trunc["__all__"]

        with patch.object(
            env, "_get_profiles_at_timestep", wraps=env._get_profiles_at_timestep
        ) as mock_lookup:
            obs, rew, term, trunc, info = env.step(default_actions)
            assert term["__all__"]
            for call in mock_lookup.call_args_list:
                args, kwargs = call
                queried_t = args[0]
                assert queried_t < len(env.episode_profiles[COLLEGE_AGENT_ID]), (
                    f"Queried out of bounds index {queried_t}"
                )
        env.close()

    def test_pandapower_lazy_import_cache(
        self, base_config, default_actions, nominal_pf_result, mock_violations_factory
    ):
        """Verify PandaPower is imported exactly once and multiple step() calls reuse the cached module."""
        # standard library
        import importlib

        config = base_config.copy()
        config["pandapower_bypass"] = False

        env = P2PEnergyTradingEnv(config)
        env.reset()

        with (
            patch(
                "p2p_energy_trading.environment.env.run_power_flow",
                return_value=nominal_pf_result,
            ),
            patch(
                "p2p_energy_trading.environment.env.check_constraints"
            ) as mock_constraints,
        ):
            mock_constraints.return_value = mock_violations_factory()

            with patch(
                "importlib.import_module", wraps=importlib.import_module
            ) as mock_import:
                env.step(default_actions)
                env.step(default_actions)

                pp_imports = [
                    call
                    for call in mock_import.call_args_list
                    if call[0][0] == "pandapower"
                ]
                assert len(pp_imports) == 1, (
                    f"pandapower imported {len(pp_imports)} times instead of exactly 1"
                )
        env.close()

    def test_failed_pandapower_import_is_cached(self, base_config, default_actions):
        """Verify that a failed PandaPower import caches failure state and is not retried."""
        # standard library
        import importlib

        config = base_config.copy()
        config["pandapower_bypass"] = False

        env = P2PEnergyTradingEnv(config)
        env.reset()

        def mock_import_fn(name, package=None):
            if name == "pandapower":
                raise ImportError("Failed mock import")
            return importlib.import_module(name, package)

        with patch(
            "importlib.import_module", side_effect=mock_import_fn
        ) as mock_import:
            # First step: fails import, truncates episode
            obs, rew, term, trunc, info = env.step(default_actions)
            assert trunc["__all__"]

            # Second step: should not invoke importlib.import_module("pandapower") again
            env.current_timestep = 0
            env.step(default_actions)

            # Count imports of "pandapower"
            pp_imports = [
                call
                for call in mock_import.call_args_list
                if call[0][0] == "pandapower"
            ]
            assert len(pp_imports) == 1, (
                f"Import was attempted {len(pp_imports)} times instead of exactly once."
            )

        env.close()

    def test_reset_clears_pandapower_import_cache(
        self, base_config, default_actions, nominal_pf_result, mock_violations_factory
    ):
        """Verify that resetting the environment clears the PandaPower import cache, allowing retry."""
        # standard library
        import importlib

        config = base_config.copy()
        config["pandapower_bypass"] = False

        env = P2PEnergyTradingEnv(config)
        env.reset()

        # 1. First rollout: Mock import to fail
        def mock_import_fail(name, package=None):
            if name == "pandapower":
                raise ImportError("Failed mock import")
            return importlib.import_module(name, package)

        with patch(
            "importlib.import_module", side_effect=mock_import_fail
        ) as mock_import:
            obs, rew, term, trunc, info = env.step(default_actions)
            assert trunc["__all__"]
            pp_imports = [
                call
                for call in mock_import.call_args_list
                if call[0][0] == "pandapower"
            ]
            assert len(pp_imports) == 1

        # 2. Call reset()
        env.reset()

        # 3. Second rollout: Mock import to succeed
        with (
            patch(
                "p2p_energy_trading.environment.env.run_power_flow",
                return_value=nominal_pf_result,
            ),
            patch(
                "p2p_energy_trading.environment.env.check_constraints"
            ) as mock_constraints,
        ):
            mock_constraints.return_value = mock_violations_factory()

            with patch(
                "importlib.import_module", wraps=importlib.import_module
            ) as mock_import_success:
                obs, rew, term, trunc, info = env.step(default_actions)
                assert not trunc["__all__"]

                pp_imports_success = [
                    call
                    for call in mock_import_success.call_args_list
                    if call[0][0] == "pandapower"
                ]
                assert len(pp_imports_success) == 1

        env.close()

    def test_cached_market_state_copy(self, base_config):
        """Verify modifying the returned MarketState after reset() does not alter the cached default object."""
        # standard library
        import dataclasses
        from unittest.mock import patch

        env = P2PEnergyTradingEnv(base_config)

        # Verify that reset() uses a copy of the default market state
        with patch(
            "p2p_energy_trading.environment.env.build_observations"
        ) as mock_build_obs:
            mock_build_obs.return_value = {}
            env.reset()
            assert mock_build_obs.called
            passed_market = mock_build_obs.call_args[1]["market_state"]
            assert passed_market is not env._default_market

            # Verify that modifying the passed state doesn't affect the cached default
            modified_market = dataclasses.replace(
                passed_market, p2p_clearing_price=999.0
            )
            assert env._default_market.p2p_clearing_price != 999.0

        env.close()

    def test_validate_actions_missing_agent_injected(
        self, base_config, default_actions
    ):
        """Verify that omitting an agent action results in ActionHandler injecting the default safe action, executing step successfully, and preserving normal rollout execution (no premature termination or truncation)."""
        env = P2PEnergyTradingEnv(base_config)
        env.reset()

        incomplete_actions = default_actions.copy()
        assert COLLEGE_AGENT_ID in incomplete_actions
        del incomplete_actions[COLLEGE_AGENT_ID]

        obs, rew, term, trunc, info = env.step(incomplete_actions)

        assert env.current_timestep == 1
        assert COLLEGE_AGENT_ID in env.last_actions
        np.testing.assert_allclose(
            env.last_actions[COLLEGE_AGENT_ID], env._default_action
        )
        assert not term["__all__"]
        assert not trunc["__all__"]
        env.close()

    def test_terminal_step_prevents_out_of_bounds(self, base_config, default_actions):
        """Verify that stepping at the end of the episode does not cause an IndexError, terminates correctly, and does not query out-of-bounds profile indices."""
        env = P2PEnergyTradingEnv(base_config)
        env.reset()

        # Step through the entire episode until the final transition
        for _ in range(env.episode_length - 1):
            env.step(default_actions)

        profile_len = len(env.episode_profiles[COLLEGE_AGENT_ID])

        with patch.object(
            env, "_get_profiles_at_timestep", wraps=env._get_profiles_at_timestep
        ) as mock_get_profiles:
            obs, rew, term, trunc, info = env.step(default_actions)

            # Verify termination and valid observations
            assert term["__all__"]
            assert len(obs) == len(ALL_AGENT_IDS)
            assert set(obs.keys()) == set(ALL_AGENT_IDS)

            # Ensure no out-of-bounds queries were made to profiles
            for call in mock_get_profiles.call_args_list:
                args, kwargs = call
                queried_t = args[0]
                assert queried_t < profile_len, (
                    f"Queried out of bounds index {queried_t}"
                )
        env.close()

    def test_truncation_step_prevents_out_of_bounds(self, base_config, default_actions):
        """Verify that triggering environment truncation via the public interface (excessive NaN actions) truncates cleanly, returns valid observations, and prevents out-of-bounds profile queries."""
        config = base_config.copy()
        config["max_nan_actions"] = 0
        env = P2PEnergyTradingEnv(config)
        env.reset()

        # Inject one NaN action to trigger truncation immediately
        nan_actions = default_actions.copy()
        nan_actions[COLLEGE_AGENT_ID] = np.array([np.nan, 0.0, 0.5], dtype=np.float32)

        profile_len = len(env.episode_profiles[COLLEGE_AGENT_ID])

        with patch.object(
            env, "_get_profiles_at_timestep", wraps=env._get_profiles_at_timestep
        ) as mock_get_profiles:
            obs, rew, term, trunc, info = env.step(nan_actions)

            # Verify clean truncation and valid observations
            assert trunc["__all__"]
            assert len(obs) == len(ALL_AGENT_IDS)
            assert set(obs.keys()) == set(ALL_AGENT_IDS)

            # Ensure no out-of-bounds queries were made to profiles
            for call in mock_get_profiles.call_args_list:
                args, kwargs = call
                queried_t = args[0]
                assert queried_t < profile_len, (
                    f"Queried out of bounds index {queried_t}"
                )
        env.close()
