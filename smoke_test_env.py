import numpy as np
import pytest
from p2p_energy_trading.environment.env import P2PEnergyTradingEnv
from p2p_energy_trading.constants import ALL_AGENT_IDS, COLLEGE_AGENT_ID

# 1. Initialize environment with basic config (without pandapower bypass to test solver)
config = {
    "episode_length": 168,
    "pandapower_bypass": False,
    "grid_buy_rate": 8.15,
    "grid_sell_rate": 3.56,
    "data_dir": "data/processed",
    "eval_mode": True,
    "seed": 42,
}

print("Initializing environment...")
env = P2PEnergyTradingEnv(config)

# 2. Reset the environment
print("Resetting environment...")
obs, info = env.reset(seed=42)

# Verify initial states
assert len(obs) == 21, f"Expected 21 agent observations, got {len(obs)}"
assert env.battery_model.soc == 0.50, f"Expected initial battery SoC to be 0.50, got {env.battery_model.soc}"

print("Running 1000 random action steps...")
for step in range(1, 1001):
    # Generate random actions (Box spaces of shape (3,) clipped in [0.0, 1.0])
    actions = {}
    for aid in ALL_AGENT_IDS:
        actions[aid] = np.random.uniform(0.0, 1.0, size=(3,)).astype(np.float32)
    
    # Step the environment
    obs, rewards, terminated, truncated, info = env.step(actions)
    
    # Check for NaNs and non-finite values
    for aid in ALL_AGENT_IDS:
        assert not np.isnan(obs[aid]["obs"]).any(), f"NaN found in actor observation for {aid} at step {step}"
        assert not np.isnan(obs[aid]["state"]).any(), f"NaN found in critic state for {aid} at step {step}"
        assert np.isfinite(rewards[aid]), f"Reward for {aid} at step {step} is not finite: {rewards[aid]}"
        
    # Check battery SoC boundaries [0.10, 0.95]
    soc = env.battery_model.soc
    assert 0.10 <= soc <= 0.95, f"Battery SoC out of bounds: {soc} at step {step}"
    
    # Check market state clearing
    college_info = info[COLLEGE_AGENT_ID]
    assert "p2p_bought_kw" in college_info, f"Market clearing info missing at step {step}"
    
    # Check if the step terminated or truncated prematurely (due to power flow failure etc.)
    if terminated["__all__"] or truncated["__all__"]:
        print(f"Episode ended at step {step}. Resetting environment...")
        obs, info = env.reset()

print("\n✓ Environment Smoke Test PASSED successfully with zero errors!")

# 3. Close the environment
env.close()
print("Environment closed.")
