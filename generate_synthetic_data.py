"""
Generate synthetic raw data CSV for P2P Energy Trading System.

Creates a realistic 3-year hourly dataset mimicking a college campus
in South India (Dharwad/Karnataka) with:
  - Seasonal solar irradiance patterns
  - Realistic campus demand with day/night and weekday/weekend variation
  - Columns matching exactly: Timestamp, Campus_Demand_kW, College_Solar_kW
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

# 3 years of hourly data: 2021-01-01 to 2023-12-31
start = pd.Timestamp("2021-01-01 00:00:00")
end   = pd.Timestamp("2023-12-31 23:00:00")
timestamps = pd.date_range(start=start, end=end, freq="h")
N = len(timestamps)

print(f"Generating {N} hourly samples from {start.date()} to {end.date()}...")

hours   = timestamps.hour.values
months  = timestamps.month.values
dows    = timestamps.dayofweek.values  # 0=Mon … 6=Sun
is_wknd = (dows >= 5).astype(float)

# ── Campus Demand (kW) ──────────────────────────────────────────────────────
# Base load ~800 kW; peak morning (8-10) and evening (18-21)
hour_demand = np.array([
    0.55, 0.50, 0.48, 0.47, 0.48, 0.52,
    0.65, 0.80, 0.95, 1.00, 0.98, 0.92,
    0.90, 0.92, 0.93, 0.94, 0.95, 1.00,
    0.98, 0.95, 0.88, 0.78, 0.68, 0.60,
])  # multiplier by hour (peak=1.0 at hour 9)

BASE_DEMAND_KW = 820.0
demand = BASE_DEMAND_KW * hour_demand[hours]

# Seasonal: slightly higher in summer (Mar-Jun) due to cooling
season_factor = 1.0 + 0.12 * np.sin(2 * np.pi * (months - 3) / 12)
demand *= season_factor

# Weekend reduction (fewer people on campus)
demand *= (1.0 - 0.25 * is_wknd)

# Gaussian noise ±5%
demand += rng.normal(0, 0.05 * demand)
demand = np.clip(demand, 50.0, 1800.0)  # physical bounds

# ── College Solar Generation (kW) ──────────────────────────────────────────
# Peak capacity ~500 kW; follows a smooth bell curve from sunrise to sunset
# Sunrise ~06:00, sunset ~18:30 in Dharwad; peak ~12:00-13:00
def solar_bell(h: np.ndarray) -> np.ndarray:
    """Fractional solar output (0-1) as function of hour."""
    peak_hour = 12.5
    width = 4.5  # std dev in hours
    val = np.exp(-0.5 * ((h - peak_hour) / width) ** 2)
    val[(h < 6) | (h > 19)] = 0.0
    return val

SOLAR_CAPACITY_KW = 480.0
solar_fraction = solar_bell(hours.astype(float))

# Seasonal irradiance: higher in summer (Mar-May), lower in monsoon (Jun-Sep)
irr_seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * (months - 4) / 12)
# Monsoon cloud-cover dip (Jun-Sep ≈ months 6-9)
monsoon_mask = (months >= 6) & (months <= 9)
irr_seasonal[monsoon_mask] *= 0.70

solar = SOLAR_CAPACITY_KW * solar_fraction * irr_seasonal

# Add cloud-cover noise (log-normal multiplicative, only when solar > 0)
cloud_noise = rng.lognormal(mean=0.0, sigma=0.15, size=N)
solar = solar * cloud_noise
solar[solar_fraction == 0] = 0.0           # force 0 at night
solar = np.clip(solar, 0.0, SOLAR_CAPACITY_KW * 1.05)

# ── Build DataFrame and write CSV ──────────────────────────────────────────
df = pd.DataFrame({
    "Timestamp":        timestamps.strftime("%Y-%m-%d %H:%M:%S"),
    "Campus_Demand_kW": np.round(demand, 4),
    "College_Solar_kW": np.round(solar,  4),
})

out_dir = Path("data/raw")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "kls_vdit_hourly_market.csv"
df.to_csv(out_path, index=False)

print(f"✓ Written {len(df)} rows → {out_path}")
print(f"  Demand : min={df['Campus_Demand_kW'].min():.1f}  "
      f"max={df['Campus_Demand_kW'].max():.1f}  "
      f"mean={df['Campus_Demand_kW'].mean():.1f} kW")
print(f"  Solar  : min={df['College_Solar_kW'].min():.1f}  "
      f"max={df['College_Solar_kW'].max():.1f}  "
      f"mean={df['College_Solar_kW'].mean():.1f} kW")
