# P2P Energy Trading System - Evaluation Framework Summary Report

This report documents performance benchmarks for MAPPO policies against non-learning baselines.

## Performance Metrics Overview

| Experiment | Mean Cost (₹) | Cost Red. vs Grid (%) | Voltage Violations (%) | P2P Utilisation (%) |
| :--- | :--- | :--- | :--- | :--- |
| Trained | 301994.82 ± 0.00 | 0.00% | 4.586% | 1.05% |

## Statistical Significance Analysis

 Welch's t-test comparing the trained policy against other baseline policies on total cost (alpha = 0.05):


## Verification Success Thresholds

- **Cost reduction (>= 10%)**: 0.00% - FAILED
- **Voltage Safety (< 1%)**: 4.586% - FAILED
- **Thermal Safety (< 1%)**: 0.000% - PASSED
- **P2P Utilisation (> 60%)**: 1.05% - FAILED
