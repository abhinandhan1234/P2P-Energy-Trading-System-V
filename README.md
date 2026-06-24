# P2P MARL Energy Trading System

A Peer-to-Peer Multi-Agent Reinforcement Learning energy trading system for a college campus microgrid. 21 autonomous agents trade energy while respecting physical grid constraints through PandaPower simulation, trained using MAPPO with centralized training and decentralized execution (CTDE).

## Features

- **21 Agents**: 1 College Building, 15 Solar Buildings, 5 Consumer Buildings
- **MAPPO with CTDE**: Centralized critic (243-dim), decentralized actors (23-dim)
- **3 Shared Policies**: College, Solar, Consumer with parameter sharing
- **IEEE 33-Bus Network**: PandaPower-validated power flow
- **Uniform Clearing Market**: Quantity-only P2P trading with pro-rata allocation
- **College Battery**: 500 kWh / 250 kW with SoC management
- **3-Stage Curriculum**: Debug → Training → Constraint-aware
- **Full Evaluation Suite**: 4 research questions, 3 baselines, 5 ablation studies

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Data Preparation

```bash
python -m p2p_energy_trading.modules.profile_generator.run
```

## Training

```bash
python -m p2p_energy_trading.training.train --config config/training_config.yaml
```

## Evaluation

```bash
python -m p2p_energy_trading.evaluation.evaluate --checkpoint checkpoints/best_model/
```

## License

MIT License
