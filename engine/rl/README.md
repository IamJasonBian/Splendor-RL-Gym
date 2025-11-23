# Splendor RL Agent Visualization System

## Overview

This workspace contains a complete visualization architecture for building and analyzing a Splendor reinforcement learning agent. The system is based on the reward model from [SplendorSimulator](https://github.com/dsmiller95/SplendorSimulator) and designed to support both local development and distributed online training.

## What's Included

### 1. Core Components (`src/rl_models.py`)

**Data Models:**
- `AgentObservation` - Complete game state representation
- `Action` - Agent actions (buy card, take gems, reserve)
- `PolicyOutput` - Policy network decisions with interpretability
- `Transition` - Single timestep in episode trajectory
- `RewardComponents` - Decomposed reward signal
- `EpisodeSummary` - Episode-level statistics

**State Analyzers:**
- `EfficiencyAnalyzer` - Points/turn, card acquisition rate, gem waste
- `StrategicPositionAnalyzer` - Win probability, resource advantage, turns-to-win
- `ActionPatternAnalyzer` - Action entropy, buy/take ratios, streak detection
- `ValueFunctionAnalyzer` - Value prediction accuracy tracking
- `CardSynergyAnalyzer` - Color diversity, bonus concentration, affordable cards
- `AnalyzerOrchestrator` - Coordinates multiple analyzers

### 2. Documentation

**`docs/visualization_architecture.md`** - Comprehensive architecture guide covering:
- Complete data model specifications
- Local vs online pipeline designs
- Event streaming protocols
- Technology stack recommendations
- Implementation roadmap
- Sample visualizations

### 3. Demo Scripts

**`demo_analyzers.py`** - Basic analyzer demonstration
```bash
python demo_analyzers.py
```
Shows all 5 analyzers working on a mid-game state.

**`demo_game_simulation.py`** - Game progression simulation
```bash
python demo_game_simulation.py
```
Simulates a game from start to near-victory, tracking metrics over time with ASCII visualizations.

**`demo_reward_model.py`** - Reward function demonstration
```bash
python demo_reward_model.py
```
Shows the 5-component reward system (matching SplendorSimulator) across different scenarios:
- Taking gems
- Buying cards
- Winning/losing
- Different reward configurations

## Reward Model Design

Based on SplendorSimulator's proven approach, the reward function has **5 configurable components**:

| Component | Purpose | Default Weight |
|-----------|---------|----------------|
| **Tokens Held** | Encourages gem collection | 0.1 |
| **Cards Held** | Values permanent bonuses | 0.5 |
| **Points Gained** | Directly rewards scoring | 1.0 |
| **Win/Lose** | Terminal outcome bonus | ±100.0 |
| **Game Length** | Penalty for longer games | -0.05/turn |

Each component can be toggled on/off to support **curriculum learning** strategies.

## Running the Demos

All demos work out of the box:

```bash
# Basic analyzer test
python demo_analyzers.py

# Game progression visualization
python demo_game_simulation.py

# Reward model scenarios
python demo_reward_model.py
```

## Sample Output

### Efficiency Analyzer
```
[EFFICIENCY]
  points_per_turn: 0.800
  cards_per_turn: 0.400
  bonus_efficiency: 0.800
  gem_waste: 0
```

### Strategic Position Analyzer
```
[STRATEGIC_POSITION]
  point_lead: 2
  bonus_advantage: 0
  resource_advantage: 14.000
  victory_probability: 0.661
  turns_to_win_estimate: 4.667
```

### Reward Decomposition
```
Buying a 2-point card:
  tokens_held         : +0.30
  cards_held          : +1.00
  points_gained       : +2.00
  win_lose            : +0.00
  game_length_penalty : -0.40
  TOTAL               : +2.90
```

## Architecture Highlights

### Local Development Pipeline
```
Agent Process → WebSocket Events → Browser UI
    ↓
Real-time visualization:
- Game board renderer
- Policy inspector (action probabilities)
- Reward graphs (time series)
- State analyzer dashboard
```

### Online Training Pipeline
```
Rollout Workers → Replay Buffer → Learner → Checkpoints
                       ↓
                 Metrics Queue
                       ↓
            Time-Series DB → Dashboards
```

## Key Design Principles

1. **Separation of Concerns** - Agent logic decoupled from visualization
2. **Composable Analyzers** - Independent modules that can be mixed/matched
3. **Event-Driven** - Async processing enables real-time updates
4. **Progressive Enhancement** - Start local, scale to distributed
5. **Reproducibility** - Episode IDs enable replay from storage

## Visualization Tradeoffs

| Aspect | Local (Dev) | Online (Training) |
|--------|-------------|-------------------|
| **Latency** | <50ms (critical) | <500ms (acceptable) |
| **Storage** | In-memory (GB) | Persistent (TB+) |
| **Visualization** | Interactive stepping | Static dashboards |
| **Infrastructure** | Single machine | Distributed cluster |
| **Use Case** | Debugging, prototyping | Production training |

## Next Steps

To build a complete RL agent:

1. **Gym Environment** - Wrap existing `State` class into OpenAI Gym interface
2. **Policy Network** - Implement MLP or attention-based architecture
3. **Local Visualization** - Build FastAPI + React UI using event pipeline
4. **Training Integration** - Connect to Ray RLlib or Stable-Baselines3
5. **Online Infrastructure** - Deploy distributed rollout workers
6. **Dashboard** - Set up Grafana with InfluxDB for metrics

## Technology Recommendations

**Local Development:**
- Backend: FastAPI (WebSocket support)
- Frontend: React + Canvas (game board) + Recharts (graphs)
- Data Format: MessagePack (faster than JSON)

**Online Training:**
- Orchestration: Ray (distributed RL)
- Metrics: InfluxDB + Grafana
- Storage: S3 (checkpoints) + PostgreSQL (metadata)
- Streaming: Kafka or Redis Streams

## File Structure

```
.
├── src/
│   ├── rl_models.py          # Core data models & analyzers
│   ├── solver.py             # Existing BFS solver
│   ├── gems.py               # Gem mechanics
│   └── cardparser.py         # Card data loading
├── docs/
│   └── visualization_architecture.md  # Full architecture spec
├── demo_analyzers.py         # Basic analyzer demo
├── demo_game_simulation.py   # Game progression demo
└── demo_reward_model.py      # Reward function demo
```

## References

- **SplendorSimulator**: https://github.com/dsmiller95/SplendorSimulator
  - Original reward model implementation
  - Proven RL training approach
- **This Codebase**: BFS solver for optimal Splendor moves
  - Can be used as baseline/evaluation metric
  - Existing `State` class is foundation for Gym environment

## Questions?

The architecture is designed to be flexible and scalable. Start with local development using the demos provided, then expand to distributed training as needed.

Key insight: The analyzers and reward model are **production-ready** - they work with your existing codebase and can be integrated into any RL framework.
