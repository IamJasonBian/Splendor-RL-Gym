# Splendor RL Agent Visualization Architecture

## Overview

This document describes the data model, pipeline, and state analyzers for visualizing a Splendor RL agent both locally (real-time development) and online (distributed training).

---

## 1. Data Model

### 1.1 Core Data Structures

```python
# Agent Observation (input to policy)
@dataclass
class AgentObservation:
    # Player state
    gems: Gems  # tuple[int, int, int, int, int]
    bonuses: Gems  # permanent gem production from cards
    points: int
    cards_owned: tuple[int, ...]  # sorted card indices
    reserved_cards: tuple[int, ...] # up to 3 cards

    # Opponent state (for 2-player game)
    opponent_gems_total: int  # hide exact distribution
    opponent_bonuses: Gems
    opponent_points: int
    opponent_cards_count: int

    # Board state
    available_cards_tier1: tuple[int, ...]  # 4 visible cards
    available_cards_tier2: tuple[int, ...]
    available_cards_tier3: tuple[int, ...]
    gem_pool: Gems  # remaining gems in pool

    # Meta
    turn_number: int
    current_player: int  # 0 or 1


# Policy Output (agent decision)
@dataclass
class PolicyOutput:
    action: Action
    action_logits: dict[Action, float]  # all valid actions + scores
    value_estimate: float  # V(s) - expected future return
    policy_entropy: float  # exploration metric

    # Attention/importance (optional, for interpretability)
    feature_importance: dict[str, float]  # which features mattered


# Agent Action (output from policy)
@dataclass
class Action:
    action_type: Literal['buy', 'take_gems', 'reserve']

    # Polymorphic based on action_type
    card_id: int | None = None  # for buy/reserve
    gem_pattern: Gems | None = None  # for take_gems
    gems_to_return: Gems | None = None  # if exceeding 10 gems


# Episode Trajectory (for training & replay)
@dataclass
class Transition:
    observation: AgentObservation
    action: Action
    reward: float
    next_observation: AgentObservation
    done: bool
    info: dict  # auxiliary data


# Reward Decomposition (for analysis)
@dataclass
class RewardComponents:
    tokens_held: float
    cards_held: float
    points_gained: float
    win_lose: float
    game_length_penalty: float
    total: float

    # Additional tracking
    breakdown_weights: dict[str, float]  # α₁, α₂, etc.


# Episode Summary (aggregate statistics)
@dataclass
class EpisodeSummary:
    episode_id: str
    steps: int
    total_reward: float
    final_points: int
    winner: int  # 0, 1, or -1 for tie

    # Performance metrics
    avg_policy_entropy: float
    avg_value_estimate: float
    max_gems_held: int
    cards_purchased: int
    gems_saved_via_bonus: int

    # Efficiency metrics
    points_per_turn: float
    cards_per_turn: float

    # Trajectory
    transitions: list[Transition]
```

### 1.2 Training Metadata

```python
# Checkpoint metadata (for online training)
@dataclass
class TrainingCheckpoint:
    checkpoint_id: str
    global_step: int
    timestamp: datetime

    # Model state
    model_weights: bytes  # serialized
    optimizer_state: bytes

    # Hyperparameters
    learning_rate: float
    reward_weights: dict[str, float]
    exploration_epsilon: float

    # Performance
    recent_win_rate: float  # vs self-play opponent
    elo_rating: float | None  # if using league training


# Aggregate Training Metrics
@dataclass
class TrainingMetrics:
    # Rolling windows (last 100 episodes)
    win_rate_vs_random: float
    win_rate_vs_greedy: float
    win_rate_self_play: float

    avg_episode_length: float
    avg_episode_reward: float

    # Learning dynamics
    policy_loss: float
    value_loss: float
    grad_norm: float

    # Exploration
    action_distribution: dict[str, int]  # counts by action_type
    unique_states_visited: int
```

---

## 2. Visualization Pipeline

### 2.1 Local Pipeline (Real-time Development)

```
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT PROCESS                            │
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│  │   Env    │───▶│  Policy  │───▶│  Action  │───▶│  Reward  │ │
│  │  State   │    │ Network  │    │ Executor │    │ Computer │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│       │               │                │               │        │
│       └───────────────┴────────────────┴───────────────┘        │
│                           │                                     │
│                  ┌────────▼─────────┐                           │
│                  │  Event Emitter   │                           │
│                  │  (WebSocket)     │                           │
│                  └────────┬─────────┘                           │
└───────────────────────────┼─────────────────────────────────────┘
                            │
                            │ JSON events
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      WEB UI (Browser)                            │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  Game Board      │  │ Policy Inspector │  │ Reward Graph  │ │
│  │  Renderer        │  │ (action probs)   │  │ (time series) │ │
│  │                  │  │                  │  │               │ │
│  │  - Cards         │  │ Buy Card A: 0.45 │  │     ^         │ │
│  │  - Gems          │  │ Take Gems:  0.32 │  │  R  │  /\     │ │
│  │  - Bonuses       │  │ Reserve:    0.23 │  │     │ /  \_   │ │
│  └──────────────────┘  └──────────────────┘  │     └────▶ t  │ │
│                                               └───────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │           State Analyzer Dashboard                         │ │
│  │                                                            │ │
│  │  Efficiency: 1.2 pts/turn  │  Gem Waste: 0 (optimal)     │ │
│  │  Bonus Value: 8 gems saved │  Victory Prob: 73%          │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Event Stream Format:**

```python
# WebSocket message types
EventType = Literal[
    'observation',
    'policy_output',
    'action_taken',
    'reward_received',
    'episode_end',
    'analyzer_update'
]

@dataclass
class VisualizationEvent:
    event_type: EventType
    timestamp: float
    data: dict  # polymorphic based on event_type

# Example events:
{
    "event_type": "policy_output",
    "timestamp": 1234567.89,
    "data": {
        "action_logits": {
            "buy_card_42": 0.45,
            "take_gems_pattern_7": 0.32,
            "reserve_card_15": 0.23
        },
        "value_estimate": 12.3,
        "entropy": 1.05
    }
}

{
    "event_type": "reward_received",
    "timestamp": 1234567.90,
    "data": {
        "components": {
            "tokens_held": 0.5,
            "cards_held": 2.0,
            "points_gained": 3.0,
            "win_lose": 0.0,
            "game_length_penalty": -0.1
        },
        "total": 5.4
    }
}
```

### 2.2 Online Pipeline (Distributed Training)

```
┌────────────────────────────────────────────────────────────────┐
│                    TRAINING CLUSTER                             │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Rollout     │  │  Rollout     │  │  Rollout     │         │
│  │  Worker 1    │  │  Worker 2    │  │  Worker N    │         │
│  │              │  │              │  │              │         │
│  │ (collect     │  │ (self-play   │  │ (vs baselines│         │
│  │  episodes)   │  │  episodes)   │  │  episodes)   │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                 │                  │
│         └─────────────────┴─────────────────┘                  │
│                           │                                    │
│                  ┌────────▼─────────┐                          │
│                  │  Replay Buffer   │                          │
│                  │  (Redis/DB)      │                          │
│                  └────────┬─────────┘                          │
│                           │                                    │
│                  ┌────────▼─────────┐                          │
│                  │  Learner Process │                          │
│                  │  (GPU training)  │                          │
│                  └────────┬─────────┘                          │
│                           │                                    │
│         ┌─────────────────┴─────────────────┐                  │
│         │                                   │                  │
│  ┌──────▼───────┐                  ┌────────▼────────┐         │
│  │ Checkpoint   │                  │  Metrics Queue  │         │
│  │ Storage (S3) │                  │  (Kafka/Kinesis)│         │
│  └──────────────┘                  └────────┬────────┘         │
└─────────────────────────────────────────────┼──────────────────┘
                                              │
                                              │ Metrics stream
                                              │
┌─────────────────────────────────────────────▼──────────────────┐
│                   ANALYTICS & VIZ LAYER                         │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ │
│  │  Time-Series DB  │  │  Aggregator      │  │  Dashboard   │ │
│  │  (InfluxDB)      │  │  (Spark/Flink)   │  │  (Grafana/   │ │
│  │                  │  │                  │  │   Custom)    │ │
│  │ - Win rate       │  │ - Compute stats  │  │              │ │
│  │ - Avg reward     │  │ - Detect         │  │  📊 📈 📉   │ │
│  │ - Policy loss    │  │   anomalies      │  │              │ │
│  └──────────────────┘  └──────────────────┘  └──────────────┘ │
│                                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Replay Viewer (On-Demand)                     │ │
│  │                                                            │ │
│  │  Load episode → Render game board → Step through actions  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Metrics Pipeline:**

```python
# Producers: Rollout workers emit events
worker.emit_metric({
    'metric_type': 'episode_complete',
    'worker_id': 'worker_7',
    'episode_id': 'ep_12345',
    'summary': EpisodeSummary(...),
    'checkpoint_version': 'v42'
})

# Consumers: Aggregators process stream
aggregator.process_window(
    window_size='1min',
    metrics=['win_rate', 'avg_reward'],
    group_by=['checkpoint_version']
)

# Storage: Time-series format
{
    'timestamp': 1699999999,
    'checkpoint_version': 'v42',
    'win_rate': 0.68,
    'avg_reward': 45.2,
    'episode_count': 1000
}
```

---

## 3. State Analyzers

### 3.1 Analyzer Interface

```python
class StateAnalyzer(ABC):
    """Base class for all state analyzers."""

    @abstractmethod
    def analyze(self, obs: AgentObservation, history: list[Transition]) -> dict:
        """Compute analysis metrics from current state and history."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Human-readable analyzer name."""
        pass
```

### 3.2 Implemented Analyzers

#### A. Efficiency Analyzer

```python
class EfficiencyAnalyzer(StateAnalyzer):
    """Measures resource utilization efficiency."""

    def analyze(self, obs: AgentObservation, history: list[Transition]) -> dict:
        turns_elapsed = obs.turn_number

        return {
            'points_per_turn': obs.points / max(turns_elapsed, 1),
            'cards_per_turn': len(obs.cards_owned) / max(turns_elapsed, 1),
            'bonus_efficiency': self._calculate_bonus_value(obs) / max(turns_elapsed, 1),
            'gem_waste': self._calculate_gem_waste(history),
        }

    def _calculate_bonus_value(self, obs: AgentObservation) -> float:
        """Estimate total gems saved via bonuses."""
        # Heuristic: each bonus gem saves ~2 turns of collection
        return sum(obs.bonuses) * 2.0

    def _calculate_gem_waste(self, history: list[Transition]) -> int:
        """Count gems returned due to 10-gem limit."""
        waste = 0
        for t in history:
            if t.action.gems_to_return:
                waste += sum(t.action.gems_to_return)
        return waste
```

#### B. Strategic Position Analyzer

```python
class StrategicPositionAnalyzer(StateAnalyzer):
    """Evaluates strategic advantage vs opponent."""

    def analyze(self, obs: AgentObservation, history: list[Transition]) -> dict:
        return {
            'point_lead': obs.points - obs.opponent_points,
            'bonus_advantage': sum(obs.bonuses) - sum(obs.opponent_bonuses),
            'resource_advantage': self._calculate_buying_power(obs),
            'victory_probability': self._estimate_win_prob(obs),
            'turns_to_win_estimate': self._estimate_turns_to_win(obs),
        }

    def _calculate_buying_power(self, obs: AgentObservation) -> float:
        """Effective purchasing power (gems + bonuses)."""
        return sum(obs.gems) + sum(obs.bonuses) * 1.5

    def _estimate_win_prob(self, obs: AgentObservation) -> float:
        """Simple heuristic: closer to 15 pts = higher probability."""
        my_progress = obs.points / 15.0
        opp_progress = obs.opponent_points / 15.0

        # Sigmoid-like function
        diff = my_progress - opp_progress
        return 1 / (1 + math.exp(-5 * diff))

    def _estimate_turns_to_win(self, obs: AgentObservation) -> float:
        """Estimate turns needed to reach 15 points."""
        points_needed = 15 - obs.points
        if points_needed <= 0:
            return 0.0

        # Assume can buy ~1.5 points per turn (rough heuristic)
        return points_needed / 1.5
```

#### C. Action Pattern Analyzer

```python
class ActionPatternAnalyzer(StateAnalyzer):
    """Analyzes temporal patterns in action selection."""

    def analyze(self, obs: AgentObservation, history: list[Transition]) -> dict:
        recent_window = history[-10:]  # last 10 actions

        action_counts = self._count_action_types(recent_window)

        return {
            'action_distribution': action_counts,
            'action_entropy': self._calculate_entropy(action_counts),
            'buy_to_take_ratio': action_counts.get('buy', 0) / max(action_counts.get('take_gems', 1), 1),
            'reserve_usage': action_counts.get('reserve', 0),
            'streak_length': self._longest_action_streak(recent_window),
        }

    def _count_action_types(self, transitions: list[Transition]) -> dict[str, int]:
        counts = defaultdict(int)
        for t in transitions:
            counts[t.action.action_type] += 1
        return dict(counts)

    def _calculate_entropy(self, counts: dict[str, int]) -> float:
        """Shannon entropy of action distribution."""
        total = sum(counts.values())
        if total == 0:
            return 0.0

        probs = [c / total for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _longest_action_streak(self, transitions: list[Transition]) -> int:
        """Find longest consecutive sequence of same action type."""
        if not transitions:
            return 0

        max_streak = 1
        current_streak = 1
        prev_type = transitions[0].action.action_type

        for t in transitions[1:]:
            if t.action.action_type == prev_type:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
                prev_type = t.action.action_type

        return max_streak
```

#### D. Value Function Analyzer

```python
class ValueFunctionAnalyzer(StateAnalyzer):
    """Analyzes value estimates and prediction accuracy."""

    def analyze(self, obs: AgentObservation, history: list[Transition]) -> dict:
        if not history:
            return {}

        # Look at last completed episode to evaluate value predictions
        returns = self._calculate_discounted_returns(history, gamma=0.99)
        predicted_values = [t.info.get('value_estimate', 0.0) for t in history]

        return {
            'value_prediction_error': self._mean_absolute_error(predicted_values, returns),
            'value_estimate_current': predicted_values[-1] if predicted_values else 0.0,
            'actual_return': returns[0] if returns else 0.0,
            'value_overestimation': sum(
                1 for v, r in zip(predicted_values, returns) if v > r
            ) / len(returns) if returns else 0.0,
        }

    def _calculate_discounted_returns(
        self,
        history: list[Transition],
        gamma: float
    ) -> list[float]:
        """Compute G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ..."""
        returns = []
        g = 0.0
        for t in reversed(history):
            g = t.reward + gamma * g
            returns.append(g)
        return list(reversed(returns))

    def _mean_absolute_error(self, predictions: list[float], targets: list[float]) -> float:
        if not predictions or len(predictions) != len(targets):
            return 0.0
        return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)
```

#### E. Card Synergy Analyzer

```python
class CardSynergyAnalyzer(StateAnalyzer):
    """Evaluates card collection synergies."""

    def __init__(self, deck: list[Card]):
        self.deck = deck

    def analyze(self, obs: AgentObservation, history: list[Transition]) -> dict:
        owned_cards = [self.deck[i] for i in obs.cards_owned]

        return {
            'color_diversity': self._calculate_color_diversity(owned_cards),
            'high_value_card_count': sum(1 for c in owned_cards if c.pt >= 3),
            'bonus_concentration': self._calculate_bonus_concentration(obs.bonuses),
            'affordable_card_count': self._count_affordable_cards(obs),
        }

    def _calculate_color_diversity(self, cards: list[Card]) -> float:
        """Shannon entropy of card color distribution."""
        color_counts = [0] * 5
        for card in cards:
            color_counts[card.bonus.value] += 1

        total = sum(color_counts)
        if total == 0:
            return 0.0

        probs = [c / total for c in color_counts if c > 0]
        return -sum(p * math.log2(p) for p in probs)

    def _calculate_bonus_concentration(self, bonuses: Gems) -> str:
        """Classify bonus distribution: 'focused', 'balanced', 'scattered'."""
        max_bonus = max(bonuses)
        total_bonus = sum(bonuses)

        if total_bonus == 0:
            return 'none'

        concentration = max_bonus / total_bonus

        if concentration >= 0.6:
            return 'focused'
        elif concentration >= 0.4:
            return 'balanced'
        else:
            return 'scattered'

    def _count_affordable_cards(self, obs: AgentObservation) -> int:
        """Count cards on board that are currently affordable."""
        available = (
            obs.available_cards_tier1 +
            obs.available_cards_tier2 +
            obs.available_cards_tier3
        )

        buying_power = tuple(
            g + b for g, b in zip(obs.gems, obs.bonuses)
        )

        affordable = 0
        for card_id in available:
            card = self.deck[card_id]
            if all(bp >= cost for bp, cost in zip(buying_power, card.cost)):
                affordable += 1

        return affordable
```

### 3.3 Analyzer Orchestrator

```python
class AnalyzerOrchestrator:
    """Coordinates multiple analyzers and aggregates results."""

    def __init__(self, analyzers: list[StateAnalyzer]):
        self.analyzers = analyzers

    def analyze_state(
        self,
        obs: AgentObservation,
        history: list[Transition]
    ) -> dict[str, dict]:
        """Run all analyzers and return aggregated results."""
        results = {}
        for analyzer in self.analyzers:
            try:
                results[analyzer.name()] = analyzer.analyze(obs, history)
            except Exception as e:
                # Don't let one analyzer crash the whole pipeline
                results[analyzer.name()] = {'error': str(e)}

        return results

    def generate_summary(self, analysis: dict[str, dict]) -> str:
        """Generate human-readable summary of key insights."""
        summary_parts = []

        # Efficiency
        if 'efficiency' in analysis:
            eff = analysis['efficiency']
            summary_parts.append(
                f"Efficiency: {eff['points_per_turn']:.2f} pts/turn"
            )

        # Position
        if 'strategic_position' in analysis:
            pos = analysis['strategic_position']
            summary_parts.append(
                f"Win Probability: {pos['victory_probability']:.0%}"
            )

        # Action patterns
        if 'action_pattern' in analysis:
            act = analysis['action_pattern']
            summary_parts.append(
                f"Strategy: {act['buy_to_take_ratio']:.1f} buy/take ratio"
            )

        return ' | '.join(summary_parts)
```

---

## 4. Architecture Decisions & Tradeoffs

### 4.1 Local vs Online: Detailed Comparison

| Dimension | Local (Dev) | Online (Training) | Decision Rationale |
|-----------|-------------|-------------------|-------------------|
| **Latency** | <50ms critical | <500ms acceptable | Local needs real-time for debugging |
| **Storage** | In-memory (GB) | Persistent (TB+) | Online needs historical analysis |
| **Data Fidelity** | Full state | Compressed features | Local benefits from complete info |
| **Query Pattern** | Random access | Sequential batch | Affects DB choice (SQLite vs InfluxDB) |
| **Visualization** | Interactive | Static dashboards | Local allows stepping through episodes |
| **Infrastructure** | Single machine | Cluster required | Cost vs capability tradeoff |

### 4.2 Technology Stack Recommendations

**Local Development:**
- **Backend**: FastAPI (async WebSocket support)
- **Frontend**: React + Canvas API (game board) + Recharts (graphs)
- **State Management**: Zustand (lightweight)
- **Data Format**: MessagePack (faster than JSON)

**Online Training:**
- **Orchestration**: Ray (distributed RL framework)
- **Metrics**: InfluxDB (time-series) + Grafana (dashboards)
- **Storage**: S3 (checkpoints) + PostgreSQL (episode metadata)
- **Streaming**: Kafka (if >1M events/min) or Redis Streams (lighter)

### 4.3 Key Design Principles

1. **Separation of Concerns**
   - Agent logic decoupled from visualization
   - Analyzers are independent, composable modules
   - Event-driven architecture enables async processing

2. **Progressive Enhancement**
   - Start local → Add online capabilities later
   - Basic analyzers → Advanced ML-based analyzers
   - Single-agent → Multi-agent population

3. **Observability First**
   - Every state change emits event
   - All analyzers log errors gracefully
   - Metrics pipeline has health checks

4. **Reproducibility**
   - Episode IDs enable replay from storage
   - Random seeds tracked in metadata
   - Checkpoint versioning for A/B tests

---

## 5. Implementation Roadmap

### Phase 1: Local Foundation (Week 1-2)
- [ ] Implement data models (`AgentObservation`, `Action`, etc.)
- [ ] Create basic analyzers (Efficiency, Strategic Position)
- [ ] Build WebSocket event emitter
- [ ] Simple web UI with game board renderer

### Phase 2: Advanced Local Features (Week 3-4)
- [ ] Add remaining analyzers
- [ ] Interactive episode replay
- [ ] A/B test different reward weights
- [ ] Export episodes to JSON

### Phase 3: Online Infrastructure (Week 5-6)
- [ ] Set up distributed rollout workers
- [ ] Implement metrics pipeline (Kafka → InfluxDB)
- [ ] Build Grafana dashboards
- [ ] S3 checkpoint storage

### Phase 4: Advanced Analytics (Week 7-8)
- [ ] Population-based training visualization
- [ ] Skill rating system (ELO/TrueSkill)
- [ ] Automatic hyperparameter tuning dashboard
- [ ] ML-based anomaly detection in training

---

## 6. Example Usage

### Local Mode

```python
# Start visualization server
from visualization.local_server import LocalVisualizationServer

server = LocalVisualizationServer(port=8080)
server.start()

# Run agent with visualization
from agents.dqn_agent import DQNAgent
from env.splendor_env import SplendorEnv

env = SplendorEnv()
agent = DQNAgent()

# Attach analyzers
from visualization.analyzers import (
    EfficiencyAnalyzer,
    StrategicPositionAnalyzer,
    ActionPatternAnalyzer
)

orchestrator = AnalyzerOrchestrator([
    EfficiencyAnalyzer(),
    StrategicPositionAnalyzer(),
    ActionPatternAnalyzer()
])

# Play episode with live visualization
obs = env.reset()
done = False
history = []

while not done:
    # Agent decision
    policy_output = agent.act(obs)

    # Emit to UI
    server.emit('policy_output', policy_output.to_dict())

    # Execute action
    next_obs, reward, done, info = env.step(policy_output.action)

    # Analyze state
    analysis = orchestrator.analyze_state(next_obs, history)
    server.emit('analyzer_update', analysis)

    # Store transition
    history.append(Transition(obs, policy_output.action, reward, next_obs, done, info))
    obs = next_obs

# Open browser to http://localhost:8080 for visualization
```

### Online Mode

```python
# Configure distributed training
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("SplendorEnv")
    .framework("torch")
    .rollouts(num_rollout_workers=16)
    .training(
        train_batch_size=4000,
        sgd_minibatch_size=128,
        num_sgd_iter=10
    )
    .callbacks(VisualizationCallbacks)  # Emit metrics to Kafka
)

# Run with metrics collection
tune.run(
    "PPO",
    config=config,
    stop={"training_iteration": 1000},
    checkpoint_freq=10
)

# View dashboard at http://grafana.mycompany.com/splendor-training
```

---

## Appendix: Sample Visualizations

### A. Local UI - Game Board
```
┌─────────────────────────────────────────────────────────┐
│ Splendor RL Agent Visualizer                            │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Your Gems:   ◆5 ◆3 ◆2 ◆1 ◆0  (Total: 11)             │
│  Bonuses:     ◆2 ◆1 ◆1 ◆0 ◆0                           │
│  Points:      8 / 15                                    │
│                                                         │
│  Available Cards (Tier 2):                              │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐                  │
│  │ 2pt  │ │ 1pt  │ │ 3pt  │ │ 1pt  │                  │
│  │ Blue │ │ Green│ │ Black│ │ Red  │                  │
│  │ 5◆2◆ │ │ 3◆1◆ │ │ 7◆   │ │ 4◆1◆ │                  │
│  └──────┘ └──────┘ └──────┘ └──────┘                  │
│     ↑ Agent targeting (p=0.65)                          │
│                                                         │
│  Policy Output:                                         │
│  • Buy 2pt Blue:    65% ████████████████               │
│  • Take 3 gems:     25% ██████                         │
│  • Reserve 3pt:     10% ██                             │
│                                                         │
│  Value Estimate: 12.3 (expected future return)          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### B. Online Dashboard - Training Metrics
```
Splendor RL Training Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Win Rate (Last 1000 Episodes)           Policy Loss
┌─────────────────────────────┐         ┌───────────────┐
│                       ╱─────│         │        ╲      │
│                  ╱───╱      │         │         ╲     │
│            ╱────╱           │         │          ╲_   │
│      ╱────╱                 │         │            ╲_ │
│ ────╱                       │         │              ╲│
│ 0%                    100%  │         │ 0          0.5│
└─────────────────────────────┘         └───────────────┘
  Step: 50000                             Step: 50000

Episode Length Distribution              Reward Components
┌─────────────────────────────┐         ┌───────────────┐
│          █                  │         │ Win:      45% │
│        █ █                  │         │ Points:   30% │
│      █ █ █ █                │         │ Cards:    15% │
│    █ █ █ █ █ █              │         │ Tokens:    8% │
│  █ █ █ █ █ █ █ █            │         │ Length:    2% │
│ 5  10  15  20  25  30  turns│         └───────────────┘
└─────────────────────────────┘

Recent Checkpoints:
• v52 @ 50000 steps - Win rate: 72% (ELO: 1450) ⭐ Best
• v51 @ 49000 steps - Win rate: 68% (ELO: 1420)
• v50 @ 48000 steps - Win rate: 65% (ELO: 1390)
```
