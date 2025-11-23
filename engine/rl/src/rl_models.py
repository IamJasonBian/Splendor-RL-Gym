"""Data models for Splendor RL agent visualization and training."""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from heuristic.src.cardparser import Card
from heuristic.src.gems import Gems

# ============================================================================
# Core Agent Data Structures
# ============================================================================


@dataclass
class AgentObservation:
    """Complete observation of game state for agent decision-making."""

    # Player state
    gems: Gems  # tuple[int, int, int, int, int]
    bonuses: Gems  # permanent gem production from cards
    points: int
    cards_owned: tuple[int, ...]  # sorted card indices
    reserved_cards: tuple[int, ...] = field(
        default_factory=tuple
    )  # up to 3 cards

    # Opponent state (for 2-player game)
    opponent_gems_total: int = 0  # hide exact distribution
    opponent_bonuses: Gems = (0, 0, 0, 0, 0)
    opponent_points: int = 0
    opponent_cards_count: int = 0

    # Board state
    available_cards_tier1: tuple[int, ...] = field(
        default_factory=tuple
    )  # 4 visible cards
    available_cards_tier2: tuple[int, ...] = field(default_factory=tuple)
    available_cards_tier3: tuple[int, ...] = field(default_factory=tuple)
    gem_pool: Gems = (7, 7, 7, 7, 7)  # remaining gems in pool

    # Meta
    turn_number: int = 0
    current_player: int = 0  # 0 or 1


@dataclass
class Action:
    """Agent action in the Splendor environment."""

    action_type: Literal['buy', 'take_gems', 'reserve']

    # Polymorphic based on action_type
    card_id: int | None = None  # for buy/reserve
    gem_pattern: Gems | None = None  # for take_gems
    gems_to_return: Gems | None = None  # if exceeding 10 gems

    def __str__(self) -> str:
        if self.action_type == 'buy':
            return f'Buy card {self.card_id}'
        if self.action_type == 'reserve':
            return f'Reserve card {self.card_id}'
        return f'Take gems {self.gem_pattern}'


@dataclass
class PolicyOutput:
    """Policy network output for a given observation."""

    action: Action
    action_logits: dict[str, float]  # all valid actions + scores
    value_estimate: float  # V(s) - expected future return
    policy_entropy: float = 0.0  # exploration metric

    # Attention/importance (optional, for interpretability)
    feature_importance: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict for visualization."""
        return {
            'action': str(self.action),
            'action_logits': self.action_logits,
            'value_estimate': self.value_estimate,
            'policy_entropy': self.policy_entropy,
            'feature_importance': self.feature_importance,
        }


@dataclass
class Transition:
    """Single timestep in an episode trajectory."""

    observation: AgentObservation
    action: Action
    reward: float
    next_observation: AgentObservation
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class RewardComponents:
    """Decomposed reward signal for analysis."""

    tokens_held: float = 0.0
    cards_held: float = 0.0
    points_gained: float = 0.0
    win_lose: float = 0.0
    game_length_penalty: float = 0.0

    # Additional tracking
    breakdown_weights: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> float:
        """Sum of all reward components."""
        return (
            self.tokens_held
            + self.cards_held
            + self.points_gained
            + self.win_lose
            + self.game_length_penalty
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dict for visualization."""
        return {
            'tokens_held': self.tokens_held,
            'cards_held': self.cards_held,
            'points_gained': self.points_gained,
            'win_lose': self.win_lose,
            'game_length_penalty': self.game_length_penalty,
            'total': self.total,
        }


@dataclass
class EpisodeSummary:
    """Aggregate statistics for a complete episode."""

    episode_id: str
    steps: int
    total_reward: float
    final_points: int
    winner: int  # 0, 1, or -1 for tie

    # Performance metrics
    avg_policy_entropy: float = 0.0
    avg_value_estimate: float = 0.0
    max_gems_held: int = 0
    cards_purchased: int = 0
    gems_saved_via_bonus: int = 0

    # Efficiency metrics
    points_per_turn: float = 0.0
    cards_per_turn: float = 0.0

    # Trajectory
    transitions: list[Transition] = field(default_factory=list)


# ============================================================================
# Training Infrastructure Data Structures
# ============================================================================


@dataclass
class TrainingCheckpoint:
    """Checkpoint metadata for model versioning."""

    checkpoint_id: str
    global_step: int
    timestamp: datetime

    # Model state (paths or serialized data)
    model_weights_path: str
    optimizer_state_path: str

    # Hyperparameters
    learning_rate: float
    reward_weights: dict[str, float]
    exploration_epsilon: float

    # Performance
    recent_win_rate: float  # vs self-play opponent
    elo_rating: float | None = None  # if using league training


@dataclass
class TrainingMetrics:
    """Aggregate training metrics over a window."""

    # Rolling windows (last 100 episodes)
    win_rate_vs_random: float = 0.0
    win_rate_vs_greedy: float = 0.0
    win_rate_self_play: float = 0.0

    avg_episode_length: float = 0.0
    avg_episode_reward: float = 0.0

    # Learning dynamics
    policy_loss: float = 0.0
    value_loss: float = 0.0
    grad_norm: float = 0.0

    # Exploration
    action_distribution: dict[str, int] = field(default_factory=dict)
    unique_states_visited: int = 0


# ============================================================================
# State Analyzers
# ============================================================================


class StateAnalyzer(ABC):
    """Base class for all state analyzers."""

    @abstractmethod
    def analyze(
        self, obs: AgentObservation, history: list[Transition]
    ) -> dict[str, Any]:
        """Compute analysis metrics from current state and history."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable analyzer name."""


class EfficiencyAnalyzer(StateAnalyzer):
    """Measures resource utilization efficiency."""

    def name(self) -> str:
        return 'efficiency'

    def analyze(
        self, obs: AgentObservation, history: list[Transition]
    ) -> dict[str, Any]:
        turns_elapsed = obs.turn_number

        return {
            'points_per_turn': obs.points / max(turns_elapsed, 1),
            'cards_per_turn': len(obs.cards_owned) / max(turns_elapsed, 1),
            'bonus_efficiency': self._calculate_bonus_value(obs)
            / max(turns_elapsed, 1),
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


class StrategicPositionAnalyzer(StateAnalyzer):
    """Evaluates strategic advantage vs opponent."""

    def name(self) -> str:
        return 'strategic_position'

    def analyze(
        self, obs: AgentObservation, history: list[Transition]
    ) -> dict[str, Any]:
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


class ActionPatternAnalyzer(StateAnalyzer):
    """Analyzes temporal patterns in action selection."""

    def name(self) -> str:
        return 'action_pattern'

    def analyze(
        self, obs: AgentObservation, history: list[Transition]
    ) -> dict[str, Any]:
        recent_window = history[-10:]  # last 10 actions

        action_counts = self._count_action_types(recent_window)

        return {
            'action_distribution': action_counts,
            'action_entropy': self._calculate_entropy(action_counts),
            'buy_to_take_ratio': action_counts.get('buy', 0)
            / max(action_counts.get('take_gems', 1), 1),
            'reserve_usage': action_counts.get('reserve', 0),
            'streak_length': self._longest_action_streak(recent_window),
        }

    def _count_action_types(
        self, transitions: list[Transition]
    ) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
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


class ValueFunctionAnalyzer(StateAnalyzer):
    """Analyzes value estimates and prediction accuracy."""

    def name(self) -> str:
        return 'value_function'

    def analyze(
        self, obs: AgentObservation, history: list[Transition]
    ) -> dict[str, Any]:
        if not history:
            return {}

        # Look at last completed episode to evaluate value predictions
        returns = self._calculate_discounted_returns(history, gamma=0.99)
        predicted_values = [
            t.info.get('value_estimate', 0.0) for t in history
        ]

        return {
            'value_prediction_error': self._mean_absolute_error(
                predicted_values, returns
            ),
            'value_estimate_current': (
                predicted_values[-1] if predicted_values else 0.0
            ),
            'actual_return': returns[0] if returns else 0.0,
            'value_overestimation': (
                sum(1 for v, r in zip(predicted_values, returns) if v > r)
                / len(returns)
                if returns
                else 0.0
            ),
        }

    def _calculate_discounted_returns(
        self, history: list[Transition], gamma: float
    ) -> list[float]:
        """Compute G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ..."""
        returns = []
        g = 0.0
        for t in reversed(history):
            g = t.reward + gamma * g
            returns.append(g)
        return list(reversed(returns))

    def _mean_absolute_error(
        self, predictions: list[float], targets: list[float]
    ) -> float:
        if not predictions or len(predictions) != len(targets):
            return 0.0
        return (
            sum(abs(p - t) for p, t in zip(predictions, targets))
            / len(predictions)
        )


class CardSynergyAnalyzer(StateAnalyzer):
    """Evaluates card collection synergies."""

    def __init__(self, deck: list[Card]):
        self.deck = deck

    def name(self) -> str:
        return 'card_synergy'

    def analyze(
        self, obs: AgentObservation, history: list[Transition]
    ) -> dict[str, Any]:
        owned_cards = [self.deck[i] for i in obs.cards_owned]

        return {
            'color_diversity': self._calculate_color_diversity(owned_cards),
            'high_value_card_count': sum(1 for c in owned_cards if c.pt >= 3),
            'bonus_concentration': self._calculate_bonus_concentration(
                obs.bonuses
            ),
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
        if concentration >= 0.4:
            return 'balanced'
        return 'scattered'

    def _count_affordable_cards(self, obs: AgentObservation) -> int:
        """Count cards on board that are currently affordable."""
        available = (
            obs.available_cards_tier1
            + obs.available_cards_tier2
            + obs.available_cards_tier3
        )

        buying_power = tuple(g + b for g, b in zip(obs.gems, obs.bonuses))

        affordable = 0
        for card_id in available:
            card = self.deck[card_id]
            if all(bp >= cost for bp, cost in zip(buying_power, card.cost)):
                affordable += 1

        return affordable


class AnalyzerOrchestrator:
    """Coordinates multiple analyzers and aggregates results."""

    def __init__(self, analyzers: list[StateAnalyzer]):
        self.analyzers = analyzers

    def analyze_state(
        self, obs: AgentObservation, history: list[Transition]
    ) -> dict[str, dict[str, Any]]:
        """Run all analyzers and return aggregated results."""
        results: dict[str, dict[str, Any]] = {}
        for analyzer in self.analyzers:
            try:
                results[analyzer.name()] = analyzer.analyze(obs, history)
            except Exception as e:
                # Don't let one analyzer crash the whole pipeline
                results[analyzer.name()] = {'error': str(e)}

        return results

    def generate_summary(self, analysis: dict[str, dict[str, Any]]) -> str:
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
