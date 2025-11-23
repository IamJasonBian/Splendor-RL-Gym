"""Demo script to test the state analyzers with simulated game data."""

from src.cardparser import get_deck
from src.rl_models import (
    Action,
    AgentObservation,
    AnalyzerOrchestrator,
    ActionPatternAnalyzer,
    CardSynergyAnalyzer,
    EfficiencyAnalyzer,
    RewardComponents,
    StrategicPositionAnalyzer,
    Transition,
    ValueFunctionAnalyzer,
)

# Load the deck
deck = get_deck()
print(f'Loaded {len(deck)} cards from deck')
print(f'Sample card: {deck[0]} - Cost: {deck[0].cost}, Points: {deck[0].pt}, Bonus: {deck[0].bonus}')
print()

# Create a simulated mid-game state
obs = AgentObservation(
    gems=(3, 2, 1, 2, 0),  # has some gems
    bonuses=(2, 1, 1, 0, 0),  # bought some cards already
    points=8,  # mid-game
    cards_owned=(0, 5, 12, 23),  # owns 4 cards
    reserved_cards=(),
    opponent_gems_total=7,
    opponent_bonuses=(1, 2, 0, 1, 0),
    opponent_points=6,
    opponent_cards_count=3,
    available_cards_tier1=(1, 2, 3, 4),
    available_cards_tier2=(15, 16, 17, 18),
    available_cards_tier3=(40, 41, 42, 43),
    gem_pool=(4, 5, 6, 5, 7),
    turn_number=10,
    current_player=0,
)

# Create a simulated history with some actions
history = [
    Transition(
        observation=obs,
        action=Action(action_type='take_gems', gem_pattern=(1, 1, 1, 0, 0)),
        reward=0.5,
        next_observation=obs,
        done=False,
        info={'value_estimate': 10.0},
    ),
    Transition(
        observation=obs,
        action=Action(action_type='buy', card_id=5),
        reward=3.0,
        next_observation=obs,
        done=False,
        info={'value_estimate': 12.0},
    ),
    Transition(
        observation=obs,
        action=Action(action_type='take_gems', gem_pattern=(2, 0, 0, 0, 0)),
        reward=0.3,
        next_observation=obs,
        done=False,
        info={'value_estimate': 11.5},
    ),
    Transition(
        observation=obs,
        action=Action(action_type='buy', card_id=12),
        reward=2.5,
        next_observation=obs,
        done=False,
        info={'value_estimate': 13.0},
    ),
    Transition(
        observation=obs,
        action=Action(action_type='take_gems', gem_pattern=(1, 1, 1, 0, 0)),
        reward=0.4,
        next_observation=obs,
        done=False,
        info={'value_estimate': 12.8},
    ),
]

print('=' * 70)
print('SPLENDOR RL AGENT - STATE ANALYZER DEMO')
print('=' * 70)
print()

print('Current Game State:')
print(f'  Turn: {obs.turn_number}')
print(f'  Your Gems: {obs.gems} (Total: {sum(obs.gems)})')
print(f'  Your Bonuses: {obs.bonuses} (Total: {sum(obs.bonuses)})')
print(f'  Your Points: {obs.points} / 15')
print(f'  Cards Owned: {len(obs.cards_owned)}')
print()
print(f'  Opponent Points: {obs.opponent_points} / 15')
print(f'  Opponent Bonuses: {obs.opponent_bonuses}')
print()

# Initialize all analyzers
analyzers = [
    EfficiencyAnalyzer(),
    StrategicPositionAnalyzer(),
    ActionPatternAnalyzer(),
    ValueFunctionAnalyzer(),
    CardSynergyAnalyzer(list(deck)),
]

orchestrator = AnalyzerOrchestrator(analyzers)

# Run analysis
print('Running State Analysis...')
print('=' * 70)
print()

analysis = orchestrator.analyze_state(obs, history)

# Display results
for analyzer_name, metrics in analysis.items():
    print(f'[{analyzer_name.upper()}]')
    if 'error' in metrics:
        print(f'  ERROR: {metrics["error"]}')
    else:
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f'  {key}: {value:.3f}')
            elif isinstance(value, dict):
                print(f'  {key}:')
                for k, v in value.items():
                    print(f'    {k}: {v}')
            else:
                print(f'  {key}: {value}')
    print()

# Generate summary
print('=' * 70)
print('SUMMARY')
print('=' * 70)
summary = orchestrator.generate_summary(analysis)
print(summary)
print()

# Test reward decomposition
print('=' * 70)
print('REWARD DECOMPOSITION EXAMPLE')
print('=' * 70)
reward = RewardComponents(
    tokens_held=0.5,
    cards_held=2.0,
    points_gained=3.0,
    win_lose=0.0,
    game_length_penalty=-0.1,
    breakdown_weights={'tokens': 0.1, 'cards': 0.5, 'points': 1.0},
)

print(f'Tokens Held Reward: {reward.tokens_held:.2f}')
print(f'Cards Held Reward: {reward.cards_held:.2f}')
print(f'Points Gained Reward: {reward.points_gained:.2f}')
print(f'Win/Lose Reward: {reward.win_lose:.2f}')
print(f'Game Length Penalty: {reward.game_length_penalty:.2f}')
print(f'Total Reward: {reward.total:.2f}')
print()

print('Analysis complete!')
