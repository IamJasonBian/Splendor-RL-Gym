"""Simulate a complete Splendor game and track analyzer metrics over time."""

import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from heuristic.src.cardparser import get_deck

from rl.src.rl_models import (
    Action,
    ActionPatternAnalyzer,
    AgentObservation,
    AnalyzerOrchestrator,
    CardSynergyAnalyzer,
    EfficiencyAnalyzer,
    StrategicPositionAnalyzer,
    Transition,
    ValueFunctionAnalyzer,
)

deck = get_deck()


def simulate_game_step(turn: int) -> tuple[AgentObservation, list[Transition]]:
    """Simulate a game state at a given turn."""
    # Simulate progression over time
    cards_owned_count = min(turn // 2, 8)
    cards_owned = tuple(range(cards_owned_count))

    # Distribute bonuses randomly across colors
    bonuses_total = cards_owned_count
    if bonuses_total > 0:
        bonuses_list = [0, 0, 0, 0, 0]
        for _ in range(bonuses_total):
            color = random.randint(0, 4)
            bonuses_list[color] += 1
        bonuses = tuple(bonuses_list)
    else:
        bonuses = (0, 0, 0, 0, 0)

    points = min(cards_owned_count * 2, 14)
    gems_total = max(0, 8 - turn // 3)
    gems = tuple(random.randint(0, gems_total // 5 + 1) for _ in range(5))

    # Opponent progresses slightly slower
    opponent_points = max(0, points - random.randint(1, 3))

    obs = AgentObservation(
        gems=gems,
        bonuses=bonuses,
        points=points,
        cards_owned=cards_owned,
        reserved_cards=(),
        opponent_gems_total=random.randint(5, 9),
        opponent_bonuses=tuple(random.randint(0, 2) for _ in range(5)),
        opponent_points=opponent_points,
        opponent_cards_count=cards_owned_count - 1,
        available_cards_tier1=tuple(range(1, 5)),
        available_cards_tier2=tuple(range(15, 19)),
        available_cards_tier3=tuple(range(40, 44)),
        gem_pool=tuple(random.randint(3, 7) for _ in range(5)),
        turn_number=turn,
        current_player=0,
    )

    # Simulate history with varied actions
    history = []
    for t in range(max(0, turn - 5), turn):
        action_type = random.choice(['buy', 'buy', 'take_gems', 'take_gems', 'take_gems'])
        if action_type == 'buy':
            action = Action(action_type='buy', card_id=random.randint(0, 89))
            reward = random.uniform(2.0, 5.0)
        else:
            action = Action(action_type='take_gems', gem_pattern=(1, 1, 1, 0, 0))
            reward = random.uniform(0.3, 0.8)

        history.append(Transition(
            observation=obs,
            action=action,
            reward=reward,
            next_observation=obs,
            done=False,
            info={'value_estimate': random.uniform(8.0, 15.0)},
        ))

    return obs, history


def print_progress_bar(current: int, total: int, label: str = '', width: int = 30):
    """Print a simple progress bar."""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percentage = current / total
    print(f'{label:20s} [{bar}] {percentage:6.1%}')


print('=' * 70)
print('SPLENDOR RL GAME SIMULATION - ANALYZER TRACKING')
print('=' * 70)
print()

# Initialize analyzers
analyzers = [
    EfficiencyAnalyzer(),
    StrategicPositionAnalyzer(),
    ActionPatternAnalyzer(),
    ValueFunctionAnalyzer(),
    CardSynergyAnalyzer(list(deck)),
]
orchestrator = AnalyzerOrchestrator(analyzers)

# Track metrics over time
turns = [0, 3, 6, 9, 12, 15]
metrics_over_time = {
    'points_per_turn': [],
    'victory_probability': [],
    'buy_to_take_ratio': [],
    'affordable_cards': [],
}

print('Simulating game progression...')
print()

for turn in turns:
    obs, history = simulate_game_step(turn)
    analysis = orchestrator.analyze_state(obs, history)

    # Extract key metrics
    if 'efficiency' in analysis:
        metrics_over_time['points_per_turn'].append(
            analysis['efficiency']['points_per_turn']
        )

    if 'strategic_position' in analysis:
        metrics_over_time['victory_probability'].append(
            analysis['strategic_position']['victory_probability']
        )

    if analysis.get('action_pattern'):
        metrics_over_time['buy_to_take_ratio'].append(
            analysis['action_pattern'].get('buy_to_take_ratio', 0.0)
        )

    if 'card_synergy' in analysis:
        metrics_over_time['affordable_cards'].append(
            analysis['card_synergy']['affordable_card_count']
        )

    # Display turn summary
    print(f'Turn {turn:2d}:')
    print(f'  Points: {obs.points:2d}/15  |  Gems: {sum(obs.gems):2d}  |  Bonuses: {sum(obs.bonuses):2d}  |  Cards: {len(obs.cards_owned):2d}')

    if 'strategic_position' in analysis:
        win_prob = analysis['strategic_position']['victory_probability']
        print_progress_bar(win_prob, 1.0, 'Win Probability')

    if 'efficiency' in analysis:
        pts_per_turn = analysis['efficiency']['points_per_turn']
        print(f'  Efficiency: {pts_per_turn:.2f} pts/turn')

    print()

# Summary visualization
print('=' * 70)
print('METRICS OVER TIME')
print('=' * 70)
print()

print('Points per Turn:')
for i, turn in enumerate(turns):
    if i < len(metrics_over_time['points_per_turn']):
        value = metrics_over_time['points_per_turn'][i]
        bar_length = int(value * 10)
        bar = '█' * bar_length
        print(f'  Turn {turn:2d}: {bar:15s} {value:.2f}')
print()

print('Victory Probability:')
for i, turn in enumerate(turns):
    if i < len(metrics_over_time['victory_probability']):
        value = metrics_over_time['victory_probability'][i]
        bar_length = int(value * 30)
        bar = '█' * bar_length
        print(f'  Turn {turn:2d}: {bar:30s} {value:.1%}')
print()

print('Buy/Take Ratio:')
for i, turn in enumerate(turns):
    if i < len(metrics_over_time['buy_to_take_ratio']):
        value = metrics_over_time['buy_to_take_ratio'][i]
        bar_length = int(min(value, 2) * 10)
        bar = '█' * bar_length
        print(f'  Turn {turn:2d}: {bar:20s} {value:.2f}')
print()

print('Affordable Cards on Board:')
for i, turn in enumerate(turns):
    if i < len(metrics_over_time['affordable_cards']):
        value = metrics_over_time['affordable_cards'][i]
        bar = '●' * value + '○' * (12 - value)
        print(f'  Turn {turn:2d}: {bar} ({value})')
print()

print('=' * 70)
print('KEY INSIGHTS')
print('=' * 70)

# Calculate trends
if len(metrics_over_time['points_per_turn']) > 1:
    ppt_trend = (
        metrics_over_time['points_per_turn'][-1] -
        metrics_over_time['points_per_turn'][0]
    )
    print(f'• Points/turn trend: {"+Improving" if ppt_trend > 0 else "Declining"}')

if len(metrics_over_time['victory_probability']) > 1:
    win_trend = (
        metrics_over_time['victory_probability'][-1] -
        metrics_over_time['victory_probability'][0]
    )
    print(f'• Win probability: {metrics_over_time["victory_probability"][-1]:.0%} ({f"+{win_trend:.0%}" if win_trend > 0 else f"{win_trend:.0%}"})')

if len(metrics_over_time['buy_to_take_ratio']) > 1:
    avg_ratio = sum(metrics_over_time['buy_to_take_ratio']) / len(metrics_over_time['buy_to_take_ratio'])
    strategy = 'Aggressive (buy-heavy)' if avg_ratio > 1.0 else 'Balanced' if avg_ratio > 0.5 else 'Conservative (gem-collection heavy)'
    print(f'• Strategy: {strategy} (avg ratio: {avg_ratio:.2f})')

print()
print('Simulation complete!')
