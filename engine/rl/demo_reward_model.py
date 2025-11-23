"""Demo of the reward function based on SplendorSimulator's approach."""

import sys
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rl.src.rl_models import AgentObservation, RewardComponents


@dataclass
class RewardSettings:
    """Configuration for reward function components."""

    tokens_held: tuple[bool, float] = (True, 0.1)  # (enabled, weight)
    cards_held: tuple[bool, float] = (True, 0.5)
    points: tuple[bool, float] = (True, 1.0)
    win_lose: tuple[bool, float] = (True, 100.0)
    length_of_game: tuple[bool, float] = (True, -0.05)


class SplendorRewardFunction:
    """Reward function matching SplendorSimulator's design."""

    def __init__(self, settings: RewardSettings):
        self.settings = settings

    def calculate_reward(
        self,
        obs: AgentObservation,
        prev_obs: AgentObservation | None,
        done: bool,
        winner: int | None = None,
    ) -> RewardComponents:
        """Calculate reward for transitioning from prev_obs to obs."""
        components = RewardComponents()

        # Tokens held reward
        if self.settings.tokens_held[0]:
            token_value = sum(obs.gems)
            # Gold tokens worth 1.5x (not in simplified version)
            components.tokens_held = token_value * self.settings.tokens_held[1]

        # Cards held reward (permanent bonuses)
        if self.settings.cards_held[0]:
            cards_value = sum(obs.bonuses)
            components.cards_held = cards_value * self.settings.cards_held[1]

        # Points reward (incremental)
        if self.settings.points[0] and prev_obs:
            points_gained = obs.points - prev_obs.points
            components.points_gained = points_gained * self.settings.points[1]

        # Win/lose reward (terminal)
        if self.settings.win_lose[0] and done:
            if winner == obs.current_player:
                components.win_lose = self.settings.win_lose[1]
            elif winner is not None and winner != obs.current_player:
                components.win_lose = -self.settings.win_lose[1]

        # Game length penalty (encourage faster games)
        if self.settings.length_of_game[0]:
            components.game_length_penalty = (
                self.settings.length_of_game[1] * obs.turn_number
            )

        components.breakdown_weights = {
            'tokens': self.settings.tokens_held[1],
            'cards': self.settings.cards_held[1],
            'points': self.settings.points[1],
            'win_lose': self.settings.win_lose[1],
            'length': self.settings.length_of_game[1],
        }

        return components


def demonstrate_reward_scenarios():
    """Show reward calculations for different game scenarios."""
    print('=' * 70)
    print('SPLENDOR REWARD FUNCTION DEMO')
    print('=' * 70)
    print()

    # Scenario 1: Taking gems
    print('[SCENARIO 1: Taking Gems]')
    prev_obs = AgentObservation(
        gems=(2, 1, 0, 1, 0),
        bonuses=(1, 0, 0, 0, 0),
        points=3,
        cards_owned=(0, 5),
        turn_number=5,
    )
    obs = AgentObservation(
        gems=(3, 2, 1, 1, 0),  # collected 3 gems
        bonuses=(1, 0, 0, 0, 0),
        points=3,
        cards_owned=(0, 5),
        turn_number=6,
    )

    settings = RewardSettings()
    reward_fn = SplendorRewardFunction(settings)
    reward = reward_fn.calculate_reward(obs, prev_obs, done=False)

    print(f'Previous state: {sum(prev_obs.gems)} gems, {prev_obs.points} pts')
    print(f'Current state:  {sum(obs.gems)} gems, {obs.points} pts')
    print('Reward breakdown:')
    for key, value in reward.to_dict().items():
        if key != 'total':
            print(f'  {key:20s}: {value:+.2f}')
    print(f'  {"TOTAL":20s}: {reward.total:+.2f}')
    print()

    # Scenario 2: Buying a card
    print('[SCENARIO 2: Buying a Card (2 points)]')
    prev_obs = AgentObservation(
        gems=(5, 3, 2, 1, 0),
        bonuses=(1, 0, 0, 0, 0),
        points=3,
        cards_owned=(0, 5),
        turn_number=7,
    )
    obs = AgentObservation(
        gems=(2, 0, 0, 1, 0),  # spent gems
        bonuses=(1, 1, 0, 0, 0),  # gained bonus
        points=5,  # gained 2 points
        cards_owned=(0, 5, 12),  # added card
        turn_number=8,
    )

    reward = reward_fn.calculate_reward(obs, prev_obs, done=False)

    print(
        f'Previous state: {sum(prev_obs.gems)} gems, {sum(prev_obs.bonuses)} bonuses, {prev_obs.points} pts'
    )
    print(
        f'Current state:  {sum(obs.gems)} gems, {sum(obs.bonuses)} bonuses, {obs.points} pts'
    )
    print('Reward breakdown:')
    for key, value in reward.to_dict().items():
        if key != 'total':
            print(f'  {key:20s}: {value:+.2f}')
    print(f'  {"TOTAL":20s}: {reward.total:+.2f}')
    print()

    # Scenario 3: Winning the game
    print('[SCENARIO 3: Winning the Game]')
    prev_obs = AgentObservation(
        gems=(1, 0, 0, 0, 0),
        bonuses=(3, 2, 2, 1, 1),
        points=13,
        cards_owned=tuple(range(10)),
        turn_number=14,
    )
    obs = AgentObservation(
        gems=(0, 0, 0, 0, 0),
        bonuses=(3, 2, 2, 1, 1),
        points=15,  # won!
        cards_owned=tuple(range(11)),
        turn_number=15,
    )

    reward = reward_fn.calculate_reward(obs, prev_obs, done=True, winner=0)

    print(
        f'Previous state: {prev_obs.points} pts (turn {prev_obs.turn_number})'
    )
    print(
        f'Current state:  {obs.points} pts (turn {obs.turn_number}) - VICTORY!'
    )
    print('Reward breakdown:')
    for key, value in reward.to_dict().items():
        if key != 'total':
            print(f'  {key:20s}: {value:+.2f}')
    print(f'  {"TOTAL":20s}: {reward.total:+.2f}')
    print()

    # Scenario 4: Losing the game
    print('[SCENARIO 4: Losing the Game]')
    prev_obs = AgentObservation(
        gems=(2, 1, 1, 0, 0),
        bonuses=(2, 1, 1, 0, 0),
        points=10,
        cards_owned=tuple(range(7)),
        turn_number=16,
        opponent_points=14,
    )
    obs = AgentObservation(
        gems=(2, 1, 1, 0, 0),
        bonuses=(2, 1, 1, 0, 0),
        points=10,
        cards_owned=tuple(range(7)),
        turn_number=17,
        opponent_points=15,  # opponent won
    )

    reward = reward_fn.calculate_reward(obs, prev_obs, done=True, winner=1)

    print(
        f'Previous state: {prev_obs.points} pts, opponent {prev_obs.opponent_points} pts'
    )
    print(
        f'Current state:  {obs.points} pts, opponent {obs.opponent_points} pts - DEFEAT!'
    )
    print('Reward breakdown:')
    for key, value in reward.to_dict().items():
        if key != 'total':
            print(f'  {key:20s}: {value:+.2f}')
    print(f'  {"TOTAL":20s}: {reward.total:+.2f}')
    print()

    # Compare different reward configurations
    print('=' * 70)
    print('REWARD CONFIGURATION COMPARISON')
    print('=' * 70)
    print()

    configs = {
        'Sparse (win only)': RewardSettings(
            tokens_held=(False, 0.0),
            cards_held=(False, 0.0),
            points=(False, 0.0),
            win_lose=(True, 100.0),
            length_of_game=(False, 0.0),
        ),
        'Dense (all components)': RewardSettings(
            tokens_held=(True, 0.1),
            cards_held=(True, 0.5),
            points=(True, 1.0),
            win_lose=(True, 100.0),
            length_of_game=(True, -0.05),
        ),
        'Points focused': RewardSettings(
            tokens_held=(False, 0.0),
            cards_held=(True, 0.3),
            points=(True, 2.0),
            win_lose=(True, 50.0),
            length_of_game=(True, -0.1),
        ),
    }

    # Test scenario: buying a 2-point card
    test_prev = AgentObservation(
        gems=(5, 3, 2, 1, 0),
        bonuses=(1, 0, 0, 0, 0),
        points=6,
        cards_owned=(0, 5),
        turn_number=8,
    )
    test_obs = AgentObservation(
        gems=(2, 0, 0, 1, 0),
        bonuses=(1, 1, 0, 0, 0),
        points=8,
        cards_owned=(0, 5, 12),
        turn_number=9,
    )

    print('Buying a 2-point card (turn 9):')
    print()

    for config_name, config in configs.items():
        reward_fn = SplendorRewardFunction(config)
        reward = reward_fn.calculate_reward(test_obs, test_prev, done=False)
        print(f'{config_name:25s}: Total reward = {reward.total:+6.2f}')
        print(
            f'  Components: tokens={reward.tokens_held:+.2f}, cards={reward.cards_held:+.2f}, '
            f'points={reward.points_gained:+.2f}, win={reward.win_lose:+.2f}, length={reward.game_length_penalty:+.2f}'
        )
        print()


if __name__ == '__main__':
    demonstrate_reward_scenarios()
    print('Demo complete!')
