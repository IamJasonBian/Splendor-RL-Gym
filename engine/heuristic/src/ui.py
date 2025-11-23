"""Terminal UI renderer for displaying Splendor game states."""

from heuristic.src.cardparser import get_deck
from heuristic.src.color import Color
from heuristic.src.gems import Gems
from heuristic.src.solver import State

deck = get_deck()


def format_gems(gems: Gems) -> str:
    """Format gems as readable color names with amounts.

    Args:
        gems: Tuple of gem counts for each color

    Returns:
        Formatted string like 'White: 2, Blue: 1, Green: 3'
    """
    color_names = [c.name.title() for c in Color]
    parts = [f'{name}: {count}' for name, count in zip(color_names, gems) if count > 0]
    return ', '.join(parts) if parts else 'None'


def format_cards(card_indices: tuple[int, ...]) -> str:
    """Format purchased cards with their IDs.

    Args:
        card_indices: Tuple of card indices from the deck

    Returns:
        Formatted string showing card IDs
    """
    if not card_indices:
        return 'None'
    card_ids = [str(deck[idx]) for idx in card_indices]
    return ', '.join(card_ids)


def format_state(state: State, step: int) -> str:
    """Generate detailed description of a game state.

    Args:
        state: The game state to format
        step: The step number in the solution sequence

    Returns:
        Multi-line formatted string showing state details
    """
    lines = [
        f'\n=== Step {step} ===',
        f'Points: {state.pts}',
        f'Gems Saved: {state.saved}',
        f'Held Gems: {format_gems(state.gems)}',
        f'Bonus Gems: {format_gems(state.bonus)}',
        f'Cards: {format_cards(state.cards)}',
    ]
    return '\n'.join(lines)


def render_solution(solution: list[State]) -> None:
    """Render a complete solution sequence to the terminal.

    Args:
        solution: List of states representing the solution path
    """
    print('\n' + '='*60)
    print('SOLUTION PATH')
    print('='*60)

    for step, state in enumerate(solution):
        print(format_state(state, step))

    final_state = solution[-1]
    print('\n' + '='*60)
    print(f'FINAL: {final_state.pts} points in {len(solution)-1} moves')
    print('='*60)
