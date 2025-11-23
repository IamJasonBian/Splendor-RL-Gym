from __future__ import annotations

import argparse
from textwrap import indent

from src.cardparser import get_deck
from src.color import Color
from src.solver import State

COLOR_LABELS = {
    Color.WHITE: 'White',
    Color.BLUE: 'Blue',
    Color.GREEN: 'Green',
    Color.RED: 'Red',
    Color.BLACK: 'Black',
}

deck = get_deck()


def format_gems(title: str, gems: tuple[int, ...]) -> str:
    color_pairs = zip(COLOR_LABELS.values(), gems)
    parts = ', '.join(f'{name}: {amount}' for name, amount in color_pairs)
    return f"{title}: {parts}"


def format_cards(cards: tuple[int, ...]) -> str:
    if not cards:
        return 'Cards: none yet'
    card_ids = ', '.join(deck[i].str_id for i in cards)
    return f'Cards: {card_ids}'


def describe_state(step: int, state: State) -> str:
    lines = [
        f'Step {step}',
        f'Points: {state.pts}',
        format_gems('Held gems', state.gems),
        format_gems('Bonus gems', state.bonus),
        format_cards(state.cards),
        f'Saved cost (discounted by bonuses): {state.saved}',
    ]
    return '\n'.join(lines)


def render_solution(goal_pts: int, use_heuristic: bool) -> str:
    solution = State.newgame().solve(goal_pts, use_heuristic=use_heuristic)
    rendered = [describe_state(step, state) for step, state in enumerate(solution)]
    return '\n\n'.join(rendered)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Render a human-friendly view of a solved Splendor game.',
    )
    parser.add_argument(
        'goal',
        nargs='?',
        type=int,
        default=15,
        help='Target number of points to reach (default: 15).',
    )
    parser.add_argument(
        '-u',
        '--use-heuristic',
        action='store_true',
        help='Use heuristic pruning when searching for the solution.',
    )
    args = parser.parse_args()

    solution_text = render_solution(args.goal, args.use_heuristic)
    header = f"Solution to reach {args.goal} points:\n"
    print(header + indent(solution_text, '  '))


if __name__ == '__main__':
    main()
