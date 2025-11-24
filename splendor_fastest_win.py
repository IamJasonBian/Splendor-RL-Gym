#!/usr/bin/env python

"""A tool to bruteforce fastest winning moves for the board game Splendor."""

import argparse
import sys

from src.buys import export_buys_to_txt, load_buys
from src.color import Color
from src.solver import HEURISTICS, GameConfig, MultiPlayerState, State
from src.ui import render_solution


def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        'goal_pts',
        help='target amount of points',
        nargs='?',
        type=int,
    )
    parser.add_argument(
        '-u',
        '--use_heuristic',
        help='use a heuristic formula to limit the search space of BFS',
        action='store_true',
    )
    parser.add_argument(
        '-b',
        '--buys',
        help='regenerate and store all possible buys',
        action='store_true',
    )
    parser.add_argument(
        '-e',
        '--export',
        help='export possible buys to a .txt file',
        action='store_true',
    )
    parser.add_argument(
        '-r',
        '--render',
        help='render the solution with the UI',
        action='store_true',
    )
    parser.add_argument(
        '-H',
        '--heuristic',
        help=f'heuristic function to use (choices: {", ".join(HEURISTICS.keys())})',
        default='simple',
        choices=list(HEURISTICS.keys()),
    )
    parser.add_argument(
        '-w',
        '--beam_width',
        help='maximum states to keep per turn when using heuristic (default: 300000)',
        type=int,
        default=300_000,
    )
    parser.add_argument(
        '-q',
        '--quiet',
        help='suppress progress output during solving',
        action='store_true',
    )
    parser.add_argument(
        '--realistic',
        help='use realistic 2-player mode with gem pool and card visibility constraints',
        action='store_true',
    )
    parser.add_argument(
        '--players',
        help='number of players for realistic mode (default: 2)',
        type=int,
        default=2,
    )
    parser.add_argument(
        '--shuffle',
        help='shuffle card market in realistic mode',
        action='store_true',
    )
    parser.add_argument(
        '--strategies',
        help='player strategies for realistic mode (e.g., "balanced,aggressive" or "defensive,balanced")',
        type=str,
        default=None,
    )
    if len(sys.argv) == 1:  # no arguments given
        parser.print_help()
        parser.exit()
    args = parser.parse_args()
    try:
        if args.export:
            export_buys_to_txt()
            return
        if args.buys:
            load_buys(update=True)
        if args.goal_pts:
            if args.realistic:
                # Realistic multi-player mode
                gems_per_color = {2: 4, 3: 5, 4: 7}.get(args.players, 4)

                # Parse player strategies
                if args.strategies:
                    strategies = tuple(s.strip() for s in args.strategies.split(','))
                    if len(strategies) != args.players:
                        print(f'Error: Number of strategies ({len(strategies)}) must match number of players ({args.players})')
                        return
                    # Validate strategies
                    valid_strategies = {'balanced', 'aggressive', 'defensive'}
                    for s in strategies:
                        if s not in valid_strategies:
                            print(f'Error: Invalid strategy "{s}". Must be one of: {", ".join(valid_strategies)}')
                            return
                else:
                    # Default: all balanced
                    strategies = tuple('balanced' for _ in range(args.players))

                config = GameConfig(
                    num_players=args.players,
                    target_points=args.goal_pts,
                    gems_per_color=gems_per_color,
                    infinite_resources=False,
                    player_strategies=strategies,
                )
                solution = MultiPlayerState.newgame(
                    config=config,
                    shuffle_market=args.shuffle,
                ).solve(
                    use_heuristic=True,  # Force heuristic for realistic mode
                    heuristic_name='competitive',
                    beam_width=args.beam_width if args.beam_width != 300_000 else 20_000,
                    verbose=not args.quiet,
                )
                # Print solution
                if solution:
                    winner_id = solution[-1].get_winner()
                    print(f'\n{"="*60}')
                    print(f'Game Over! Winner: Player {winner_id}')
                    print(f'Final Scores:')
                    for p in solution[-1].players:
                        strategy = strategies[p.player_id]
                        print(f'  Player {p.player_id} ({strategy}): {p.pts} points, {len(p.cards)} cards')
                    print(f'Total moves: {solution[-1].turn_number}')
                    print(f'{"="*60}\n')

                    if args.render:
                        print('Move-by-move breakdown:')
                        for i, state in enumerate(solution):
                            print(f'\nMove {i}: {state}')
                            for p in state.players:
                                print(f'  P{p.player_id}: {p.pts}pts, gems={p.gems}, bonus={p.bonus}')
            else:
                # Speedrun mode (original)
                solution = State.newgame().solve(
                    goal_pts=args.goal_pts,
                    use_heuristic=args.use_heuristic,
                    heuristic_name=args.heuristic,
                    beam_width=args.beam_width,
                    verbose=not args.quiet,
                )
                if args.render:
                    render_solution(solution)
                else:
                    print('\nSolution:')
                    print(f'({", ".join(c.name.title() for c in Color)}) Cards')
                    for state in solution:
                        print(state)
    except KeyboardInterrupt:
        print('Execution stopped by the user.')
        parser.exit()


if __name__ == '__main__':
    cli()
