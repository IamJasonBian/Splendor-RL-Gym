#!/usr/bin/env python

"""A tool to bruteforce fastest winning moves for the board game Splendor."""

import argparse
import sys
from typing import Optional

from src.buys import export_buys_to_txt, load_buys
from src.color import Color
from src.solver import State

# Import simulator functionality only when needed
try:
    from simulator import run_simulations, print_simulation_results, DEFAULT_GOAL_POINTS
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False


def run_simulator(goal_pts: int, use_heuristic: bool, num_simulations: int = 10, show_boards: int = 3) -> None:
    """Run the simulator with the given parameters."""
    if not SIMULATOR_AVAILABLE:
        print("Error: Simulator module not found. Make sure simulator.py is in the same directory.")
        return
    
    print(f"Running {num_simulations} simulations to {goal_pts} points...")
    result = run_simulations(
        num_simulations=num_simulations,
        goal_pts=goal_pts,
        use_heuristic=use_heuristic,
        verbose=False
    )
    print_simulation_results(result, show_boards=show_boards)

def cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    
    # Main arguments
    parser.add_argument(
        'goal_pts',
        help='target amount of points',
        nargs='?',
        type=int,
    )
    
    # Solver options
    parser.add_argument(
        '-u', '--use_heuristic',
        help='use a heuristic formula to limit the search space of BFS',
        action='store_true',
    )
    
    # Simulator options
    if SIMULATOR_AVAILABLE:
        sim_group = parser.add_argument_group('simulator options')
        sim_group.add_argument(
            '-s', '--simulate',
            metavar='N',
            type=int,
            nargs='?',
            const=10,
            help='run N simulations (default: 10, use with goal_pts)',
        )
        sim_group.add_argument(
            '--show-boards',
            type=int,
            default=3,
            help='number of board states to display (0-10, default: 3)',
        )
    
    # Buy table options
    buy_group = parser.add_argument_group('buy table options')
    buy_group.add_argument(
        '-b', '--buys',
        help='regenerate and store all possible buys',
        action='store_true',
    )
    buy_group.add_argument(
        '-e', '--export',
        help='export possible buys to a .txt file',
        action='store_true',
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
            return
            
        if args.goal_pts:
            if hasattr(args, 'simulate') and args.simulate is not None:
                # Run simulator
                run_simulator(
                    goal_pts=args.goal_pts,
                    use_heuristic=args.use_heuristic,
                    num_simulations=args.simulate,
                    show_boards=args.show_boards
                )
            else:
                # Run single game
                solution = State.newgame().solve(
                    goal_pts=args.goal_pts,
                    use_heuristic=args.use_heuristic,
                )
                print('\nSolution:')
                print(f'({ ", ".join(c.name.title() for c in Color)}) Cards')
                for state in solution:
                    print(state)
    except KeyboardInterrupt:
        print('Execution stopped by the user.')
        parser.exit()


if __name__ == '__main__':
    cli()
