# splendor-fastest-win
An experimental tool to find fastest winning moves using bruteforce for the board game Splendor. It was born after a question on boardgames.stackexchange.com:

[Q: What is the fastest possible game in Splendor?](https://boardgames.stackexchange.com/questions/44948/what-is-the-fastest-possible-game-in-splendor)

The goal of the board game Splendor is to reach 15 points by either taking chips (gems) from the pool or buying cards with gems that give a permament gem bonus - a 1 gem discount for later buys.

<img src="https://github.com/monk-time/splendor-fastest-win/assets/7759622/9baa7e54-dd27-410d-8e86-e5e21c879de3" height="300px"/>

### How to run
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   . venv/Scripts/activate
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python splendor_fastest_win.py
   ```

### How it works
The tool uses breadth-first search to greedily check all possible move sequences. Optionally you can use a heuristic that limits search space of BFS by using only the most promising game states.

Data for all cards is stored in [cards.csv](cards.csv). Cards are referenced by a string consisting of a card's point value, one-letter color and a sorted list of its non-zero cost values.

Before the first run the tool caches (pickles) a dictionary that lists all cards that can be bought with each set of gems (possible buys).

### Limitations
Unoptimized bruteforce takes way too much memory and is unfeasible beyond 8 moves, but with a `-u` key it can get to 15 moves needed to reach the goal. Further optimization is required.

### Usage

```
usage: splendor_fastest_win.py [-h] [-u] [-b] [-e] [-r]
                               [-H {simple,balanced,aggressive,efficiency}]
                               [-w BEAM_WIDTH] [-q]
                               [goal_pts]

A tool to bruteforce fastest winning moves for the board game Splendor.

positional arguments:
  goal_pts              target amount of points

options:
  -h, --help            show this help message and exit
  -u, --use_heuristic   use a heuristic formula to limit the search space of BFS
  -b, --buys            regenerate and store all possible buys
  -e, --export          export possible buys to a .txt file
  -r, --render          render the solution with the UI
  -H {simple,balanced,aggressive,efficiency}, --heuristic
                        heuristic function to use (default: simple)
  -w BEAM_WIDTH, --beam_width BEAM_WIDTH
                        maximum states to keep per turn when using heuristic
                        (default: 300000)
  -q, --quiet           suppress progress output during solving
```

### Quick Start Examples

```bash
# Basic solve for 10 points
python splendor_fastest_win.py 10 -u

# Solve for 12 points with balanced heuristic and UI rendering
python splendor_fastest_win.py 12 -u -H balanced -r

# Fast solve for 15 points using aggressive heuristic
python splendor_fastest_win.py 15 -u -H aggressive -q -r

# Find efficient solution with minimal moves
python splendor_fastest_win.py 10 -u -H efficiency -w 500000 -r
```

### Heuristic Engine

This tool now includes a sophisticated heuristic-based search engine with four different strategies:

- **Simple** - Original balanced approach (default)
- **Balanced** - Multi-factor evaluation for quality solutions
- **Aggressive** - Heavy point prioritization for fast high scores
- **Efficiency** - Resource optimization for minimal-move solutions

For detailed information about heuristics, scoring functions, and performance characteristics, see [HEURISTICS.md](HEURISTICS.md).

### UI Renderer

The tool includes a terminal UI renderer to visualize solutions:

```bash
python splendor_fastest_win.py 10 -u -r
```

This displays each step of the solution with:
- Current points and gems saved
- Held gems and bonus gems
- Purchased cards
- Final statistics

### Advanced Features

**Custom Beam Width:**
Adjust the number of states explored at each turn:
```bash
python splendor_fastest_win.py 12 -u -w 500000  # More exploration
```

**Quiet Mode:**
Suppress progress output for cleaner results:
```bash
python splendor_fastest_win.py 10 -u -q -r
```

**Combining Options:**
Use multiple flags together for optimal results:
```bash
python splendor_fastest_win.py 15 -u -H balanced -w 400000 -q -r
```
### Realistic Multi-Player Mode

In addition to the speedrun mode (which assumes infinite gems and all cards visible), the tool now supports **realistic 2-player mode** with actual game constraints:

**Key Differences:**
- **Gem Pool Constraints**: Limited gem availability (4 per color for 2-player)
- **Card Visibility**: Only 12 cards visible at a time (4 per tier)
- **Turn-Based Play**: Players alternate turns
- **Player Strategies**: Configure different playstyles (balanced, aggressive, defensive)

**Usage:**
```bash
# Basic realistic 2-player game to 6 points
python splendor_fastest_win.py 6 --realistic -w 5000 -q

# Realistic game to 15 points with shuffled market
python splendor_fastest_win.py 15 --realistic --shuffle -w 20000 -q

# 3-player realistic game (uses 5 gems per color)
python splendor_fastest_win.py 12 --realistic --players 3 -w 15000 -q

# Aggressive vs defensive player matchup
python splendor_fastest_win.py 6 --realistic -w 5000 --strategies "aggressive,defensive" -q

# 3-player game with mixed strategies
python splendor_fastest_win.py 10 --realistic --players 3 --strategies "balanced,aggressive,defensive" -q
```

**Player Strategies:**

The realistic mode supports three distinct player strategies:

1. **Balanced** (default) - Plays to win efficiently, balances point-scoring with resource management
2. **Aggressive** - Actively blocks opponents, prioritizes denying cards to the leader
3. **Defensive** - Plays conservatively, builds engine (bonuses) early, minimizes vulnerability

Use `--strategies` to configure each player's strategy as a comma-separated list:
```bash
--strategies "balanced,aggressive"  # P0=balanced, P1=aggressive
--strategies "defensive,defensive"  # Both players play defensively
```

**Performance Notes:**
- Realistic mode has a much larger state space than speedrun mode
- Heuristic search is **mandatory** (automatically enabled)
- Default beam width is 20,000 (vs 300,000 for speedrun)
- Solving to 15 points may take 1-5 minutes
- Lower beam widths for faster (but potentially suboptimal) solutions

**Output Example:**
```
============================================================
REALISTIC MODE SOLVER
============================================================
Target Points: 6
Number of Players: 2
Player Strategies: P0=aggressive, P1=defensive
Gems per Color: 4
Beam Width: 5,000
Card Visibility: 12 cards (4 per tier)
Market Shuffled: Yes
============================================================

...

============================================================
Game Over! Winner: Player 0
Final Scores:
  Player 0 (aggressive): 8 points, 15 cards
  Player 1 (defensive): 6 points, 17 cards
Total moves: 47
============================================================
```

The solver simulates competitive gameplay where players use different strategies, with outcomes depending on strategy matchups and first-player advantage.

### Mode Comparison

| Feature | Speedrun Mode | Realistic Mode |
|---------|---------------|----------------|
| Gem Pool | Infinite | Limited (4-7 per color) |
| Card Visibility | All 90 cards | 12 cards (4 per tier) |
| Players | 1 (solo optimization) | 2-4 (competitive) |
| Strategy | Max your score | Beat opponent |
| State Space | Medium | Very Large |
| Solve Time | Seconds | Minutes |
| Use Case | Theoretical fastest win | Simulate real gameplay |

