# Heuristic-Based Splendor Engine

This document describes the heuristic-based search engine for solving Splendor game scenarios.

## Overview

The Splendor solver uses a breadth-first search (BFS) algorithm with optional heuristic guidance to find optimal paths to reach a target point total. When heuristics are enabled, the solver uses a **beam search** approach, keeping only the most promising states at each turn based on a scoring function.

## Heuristic Functions

The engine provides four different heuristic functions, each optimized for different solving strategies:

### 1. Simple Heuristic (`simple`)

**Original heuristic from the base implementation.**

Formula: `(saved^0.4) * (pts^2.5) + random_noise`

**Best for:**
- General-purpose solving
- Quick experimentation
- Baseline comparisons

**Characteristics:**
- Balances saved gems and points
- Moderate point emphasis
- Small random factor for exploration

---

### 2. Balanced Heuristic (`balanced`)

**Multi-factor evaluation for well-rounded solutions.**

Factors evaluated:
- Points earned (heavily weighted)
- Gems saved through bonuses
- Total gem resources (gems + bonuses)
- Card count (more cards = more bonuses)
- Bonus diversity (variety of bonus colors)

**Best for:**
- Finding efficient solutions
- Building strong bonus engines
- Medium-term strategic planning

**Scoring weights:**
- Points: 100x (pts^2.8)
- Saved gems: 10x (saved^0.5)
- Resources: 5x ((gems + bonus*2)^0.3)
- Card count: 3x (cards^0.6)
- Diversity: 2x (unique_bonuses^0.4)

---

### 3. Aggressive Heuristic (`aggressive`)

**Heavy point prioritization for fast high-score solutions.**

Formula emphasizes:
- Very high point weight (pts^3.2)
- Low saved gem weight
- Moderate bonus consideration

**Best for:**
- Quickly finding any solution
- High point targets
- Greedy point accumulation

**Characteristics:**
- Fastest convergence
- May skip efficiency optimizations
- Good for high goal_pts (15+)

---

### 4. Efficiency Heuristic (`efficiency`)

**Resource optimization for minimal-move solutions.**

Factors evaluated:
- Moderate point weight (pts^2.0)
- Heavy saved gem emphasis (saved^0.7)
- Strong bonus value (bonus^1.2)
- High diversity premium (unique_bonuses^0.8)

**Best for:**
- Finding shortest paths
- Maximizing resource efficiency
- Building strong bonus engines early

**Characteristics:**
- Explores more states before committing to points
- Optimizes for gem savings
- Better for lower goal_pts (6-12)

---

### 5. Competitive Heuristic (`competitive`)

**Multi-player adversarial strategy for realistic mode.**

Factors evaluated:
- Point differential vs opponent (heavily weighted)
- Resource advantage (gems + bonuses)
- Market control (affordable cards)
- Bonus diversity

**Best for:**
- Realistic 2-player mode (`--realistic`)
- Competitive gameplay simulation
- Beating an opponent (not solo optimization)

**Characteristics:**
- Automatically used in realistic mode
- Considers opponent state and resources
- Zero-sum thinking (my points - opponent points)
- Market awareness (deny opponent valuable cards)

**Scoring Formula:**
- Point differential: Maximized (most important)
- Resource advantage: My resources vs opponent's
- Market control: Cards I can afford vs cards opponent can afford
- Diversity bonus: Variety of bonus colors

**Note:** This heuristic is specifically designed for `MultiPlayerState` and realistic mode. It won't work well for speedrun mode which is solo optimization.

---

## Usage Examples

### Basic Usage with Heuristics

```bash
# Solve for 10 points with simple heuristic
python splendor_fastest_win.py 10 -u

# Use balanced heuristic for 12 points
python splendor_fastest_win.py 12 -u -H balanced

# Use aggressive heuristic for 15 points
python splendor_fastest_win.py 15 -u -H aggressive

# Use efficiency heuristic with UI rendering
python splendor_fastest_win.py 10 -u -H efficiency -r
```

### Advanced Parameters

```bash
# Adjust beam width for more/less exploration
python splendor_fastest_win.py 12 -u -w 500000  # More states = slower but possibly better

# Quiet mode for cleaner output
python splendor_fastest_win.py 10 -u -q -r

# Combine multiple options
python splendor_fastest_win.py 15 -u -H aggressive -w 200000 -q -r
```

## Beam Width

The `beam_width` parameter controls how many states are kept at each turn:

- **Default:** 300,000 states
- **Lower values** (50,000-100,000): Faster but may miss optimal solutions
- **Higher values** (500,000+): Slower but more thorough exploration
- **Trade-off:** Memory usage vs solution quality

## Performance Characteristics

### Heuristic Comparison

| Heuristic   | Speed | Solution Quality | Resource Usage | Best For           |
|-------------|-------|------------------|----------------|--------------------|
| Simple      | ⭐⭐⭐  | ⭐⭐⭐            | ⭐⭐⭐          | General use        |
| Balanced    | ⭐⭐   | ⭐⭐⭐⭐           | ⭐⭐⭐          | Quality solutions  |
| Aggressive  | ⭐⭐⭐⭐ | ⭐⭐              | ⭐⭐⭐⭐         | Fast high scores   |
| Efficiency  | ⭐⭐   | ⭐⭐⭐⭐⭐          | ⭐⭐           | Minimal moves      |
| Competitive | ⭐     | ⭐⭐⭐⭐           | ⭐             | Multi-player games |

### Recommended Settings by Goal

- **Goal 6-8:** Efficiency heuristic, beam_width=200000
- **Goal 9-12:** Balanced heuristic, beam_width=300000
- **Goal 13-15:** Aggressive heuristic, beam_width=400000
- **Goal 16+:** Aggressive heuristic, beam_width=500000+

## Implementation Details

### Heuristic Function Signature

```python
def heuristic(state: State) -> float:
    """
    Args:
        state: Game state to evaluate

    Returns:
        Float score (higher = better)
    """
```

### Adding Custom Heuristics

To add a new heuristic function:

1. Define the function in `src/solver.py`:
```python
def my_heuristic(state: State) -> float:
    # Your scoring logic here
    return score
```

2. Register it in the `HEURISTICS` dictionary:
```python
HEURISTICS = {
    'simple': simple_heuristic,
    'balanced': balanced_heuristic,
    'aggressive': aggressive_heuristic,
    'efficiency': efficiency_heuristic,
    'my_custom': my_heuristic,  # Add here
}
```

3. Use it via CLI:
```bash
python splendor_fastest_win.py 10 -u -H my_custom
```

## Algorithm Overview

The solver uses **beam search**, a variant of BFS:

1. Start with initial game state
2. Generate all possible next states
3. Score each state using the heuristic function
4. Keep only the top N states (beam_width)
5. Repeat until goal is reached

### Complexity

- **Time:** O(turns × beam_width × branching_factor)
- **Space:** O(beam_width × state_size)
- **Typical solve time:** 1-30 seconds depending on settings

## Tips for Best Results

1. **Start with balanced** - Good general-purpose choice
2. **Use quiet mode** (`-q`) when you don't need debug output
3. **Enable rendering** (`-r`) to visualize solutions
4. **Adjust beam width** based on time constraints
5. **Try multiple heuristics** for the same goal to compare

## Future Enhancements

Potential improvements to the heuristic engine:

- Machine learning-based heuristics
- Dynamic beam width adjustment
- Multi-objective optimization
- Parallel beam search
- Adaptive heuristic selection
