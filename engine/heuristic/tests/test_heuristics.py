"""Tests for heuristic functions."""

import pytest
from heuristic.src.solver import (
    HEURISTICS,
    State,
    aggressive_heuristic,
    balanced_heuristic,
    efficiency_heuristic,
    simple_heuristic,
)


def test_all_heuristics_registered():
    """Test that all heuristic functions are registered."""
    assert 'simple' in HEURISTICS
    assert 'balanced' in HEURISTICS
    assert 'aggressive' in HEURISTICS
    assert 'efficiency' in HEURISTICS


def test_simple_heuristic():
    """Test simple heuristic returns positive scores."""
    state = State.newgame()
    score = simple_heuristic(state)
    assert score >= 0
    assert isinstance(score, float)


def test_balanced_heuristic():
    """Test balanced heuristic returns positive scores."""
    state = State.newgame()
    score = balanced_heuristic(state)
    assert score >= 0
    assert isinstance(score, float)


def test_aggressive_heuristic():
    """Test aggressive heuristic returns positive scores."""
    state = State.newgame()
    score = aggressive_heuristic(state)
    assert score >= 0
    assert isinstance(score, float)


def test_efficiency_heuristic():
    """Test efficiency heuristic returns positive scores."""
    state = State.newgame()
    score = efficiency_heuristic(state)
    assert score >= 0
    assert isinstance(score, float)


def test_heuristic_scores_increase_with_points():
    """Test that all heuristics prefer higher point states."""
    # Note: simple heuristic requires saved > 0, so we test with saved=5
    state_low = State(cards=(), bonus=(0, 0, 0, 0, 0), gems=(0, 0, 0, 0, 0), pts=2, saved=5)
    state_high = State(cards=(), bonus=(0, 0, 0, 0, 0), gems=(0, 0, 0, 0, 0), pts=12, saved=5)

    for name, heuristic in HEURISTICS.items():
        # Skip competitive heuristic (designed for MultiPlayerState)
        if name == 'competitive':
            continue
        # Run multiple times to account for random noise
        scores_low = [heuristic(state_low) for _ in range(10)]
        scores_high = [heuristic(state_high) for _ in range(10)]
        avg_low = sum(scores_low) / len(scores_low)
        avg_high = sum(scores_high) / len(scores_high)
        assert avg_high > avg_low, f'{name} heuristic should prefer higher points'


def test_heuristic_values_saved_gems():
    """Test that heuristics value saved gems positively."""
    state_no_saved = State(cards=(), bonus=(0, 0, 0, 0, 0), gems=(5, 5, 5, 5, 5), pts=3, saved=0)
    state_saved = State(cards=(), bonus=(0, 0, 0, 0, 0), gems=(5, 5, 5, 5, 5), pts=3, saved=10)

    for name, heuristic in HEURISTICS.items():
        score_no_saved = heuristic(state_no_saved)
        score_saved = heuristic(state_saved)
        assert score_saved > score_no_saved, f'{name} heuristic should value saved gems'


def test_aggressive_emphasizes_points():
    """Test that aggressive heuristic emphasizes points more than others."""
    state_low = State(cards=(), bonus=(0, 0, 0, 0, 0), gems=(0, 0, 0, 0, 0), pts=1, saved=0)
    state_high = State(cards=(), bonus=(0, 0, 0, 0, 0), gems=(0, 0, 0, 0, 0), pts=5, saved=0)

    # Calculate score ratios for each heuristic
    aggressive_ratio = aggressive_heuristic(state_high) / max(aggressive_heuristic(state_low), 0.001)
    balanced_ratio = balanced_heuristic(state_high) / max(balanced_heuristic(state_low), 0.001)

    # Aggressive should have a higher ratio (more emphasis on point difference)
    assert aggressive_ratio > balanced_ratio


def test_efficiency_values_bonuses():
    """Test that efficiency heuristic values bonuses highly."""
    state_no_bonus = State(cards=(), bonus=(0, 0, 0, 0, 0), gems=(5, 5, 5, 5, 5), pts=2, saved=5)
    state_bonus = State(cards=(), bonus=(2, 2, 2, 2, 2), gems=(5, 5, 5, 5, 5), pts=2, saved=5)

    efficiency_diff = efficiency_heuristic(state_bonus) - efficiency_heuristic(state_no_bonus)
    simple_diff = simple_heuristic(state_bonus) - simple_heuristic(state_no_bonus)

    # Efficiency should value bonuses more than simple
    assert efficiency_diff > simple_diff


def test_heuristics_are_deterministic_without_noise():
    """Test that heuristics produce consistent scores (within random noise range)."""
    state = State(cards=(), bonus=(1, 1, 1, 1, 1), gems=(3, 3, 3, 3, 3), pts=7, saved=12)

    for name, heuristic in HEURISTICS.items():
        # Run multiple times and check variation is small (just random noise)
        scores = [heuristic(state) for _ in range(10)]
        # Max variation should be less than 1.0 (random component is max 1.0)
        variation = max(scores) - min(scores)
        assert variation < 1.0, f'{name} has too much variation: {variation}'


@pytest.mark.parametrize('heuristic_name', ['simple', 'balanced', 'aggressive', 'efficiency'])
def test_solve_with_heuristic(heuristic_name):
    """Test that solver works with each heuristic."""
    solution = State.newgame().solve(
        goal_pts=3,
        use_heuristic=True,
        heuristic_name=heuristic_name,
        beam_width=10000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[0].pts == 0  # Starts at 0
    assert solution[-1].pts >= 3  # Reaches goal


def test_solve_with_custom_beam_width():
    """Test that beam width parameter works."""
    # Small beam width should still find a solution for low goals
    solution = State.newgame().solve(
        goal_pts=3,
        use_heuristic=True,
        heuristic_name='simple',
        beam_width=1000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].pts >= 3


def test_solve_verbose_parameter():
    """Test that verbose parameter works without errors."""
    # Should not raise any exceptions
    solution = State.newgame().solve(
        goal_pts=3,
        use_heuristic=True,
        verbose=True,
    )

    assert len(solution) > 0


def test_invalid_heuristic_name_falls_back():
    """Test that invalid heuristic name falls back to simple."""
    # Should use simple heuristic as fallback
    solution = State.newgame().solve(
        goal_pts=3,
        use_heuristic=True,
        heuristic_name='invalid_name',
        beam_width=10000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].pts >= 3
