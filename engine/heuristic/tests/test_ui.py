"""Tests for the UI rendering module."""

from heuristic.src.color import Color
from heuristic.src.solver import State
from heuristic.src.ui import format_cards, format_gems, format_state


def test_format_gems_empty():
    """Test formatting empty gem collection."""
    gems = (0, 0, 0, 0, 0)
    assert format_gems(gems) == 'None'


def test_format_gems_single():
    """Test formatting single gem color."""
    gems = (2, 0, 0, 0, 0)
    assert format_gems(gems) == 'White: 2'


def test_format_gems_multiple():
    """Test formatting multiple gem colors."""
    gems = (1, 2, 0, 3, 0)
    result = format_gems(gems)
    assert 'White: 1' in result
    assert 'Blue: 2' in result
    assert 'Red: 3' in result


def test_format_cards_empty():
    """Test formatting empty card collection."""
    assert format_cards(()) == 'None'


def test_format_cards_single():
    """Test formatting single card."""
    result = format_cards((0,))
    assert result != 'None'
    assert len(result) > 0


def test_format_state():
    """Test formatting a game state."""
    state = State.newgame()
    result = format_state(state, 0)
    assert 'Step 0' in result
    assert 'Points: 0' in result
    assert 'Gems Saved: 0' in result
