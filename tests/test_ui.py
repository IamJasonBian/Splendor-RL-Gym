from src.solver import State
from src.ui import describe_state, format_cards, format_gems


def test_format_gems_human_readable():
    text = format_gems('Held gems', (1, 2, 3, 4, 5))
    assert 'Held gems' in text
    assert 'White: 1' in text
    assert 'Black: 5' in text


def test_format_cards_includes_card_ids():
    text = format_cards((0,))
    assert text.startswith('Cards: ')
    assert len(text.split(',')) >= 1


def test_describe_state_summarizes_numbers():
    state = State(cards=(0,), bonus=(1, 0, 0, 0, 0), gems=(2, 3, 4, 5, 6), pts=2, saved=1)
    summary = describe_state(3, state)
    assert 'Step 3' in summary
    assert 'Points: 2' in summary
    assert 'Saved cost (discounted by bonuses): 1' in summary
