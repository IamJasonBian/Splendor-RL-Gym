"""Tests for realistic multi-player mode."""

import pytest

from src.solver import CardMarket, GameConfig, GemPool, MultiPlayerState, PlayerState


def test_game_config_defaults():
    """Test GameConfig default values."""
    config = GameConfig()
    assert config.num_players == 2
    assert config.target_points == 15
    assert config.gems_per_color == 4
    assert config.infinite_resources is True


def test_gem_pool_creation():
    """Test creating a gem pool."""
    pool = GemPool.new_pool(4)
    assert pool.available == (4, 4, 4, 4, 4)


def test_gem_pool_take_three_different():
    """Test taking 3 different colored gems."""
    pool = GemPool.new_pool(4)
    gems_to_take = (1, 1, 1, 0, 0)

    assert pool.can_take_three_different(gems_to_take) is True

    new_pool = pool.take(gems_to_take)
    assert new_pool.available == (3, 3, 3, 4, 4)


def test_gem_pool_take_two_same():
    """Test taking 2 of same color requires 4 in pool."""
    pool = GemPool.new_pool(4)
    gems_to_take = (2, 0, 0, 0, 0)

    assert pool.can_take_two_same(gems_to_take) is True

    # Pool with only 3 left should reject
    pool3 = GemPool(available=(3, 4, 4, 4, 4))
    assert pool3.can_take_two_same(gems_to_take) is False


def test_gem_pool_return_gems():
    """Test returning gems to pool."""
    pool = GemPool(available=(2, 3, 4, 4, 4))
    gems_to_return = (1, 1, 0, 0, 0)

    new_pool = pool.return_gems(gems_to_return)
    assert new_pool.available == (3, 4, 4, 4, 4)


def test_card_market_creation():
    """Test creating a card market."""
    market = CardMarket.from_full_deck(shuffle=False)

    # Should have 4 cards visible per tier
    assert len(market.tier1_visible) == 4
    assert len(market.tier2_visible) == 4
    assert len(market.tier3_visible) == 4

    # Total visible cards
    assert len(market.all_visible_cards()) == 12


def test_card_market_buy_card():
    """Test buying a card from the market."""
    market = CardMarket.from_full_deck(shuffle=False)

    # Get first card from tier 1
    card_to_buy = market.tier1_visible[0]
    initial_visible_count = len(market.tier1_visible)
    initial_deck_count = len(market.tier1_deck)

    new_market = market.buy_card(card_to_buy)

    # Card should be replaced if deck has cards
    if initial_deck_count > 0:
        assert len(new_market.tier1_visible) == initial_visible_count
        assert len(new_market.tier1_deck) == initial_deck_count - 1
    else:
        assert len(new_market.tier1_visible) == initial_visible_count - 1

    # Card should no longer be visible
    assert card_to_buy not in new_market.tier1_visible


def test_player_state_can_afford():
    """Test checking if player can afford a card."""
    player = PlayerState(
        player_id=0,
        cards=(),
        bonus=(1, 0, 0, 0, 0),  # 1 white bonus
        gems=(3, 0, 0, 0, 0),  # 3 white gems
        pts=0,
        saved=0,
    )

    # Total resources: 4 white (3 gems + 1 bonus)
    # Can afford 4-cost white card but not 5-cost
    market = CardMarket.from_full_deck(shuffle=False)

    # This is just a structural test - actual affordability depends on card costs


def test_multiplayer_state_initialization():
    """Test creating a new multi-player game."""
    config = GameConfig(num_players=2, infinite_resources=False)
    state = MultiPlayerState.newgame(config=config)

    assert len(state.players) == 2
    assert state.current_player == 0
    assert state.turn_number == 0
    assert state.final_round_triggered is False
    assert state.gem_pool.available == (4, 4, 4, 4, 4)
    assert len(state.market.all_visible_cards()) == 12


def test_multiplayer_state_game_over():
    """Test game over detection."""
    config = GameConfig(num_players=2, target_points=15, infinite_resources=False)
    state = MultiPlayerState.newgame(config=config)

    # Initial state: not over
    assert state.is_game_over() is False

    # Create state where player 0 has 15 points
    winning_player = PlayerState(
        player_id=0,
        cards=(),
        bonus=(0, 0, 0, 0, 0),
        gems=(0, 0, 0, 0, 0),
        pts=15,
        saved=0,
    )
    other_player = state.players[1]

    winning_state = MultiPlayerState(
        config=config,
        players=(winning_player, other_player),
        gem_pool=state.gem_pool,
        market=state.market,
        current_player=0,
        turn_number=10,
    )

    # Should be game over
    assert winning_state.is_game_over() is True


def test_multiplayer_state_get_winner():
    """Test determining the winner."""
    config = GameConfig(num_players=2, target_points=15, infinite_resources=False)

    # Player 0 wins with more points
    player0 = PlayerState(
        player_id=0,
        cards=(),
        bonus=(0, 0, 0, 0, 0),
        gems=(0, 0, 0, 0, 0),
        pts=15,
        saved=0,
    )
    player1 = PlayerState(
        player_id=1,
        cards=(),
        bonus=(0, 0, 0, 0, 0),
        gems=(0, 0, 0, 0, 0),
        pts=10,
        saved=0,
    )

    state = MultiPlayerState(
        config=config,
        players=(player0, player1),
        gem_pool=GemPool.new_pool(4),
        market=CardMarket.from_full_deck(),
        current_player=0,
        turn_number=20,
        final_round_triggered=True,
        final_round_player=0,
    )

    assert state.get_winner() == 0


def test_multiplayer_state_move_generation():
    """Test that move generation works."""
    config = GameConfig(num_players=2, infinite_resources=False)
    state = MultiPlayerState.newgame(config=config)

    # Should be able to generate at least some moves
    moves = list(state)
    assert len(moves) > 0

    # Moves should alternate current player
    if len(moves) > 0:
        first_move = moves[0]
        assert first_move.current_player == 1  # Next player after 0


def test_realistic_mode_gem_constraints():
    """Test that realistic mode enforces gem pool constraints."""
    config = GameConfig(num_players=2, infinite_resources=False)
    state = MultiPlayerState.newgame(config=config)

    # Generate moves and check gem pool updates
    moves = list(state)

    for move in moves:
        # Gem pool should never have negative gems
        assert all(g >= 0 for g in move.gem_pool.available)

        # Player gems should not exceed limits
        for player in move.players:
            assert player.total_gem_count() <= 10
            assert all(g <= 7 for g in player.gems)
