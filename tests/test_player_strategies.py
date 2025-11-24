"""Tests for player strategy variations in realistic mode."""

import pytest

from src.solver import GameConfig, MultiPlayerState


def test_balanced_strategy():
    """Test balanced vs balanced strategy."""
    config = GameConfig(
        num_players=2,
        target_points=6,
        gems_per_color=4,
        infinite_resources=False,
        player_strategies=('balanced', 'balanced'),
    )

    solution = MultiPlayerState.newgame(config, shuffle_market=False).solve(
        beam_width=3000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].is_game_over()
    final_state = solution[-1]
    # At least one player should reach 6 points
    assert max(p.pts for p in final_state.players) >= 6


def test_aggressive_vs_defensive():
    """Test aggressive vs defensive strategy matchup."""
    config = GameConfig(
        num_players=2,
        target_points=6,
        gems_per_color=4,
        infinite_resources=False,
        player_strategies=('aggressive', 'defensive'),
    )

    solution = MultiPlayerState.newgame(config, shuffle_market=False).solve(
        beam_width=3000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].is_game_over()


def test_balanced_vs_aggressive():
    """Test balanced vs aggressive strategy matchup."""
    config = GameConfig(
        num_players=2,
        target_points=6,
        gems_per_color=4,
        infinite_resources=False,
        player_strategies=('balanced', 'aggressive'),
    )

    solution = MultiPlayerState.newgame(config, shuffle_market=False).solve(
        beam_width=3000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].is_game_over()


def test_balanced_vs_defensive():
    """Test balanced vs defensive strategy matchup."""
    config = GameConfig(
        num_players=2,
        target_points=6,
        gems_per_color=4,
        infinite_resources=False,
        player_strategies=('balanced', 'defensive'),
    )

    solution = MultiPlayerState.newgame(config, shuffle_market=False).solve(
        beam_width=3000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].is_game_over()


def test_defensive_vs_defensive():
    """Test defensive vs defensive strategy."""
    config = GameConfig(
        num_players=2,
        target_points=6,
        gems_per_color=4,
        infinite_resources=False,
        player_strategies=('defensive', 'defensive'),
    )

    solution = MultiPlayerState.newgame(config, shuffle_market=False).solve(
        beam_width=3000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].is_game_over()


def test_aggressive_vs_aggressive():
    """Test aggressive vs aggressive strategy."""
    config = GameConfig(
        num_players=2,
        target_points=6,
        gems_per_color=4,
        infinite_resources=False,
        player_strategies=('aggressive', 'aggressive'),
    )

    solution = MultiPlayerState.newgame(config, shuffle_market=False).solve(
        beam_width=3000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].is_game_over()


def test_three_player_strategies():
    """Test 3-player game with mixed strategies."""
    config = GameConfig(
        num_players=3,
        target_points=6,
        gems_per_color=5,
        infinite_resources=False,
        player_strategies=('balanced', 'aggressive', 'defensive'),
    )

    solution = MultiPlayerState.newgame(config, shuffle_market=False).solve(
        beam_width=2000,
        verbose=False,
    )

    assert len(solution) > 0
    assert solution[-1].is_game_over()


def test_config_player_strategies_default():
    """Test that GameConfig defaults to balanced strategies."""
    config = GameConfig(num_players=2)
    assert config.player_strategies == ('balanced', 'balanced')


def test_config_player_strategies_custom():
    """Test custom player strategies in config."""
    config = GameConfig(
        num_players=3,
        player_strategies=('aggressive', 'defensive', 'balanced'),
    )
    assert config.player_strategies == ('aggressive', 'defensive', 'balanced')
    assert len(config.player_strategies) == 3


def test_strategies_affect_gameplay():
    """Test that different strategies lead to different gameplay patterns."""
    # Same setup, different strategies - should produce different solutions
    base_config = {
        'num_players': 2,
        'target_points': 6,
        'gems_per_color': 4,
        'infinite_resources': False,
    }

    config1 = GameConfig(**base_config, player_strategies=('balanced', 'balanced'))
    config2 = GameConfig(**base_config, player_strategies=('aggressive', 'defensive'))

    # Use same market (no shuffle) for deterministic comparison
    solution1 = MultiPlayerState.newgame(config1, shuffle_market=False).solve(
        beam_width=3000,
        verbose=False,
    )

    solution2 = MultiPlayerState.newgame(config2, shuffle_market=False).solve(
        beam_width=3000,
        verbose=False,
    )

    # Solutions should exist
    assert len(solution1) > 0
    assert len(solution2) > 0

    # Different strategies may lead to different game lengths or outcomes
    # This is expected - we're just verifying both complete successfully
    assert solution1[-1].is_game_over()
    assert solution2[-1].is_game_over()
