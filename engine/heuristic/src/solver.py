from bisect import insort
from collections.abc import Callable
from dataclasses import dataclass
from random import randint

from heuristic.src.buys import get_buys
from heuristic.src.cardparser import CardIndices, get_deck
from heuristic.src.color import COLOR_NUM
from heuristic.src.gems import (
    MAX_GEMS,
    Gems,
    get_takes,
    increase_bonus,
    subtract_with_bonus,
)

deck = get_deck()


# ============================================================================
# Realistic Multi-Player Game Infrastructure
# ============================================================================


@dataclass(frozen=True)
class GameConfig:
    """Configuration for game rules."""

    num_players: int = 2
    target_points: int = 15
    gems_per_color: int = 4  # 4 for 2p, 5 for 3p, 7 for 4p
    cards_visible_per_tier: int = 4
    infinite_resources: bool = True  # Speedrun mode toggle


@dataclass(frozen=True)
class GemPool:
    """Global gem pool tracking for realistic mode."""

    available: Gems  # How many of each color remain in pool

    @classmethod
    def new_pool(cls, gems_per_color: int) -> 'GemPool':
        """Create initial gem pool."""
        return cls(available=tuple(gems_per_color for _ in range(COLOR_NUM)))

    def can_take_three_different(self, gems_requested: Gems) -> bool:
        """Check if taking 3 different gems is legal."""
        count = sum(1 for g in gems_requested if g > 0)
        if count != 3:
            return False
        return all(
            gems_requested[i] <= 1 and self.available[i] >= gems_requested[i]
            for i in range(COLOR_NUM)
        )

    def can_take_two_same(self, gems_requested: Gems) -> bool:
        """Check if taking 2 of same color is legal."""
        if sum(gems_requested) != 2:
            return False
        color_idx = next(
            (i for i in range(COLOR_NUM) if gems_requested[i] == 2), None
        )
        if color_idx is None:
            return False
        return self.available[color_idx] >= 4  # Need 4 in pool to take 2

    def take(self, gems: Gems) -> 'GemPool':
        """Return new pool with gems removed."""
        new_available = tuple(
            self.available[i] - gems[i] for i in range(COLOR_NUM)
        )
        return GemPool(new_available)

    def return_gems(self, gems: Gems) -> 'GemPool':
        """Return new pool with gems added back."""
        new_available = tuple(
            self.available[i] + gems[i] for i in range(COLOR_NUM)
        )
        return GemPool(new_available)


@dataclass(frozen=True)
class CardMarket:
    """Visible cards and deck state for realistic mode."""

    tier1_visible: tuple[int, ...]  # Card indices (up to 4 cards)
    tier2_visible: tuple[int, ...]  # Card indices (up to 4 cards)
    tier3_visible: tuple[int, ...]  # Card indices (up to 4 cards)
    tier1_deck: tuple[int, ...]  # Remaining deck
    tier2_deck: tuple[int, ...]
    tier3_deck: tuple[int, ...]

    @classmethod
    def from_full_deck(
        cls, shuffle: bool = False, seed: int | None = None
    ) -> 'CardMarket':
        """Initialize market from full deck, optionally shuffled."""
        import random

        # Separate by tier based on points
        tier1_cards = [i for i, c in enumerate(deck) if c.pt == 0]
        tier2_cards = [i for i, c in enumerate(deck) if c.pt in [1, 2]]
        tier3_cards = [i for i, c in enumerate(deck) if c.pt >= 3]

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(tier1_cards)
            rng.shuffle(tier2_cards)
            rng.shuffle(tier3_cards)

        return cls(
            tier1_visible=tuple(tier1_cards[:4]),
            tier2_visible=tuple(tier2_cards[:4]),
            tier3_visible=tuple(tier3_cards[:4]),
            tier1_deck=tuple(tier1_cards[4:]),
            tier2_deck=tuple(tier2_cards[4:]),
            tier3_deck=tuple(tier3_cards[4:]),
        )

    def buy_card(self, card_idx: int) -> 'CardMarket':
        """Return new market with card bought and replaced from deck."""
        # Determine which tier the card is in
        if card_idx in self.tier1_visible:
            visible = list(self.tier1_visible)
            deck_remaining = self.tier1_deck
            visible.remove(card_idx)
            if deck_remaining:
                visible.append(deck_remaining[0])
                deck_remaining = deck_remaining[1:]
            return CardMarket(
                tier1_visible=tuple(visible),
                tier2_visible=self.tier2_visible,
                tier3_visible=self.tier3_visible,
                tier1_deck=deck_remaining,
                tier2_deck=self.tier2_deck,
                tier3_deck=self.tier3_deck,
            )
        elif card_idx in self.tier2_visible:
            visible = list(self.tier2_visible)
            deck_remaining = self.tier2_deck
            visible.remove(card_idx)
            if deck_remaining:
                visible.append(deck_remaining[0])
                deck_remaining = deck_remaining[1:]
            return CardMarket(
                tier1_visible=self.tier1_visible,
                tier2_visible=tuple(visible),
                tier3_visible=self.tier3_visible,
                tier1_deck=self.tier1_deck,
                tier2_deck=deck_remaining,
                tier3_deck=self.tier3_deck,
            )
        elif card_idx in self.tier3_visible:
            visible = list(self.tier3_visible)
            deck_remaining = self.tier3_deck
            visible.remove(card_idx)
            if deck_remaining:
                visible.append(deck_remaining[0])
                deck_remaining = deck_remaining[1:]
            return CardMarket(
                tier1_visible=self.tier1_visible,
                tier2_visible=self.tier2_visible,
                tier3_visible=tuple(visible),
                tier1_deck=self.tier1_deck,
                tier2_deck=self.tier2_deck,
                tier3_deck=deck_remaining,
            )
        # Card not found in any tier
        return self

    def all_visible_cards(self) -> tuple[int, ...]:
        """Get all currently visible card indices."""
        return self.tier1_visible + self.tier2_visible + self.tier3_visible


@dataclass(frozen=True)
class PlayerState:
    """State for one player in a multi-player game."""

    player_id: int
    cards: CardIndices
    bonus: Gems
    gems: Gems
    pts: int
    saved: int  # For statistics

    def total_gem_count(self) -> int:
        """Total gems held (for 10-gem limit)."""
        return sum(self.gems)

    def can_afford(self, card_idx: int) -> bool:
        """Check if player can afford a card."""
        card = deck[card_idx]
        total_resources = tuple(
            self.gems[i] + self.bonus[i] for i in range(COLOR_NUM)
        )
        return all(
            total_resources[i] >= card.cost[i] for i in range(COLOR_NUM)
        )


# ============================================================================
# Heuristic Functions (Speedrun Mode)
# ============================================================================

HeuristicFunc = Callable[['State'], float]


def simple_heuristic(state: 'State') -> float:
    """Simple heuristic that balances saved gems and points with randomness.

    This is the original heuristic from the initial implementation.
    """
    return (state.saved**0.4) * (state.pts**2.5) + randint(1, 100) * 0.01


def balanced_heuristic(state: 'State') -> float:
    """Balanced heuristic that considers multiple factors.

    Factors:
    - Points earned (heavily weighted)
    - Gems saved through bonuses
    - Total gem resources available
    - Card count (buying cards is good)
    - Bonus diversity (having multiple bonus colors)
    """
    # Points are most important
    pts_score = state.pts**2.8

    # Saved gems indicate efficient purchasing
    saved_score = state.saved**0.5

    # Total gems + bonuses indicate purchasing power
    total_gems = sum(state.gems)
    total_bonus = sum(state.bonus)
    resources = (total_gems + total_bonus * 2) ** 0.3

    # Card count matters - more cards = more bonuses
    card_count = len(state.cards) ** 0.6

    # Bonus diversity - having multiple colors is valuable
    unique_bonuses = sum(1 for b in state.bonus if b > 0)
    diversity = unique_bonuses ** 0.4

    # Small random factor for tie-breaking
    noise = randint(1, 100) * 0.01

    return pts_score * 100 + saved_score * 10 + resources * 5 + card_count * 3 + diversity * 2 + noise


def aggressive_heuristic(state: 'State') -> float:
    """Aggressive heuristic that heavily prioritizes points.

    Best for quickly finding high-point solutions.
    """
    pts_score = state.pts**3.2
    saved_score = state.saved**0.3
    total_bonus = sum(state.bonus) ** 0.5
    noise = randint(1, 100) * 0.01

    return pts_score * 200 + saved_score * 5 + total_bonus * 2 + noise


def efficiency_heuristic(state: 'State') -> float:
    """Efficiency heuristic that prioritizes resource optimization.

    Best for finding solutions with minimal moves.
    """
    # Lower weight on points to explore more efficiently
    pts_score = state.pts**2.0

    # Heavy emphasis on saved gems
    saved_score = state.saved**0.7

    # Bonus gems are highly valued
    total_bonus = sum(state.bonus)
    bonus_score = total_bonus**1.2

    # Card diversity
    unique_bonuses = sum(1 for b in state.bonus if b > 0)
    diversity = unique_bonuses ** 0.8

    noise = randint(1, 100) * 0.01

    return pts_score * 50 + saved_score * 30 + bonus_score * 20 + diversity * 10 + noise


def competitive_heuristic(state: 'State') -> float:
    """Competitive heuristic for multi-player games.

    NOTE: This is a placeholder for single-player State objects.
    The real competitive heuristic operates on MultiPlayerState objects.
    """
    # Fall back to balanced heuristic for single-player
    return balanced_heuristic(state)


HEURISTICS: dict[str, HeuristicFunc] = {
    'simple': simple_heuristic,
    'balanced': balanced_heuristic,
    'aggressive': aggressive_heuristic,
    'efficiency': efficiency_heuristic,
    'competitive': competitive_heuristic,
}


class State:
    # Based on Raymond Hettinger's generic puzzle solver:
    # https://rhettinger.github.io/puzzle.html

    def __init__(self, cards, bonus, gems, pts, saved):
        self.cards: CardIndices = cards
        self.bonus: Gems = bonus
        self.gems: Gems = gems
        self.pts: int = pts
        self.saved: int = saved
        self.hash: int = hash((self.cards, self.gems))

    @classmethod
    def newgame(cls) -> 'State':
        no_gems = (0,) * COLOR_NUM
        return State(cards=(), bonus=no_gems, gems=no_gems, pts=0, saved=0)

    def __repr__(self):  # a string representation for printing
        if self.cards:
            return (
                f'{self.gems!r} {"-".join(str(deck[c]) for c in self.cards)}'
            )
        return f'{self.gems!r}'

    def __hash__(self):
        return self.hash

    def __eq__(self, other) -> bool:
        return self.hash == other.hash

    def buy_card(self, card_num: int) -> 'State':
        # Because the simulation uses pre-generated table of possible buys,
        # this method doesn't check if player has enough gems to buy a card.
        cards_mut = list(self.cards)
        # noinspection PyArgumentList
        insort(cards_mut, card_num)
        cards = tuple(cards_mut)
        card = deck[card_num]
        bonus = increase_bonus(self.bonus, card.bonus)
        gems, saved = subtract_with_bonus(self.gems, card.cost, self.bonus)

        return State(
            cards=cards,
            bonus=bonus,
            gems=gems,
            pts=self.pts + card.pt,
            saved=self.saved + saved,
        )

    def __iter__(self):
        # Possible actions:
        # 1. Buy 1 card
        g1, g2, g3, g4, g5 = self.gems
        b1, b2, b3, b4, b5 = self.bonus
        key = (
            min(g1 + b1, MAX_GEMS),
            min(g2 + b2, MAX_GEMS),
            min(g3 + b3, MAX_GEMS),
            min(g4 + b4, MAX_GEMS),
            min(g5 + b5, MAX_GEMS),
        )
        for card_num in get_buys()[key]:
            # Can't buy the same card twice
            if card_num in self.cards:
                continue

            yield self.buy_card(card_num)

        # 2. Take 3 different chips (5*4*3 / 3! = 10 options)
        #    or 2 chips of the same color (5 options)
        # Assuming the best scenario, there's no need to track gems
        # in the pool, since we are only limited by the total number
        # of gems in the game.
        for gems in get_takes()[self.gems]:
            yield State(
                cards=self.cards,
                bonus=self.bonus,
                gems=gems,
                pts=self.pts,
                saved=self.saved,
            )

    def solve(
        self,
        goal_pts: int = 15,
        *,
        use_heuristic: bool = False,
        heuristic_name: str = 'simple',
        beam_width: int = 300_000,
        verbose: bool = True,
    ) -> list['State']:
        """Solve the game using BFS with optional heuristic search.

        Args:
            goal_pts: Target points to reach
            use_heuristic: Whether to use heuristic-guided search
            heuristic_name: Name of heuristic function to use
            beam_width: Maximum states to keep per turn when using heuristic
            verbose: Whether to print progress information

        Returns:
            List of states representing the solution path
        """
        queue: list[State] = [self]
        trail: dict[State, State | None] = {self: None}

        # Select heuristic function
        heuristic = HEURISTICS.get(heuristic_name, simple_heuristic)

        puzzle = self
        turn = 0
        max_pts = 0
        while queue:
            if verbose:
                print(f'{turn=:<10} {queue[0]}')
            next_queue = []
            for puzzle in queue:
                if puzzle.pts > max_pts:
                    max_pts = puzzle.pts
                    if verbose:
                        print(f'{max_pts=:<7} {puzzle}')
                if puzzle.pts >= goal_pts:
                    next_queue.clear()
                    break
                for next_step in puzzle:
                    if next_step in trail:
                        continue
                    trail[next_step] = puzzle
                    next_queue.append(next_step)

            queue = (
                sorted(next_queue, key=heuristic, reverse=True)[:beam_width]
                if use_heuristic
                else next_queue
            )
            turn += 1

        solution = []
        while puzzle:
            solution.append(puzzle)
            puzzle = trail[puzzle]

        return list(reversed(solution))

# ============================================================================
# Multi-Player State (Realistic Mode)
# ============================================================================


class MultiPlayerState:
    """Complete game state for realistic multi-player mode."""

    def __init__(
        self,
        config: GameConfig,
        players: tuple[PlayerState, ...],
        gem_pool: GemPool,
        market: CardMarket,
        current_player: int,
        turn_number: int,
        final_round_triggered: bool = False,
        final_round_player: int | None = None,
    ):
        self.config = config
        self.players = players
        self.gem_pool = gem_pool
        self.market = market
        self.current_player = current_player
        self.turn_number = turn_number
        self.final_round_triggered = final_round_triggered
        self.final_round_player = final_round_player

        # For BFS - hash based on game state
        self.hash = hash((
            self.players,
            self.gem_pool.available,
            self.market.all_visible_cards(),
            self.current_player,
        ))

    @classmethod
    def newgame(cls, config: GameConfig | None = None, shuffle_market: bool = False, seed: int | None = None) -> 'MultiPlayerState':
        """Initialize a new multi-player game."""
        if config is None:
            config = GameConfig(infinite_resources=False)

        no_gems = (0,) * COLOR_NUM

        players = tuple(
            PlayerState(
                player_id=i,
                cards=(),
                bonus=no_gems,
                gems=no_gems,
                pts=0,
                saved=0,
            )
            for i in range(config.num_players)
        )

        return cls(
            config=config,
            players=players,
            gem_pool=GemPool.new_pool(config.gems_per_color),
            market=CardMarket.from_full_deck(shuffle_market, seed),
            current_player=0,
            turn_number=0,
        )

    def __repr__(self):
        """String representation for printing."""
        current = self.players[self.current_player]
        return f'Turn {self.turn_number}, P{self.current_player}: {current.pts}pts, {current.gems!r}'

    def __hash__(self):
        return self.hash

    def __eq__(self, other) -> bool:
        return self.hash == other.hash

    def is_game_over(self) -> bool:
        """Check if game should end."""
        if not self.final_round_triggered:
            # Check if any player reached target
            return any(p.pts >= self.config.target_points for p in self.players)
        else:
            # Final round: game ends when we return to the player who triggered it
            return self.current_player == self.final_round_player

    def get_winner(self) -> int | None:
        """Return winning player ID or None if tied/ongoing."""
        if not self.is_game_over():
            return None

        max_pts = max(p.pts for p in self.players)
        winners = [p for p in self.players if p.pts == max_pts]

        if len(winners) == 1:
            return winners[0].player_id

        # Tiebreaker: fewest cards purchased
        min_cards = min(len(p.cards) for p in winners)
        winners = [p for p in winners if len(p.cards) == min_cards]

        return winners[0].player_id if len(winners) == 1 else None

    def __iter__(self):
        """Generate all possible next states."""
        current = self.players[self.current_player]
        next_player = (self.current_player + 1) % self.config.num_players

        # Action 1: Buy a visible card
        for card_idx in self.market.all_visible_cards():
            if card_idx in current.cards:
                continue

            if not current.can_afford(card_idx):
                continue

            # Purchase card
            card = deck[card_idx]
            new_gems, saved = subtract_with_bonus(
                current.gems, card.cost, current.bonus
            )
            new_bonus = increase_bonus(current.bonus, card.bonus)

            new_player = PlayerState(
                player_id=current.player_id,
                cards=tuple(sorted(list(current.cards) + [card_idx])),
                bonus=new_bonus,
                gems=new_gems,
                pts=current.pts + card.pt,
                saved=current.saved + saved,
            )

            new_market = self.market.buy_card(card_idx)

            # Return gems to pool
            gems_returned = tuple(
                current.gems[i] - new_gems[i]
                for i in range(COLOR_NUM)
            )
            new_pool = self.gem_pool.return_gems(gems_returned)

            # Check if final round should trigger
            final_round_triggered = (
                self.final_round_triggered or
                new_player.pts >= self.config.target_points
            )
            final_round_player = (
                self.final_round_player if self.final_round_triggered
                else self.current_player if final_round_triggered
                else None
            )

            # Update players tuple
            new_players = tuple(
                new_player if i == self.current_player else p
                for i, p in enumerate(self.players)
            )

            yield MultiPlayerState(
                config=self.config,
                players=new_players,
                gem_pool=new_pool,
                market=new_market,
                current_player=next_player,
                turn_number=self.turn_number + 1,
                final_round_triggered=final_round_triggered,
                final_round_player=final_round_player,
            )

        # Action 2: Take gems
        if self.config.infinite_resources:
            # Use existing speedrun logic
            for new_gems in get_takes()[current.gems]:
                new_player = PlayerState(
                    player_id=current.player_id,
                    cards=current.cards,
                    bonus=current.bonus,
                    gems=new_gems,
                    pts=current.pts,
                    saved=current.saved,
                )
                new_players = tuple(
                    new_player if i == self.current_player else p
                    for i, p in enumerate(self.players)
                )
                yield MultiPlayerState(
                    config=self.config,
                    players=new_players,
                    gem_pool=self.gem_pool,
                    market=self.market,
                    current_player=next_player,
                    turn_number=self.turn_number + 1,
                    final_round_triggered=self.final_round_triggered,
                    final_round_player=self.final_round_player,
                )
        else:
            # Realistic gem taking with pool constraints
            yield from self._generate_realistic_gem_takes(current, next_player)

    def _generate_realistic_gem_takes(self, current: PlayerState, next_player: int):
        """Generate valid gem taking moves for realistic mode."""
        from itertools import combinations

        # Generate all possible 3-different gem takes
        available_colors = [i for i in range(COLOR_NUM) if self.gem_pool.available[i] > 0]
        
        for color_combo in combinations(available_colors, 3):
            if current.total_gem_count() + 3 > 10:
                continue  # Would exceed hand limit

            gems_to_take = tuple(
                1 if i in color_combo else 0
                for i in range(COLOR_NUM)
            )

            if all(self.gem_pool.available[i] >= gems_to_take[i] for i in range(COLOR_NUM)):
                new_gems = tuple(
                    current.gems[i] + gems_to_take[i]
                    for i in range(COLOR_NUM)
                )
                new_pool = self.gem_pool.take(gems_to_take)
                new_player = PlayerState(
                    player_id=current.player_id,
                    cards=current.cards,
                    bonus=current.bonus,
                    gems=new_gems,
                    pts=current.pts,
                    saved=current.saved,
                )
                new_players = tuple(
                    new_player if i == self.current_player else p
                    for i, p in enumerate(self.players)
                )
                yield MultiPlayerState(
                    config=self.config,
                    players=new_players,
                    gem_pool=new_pool,
                    market=self.market,
                    current_player=next_player,
                    turn_number=self.turn_number + 1,
                    final_round_triggered=self.final_round_triggered,
                    final_round_player=self.final_round_player,
                )

        # Generate all possible 2-same gem takes
        for color_idx in available_colors:
            if self.gem_pool.available[color_idx] < 4:
                continue  # Need 4 in pool to take 2

            if current.total_gem_count() + 2 > 10:
                continue  # Would exceed hand limit

            gems_to_take = tuple(
                2 if i == color_idx else 0
                for i in range(COLOR_NUM)
            )

            new_gems = tuple(
                current.gems[i] + gems_to_take[i]
                for i in range(COLOR_NUM)
            )
            new_pool = self.gem_pool.take(gems_to_take)
            new_player = PlayerState(
                player_id=current.player_id,
                cards=current.cards,
                bonus=current.bonus,
                gems=new_gems,
                pts=current.pts,
                saved=current.saved,
            )
            new_players = tuple(
                new_player if i == self.current_player else p
                for i, p in enumerate(self.players)
            )
            yield MultiPlayerState(
                config=self.config,
                players=new_players,
                gem_pool=new_pool,
                market=self.market,
                current_player=next_player,
                turn_number=self.turn_number + 1,
                final_round_triggered=self.final_round_triggered,
                final_round_player=self.final_round_player,
            )

    def solve(
        self,
        *,
        use_heuristic: bool = True,  # Force heuristic for realistic mode
        heuristic_name: str = 'competitive',
        beam_width: int = 20_000,  # Lower default for realistic
        verbose: bool = True,
    ) -> list['MultiPlayerState']:
        """Solve multi-player game using BFS with heuristic search."""
        queue: list[MultiPlayerState] = [self]
        trail: dict[MultiPlayerState, MultiPlayerState | None] = {self: None}

        # Define competitive heuristic for multi-player games
        def multi_competitive_heuristic(state: MultiPlayerState) -> float:
            """Competitive heuristic for 2-player games."""
            # Score from perspective of player whose turn just ended
            # (the one who just made this state)
            prev_player_idx = (state.current_player - 1) % state.config.num_players
            me = state.players[prev_player_idx]
            opp = state.players[state.current_player]  # Next player (opponent)

            # Point differential (most important)
            point_diff = (me.pts - opp.pts) ** 2.5 if me.pts > opp.pts else -(opp.pts - me.pts) ** 2.5

            # Resource advantage
            my_resources = sum(me.gems) + sum(me.bonus) * 2
            opp_resources = sum(opp.gems) + sum(opp.bonus) * 2
            resource_diff = (my_resources - opp_resources) ** 0.5 if my_resources > opp_resources else 0

            # Market control (how many visible cards can I afford vs opponent)
            my_affordable = sum(
                1 for card_idx in state.market.all_visible_cards()
                if me.can_afford(card_idx)
            )
            opp_affordable = sum(
                1 for card_idx in state.market.all_visible_cards()
                if opp.can_afford(card_idx)
            )
            market_control = (my_affordable - opp_affordable) * 5

            # Bonus diversity
            my_diversity = sum(1 for b in me.bonus if b > 0)
            diversity_score = my_diversity ** 0.5

            # Random tiebreaker
            noise = randint(1, 100) * 0.01

            return point_diff * 100 + resource_diff * 20 + market_control + diversity_score * 3 + noise

        heuristic = multi_competitive_heuristic

        puzzle = self
        turn = 0
        max_pts = 0

        while queue:
            if verbose and turn % 100 == 0:
                print(f'{turn=:<10} Queue size: {len(queue)}')

            next_queue = []
            for puzzle in queue:
                # Check for game over
                if puzzle.is_game_over():
                    next_queue.clear()
                    break

                # Track progress
                current_max = max(p.pts for p in puzzle.players)
                if current_max > max_pts:
                    max_pts = current_max
                    if verbose:
                        print(f'{max_pts=:<7} {puzzle}')

                # Generate next states
                for next_step in puzzle:
                    if next_step in trail:
                        continue
                    trail[next_step] = puzzle
                    next_queue.append(next_step)

            # Apply beam search
            queue = sorted(next_queue, key=heuristic, reverse=True)[:beam_width]
            turn += 1

            # Safety: stop after reasonable number of turns
            if turn > 1000:
                print("Warning: Reached turn limit (1000)")
                break

        # Reconstruct solution
        solution = []
        while puzzle:
            solution.append(puzzle)
            puzzle = trail.get(puzzle)

        return list(reversed(solution))
