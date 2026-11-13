"""Small one-turn scoring model, separate from Pokemon state tracking."""

import math
from dataclasses import dataclass
from itertools import product

from pokeagent.config import PAYOFF_KO_BONUS


@dataclass(frozen=True)
class TurnEffect:
    """Approximate damage, recovery and protection in fractions of maximum HP."""

    damage: tuple[float, float] = (0.0, 0.0)
    hit_chance: float = 1.0
    heal: float = 0.0
    protect: bool = False
    blocked_by_protect: bool = True

    def outcomes(self) -> list[tuple[float, float, float, bool]]:
        """Return probability, damage, recovery and protection for miss/low/high outcomes."""
        low, high = self.damage
        if not all(math.isfinite(x) for x in (low, high, self.hit_chance, self.heal)):
            raise ValueError("Move effects must be finite")
        if not 0 <= low <= high or not 0 <= self.hit_chance <= 1 or not 0 <= self.heal <= 1:
            raise ValueError("Invalid move effect range or probability")
        return [
            (1 - self.hit_chance, 0.0, 0.0, False),
            (self.hit_chance / 2, low, self.heal, self.protect),
            (self.hit_chance / 2, high, self.heal, self.protect),
        ]


def turn_value(
    our_hp: float,
    opp_hp: float,
    ours: TurnEffect,
    theirs: TurnEffect,
    our_first: float,
    entry_losses: tuple[float, float] = (0.0, 0.0),
    ko_bonus: float = PAYOFF_KO_BONUS,
) -> float:
    """Average utility after HP capping, KO cancellation and hit/miss/order branches.

    Entry losses happen first. This scoring approximation is not a battle simulator.
    """
    if not all(math.isfinite(x) for x in (our_hp, opp_hp, our_first, *entry_losses, ko_bonus)):
        raise ValueError("Turn inputs must be finite")
    if not (0 < our_hp <= 1 and 0 < opp_hp <= 1 and 0 <= our_first <= 1):
        raise ValueError("Invalid starting HP or order probability")
    if min(*entry_losses, ko_bonus) < 0:
        raise ValueError("Entry damage and KO bonus must be non-negative")
    value = 0.0
    for first, order_probability in ((0, our_first), (1, 1 - our_first)):
        for our_outcome, opp_outcome in product(ours.outcomes(), theirs.outcomes()):
            chance = order_probability * our_outcome[0] * opp_outcome[0]
            if chance == 0:
                continue
            hp = [our_hp, opp_hp]
            lost = [min(our_hp, entry_losses[0]), min(opp_hp, entry_losses[1])]
            hp = [hp[i] - lost[i] for i in (0, 1)]
            recovered, protected = [0.0, 0.0], [False, False]
            results, effects = [our_outcome, opp_outcome], [ours, theirs]
            for actor in (first, 1 - first):
                target = 1 - actor
                if hp[actor] <= 0 or hp[target] <= 0:
                    continue
                _, damage, heal, protection = results[actor]
                protected[actor] = protection
                dealt = 0.0
                if not (protected[target] and effects[actor].blocked_by_protect):
                    dealt = min(hp[target], damage)
                hp[target] -= dealt
                lost[target] += dealt
                restored = min(1 - hp[actor], heal)
                hp[actor] += restored
                recovered[actor] += restored
            utility = lost[1] - lost[0] + recovered[0] - recovered[1]
            utility += ko_bonus * (int(hp[1] <= 0) - int(hp[0] <= 0))
            value += chance * utility
    return value
