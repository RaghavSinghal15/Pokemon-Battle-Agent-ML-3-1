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
    act_chance: float = 1.0
    self_ko: bool = False
    # Lasting effect (F8): change in OUR next-turn table value if this move takes effect
    # (positive for our Swords Dance, negative for their Thunder Wave on us). It counts only
    # when the move lands, and only if the user / target is still standing when required.
    future: float = 0.0
    future_needs_user: bool = False
    future_needs_target: bool = False
    # F11. drain / recoil: HP the user gains / loses per unit of damage dealt, both already in
    # the user's max-HP fractions (Giga Drain heals half the damage, Brave Bird loses a third).
    drain: float = 0.0
    recoil: float = 0.0
    substitute: bool = False  # the user puts up a Substitute (costs 1/4 of its max HP)
    blocked_by_substitute: bool = True  # False for moves that ignore it (sound, self-targeted)

    def outcomes(self) -> list[tuple[float, float, float, bool, bool, bool]]:
        """Branch on inability to act, then miss/low/high; self-KO requires an attempt.

        Each outcome is (probability, damage, heal, protect, self_ko, landed).
        """
        low, high = self.damage
        if not all(
            math.isfinite(x)
            for x in (
                low,
                high,
                self.hit_chance,
                self.heal,
                self.act_chance,
                self.future,
                self.drain,
                self.recoil,
            )
        ):
            raise ValueError("Move effects must be finite")
        if (
            not 0 <= low <= high
            or not 0 <= self.hit_chance <= 1
            or not 0 <= self.heal <= 1
            or not 0 <= self.act_chance <= 1
            or self.drain < 0
            or self.recoil < 0
        ):
            raise ValueError("Invalid move effect range or probability")
        hit = self.act_chance * self.hit_chance / 2
        return [
            (1 - self.act_chance, 0.0, 0.0, False, False, False),
            (self.act_chance * (1 - self.hit_chance), 0.0, 0.0, False, self.self_ko, False),
            (hit, low, self.heal, self.protect, self.self_ko, True),
            (hit, high, self.heal, self.protect, self.self_ko, True),
        ]


def turn_value(
    our_hp: float,
    opp_hp: float,
    ours: TurnEffect,
    theirs: TurnEffect,
    our_first: float,
    entry_losses: tuple[float, float] = (0.0, 0.0),
    ko_bonus: float = PAYOFF_KO_BONUS,
    residuals: tuple[float, float] = (0.0, 0.0),
    substitutes: tuple[float, float] = (0.0, 0.0),
    end_value=None,
    start_value: float | None = None,
) -> float:
    """Average utility after HP capping, KO cancellation and hit/miss/order branches.

    Entry losses happen first. residuals are each side's end-of-turn HP change (Leftovers
    +1/16, burn -1/16 ...), applied to a Pokemon still standing. substitutes are the HP left in
    each side's existing Substitute. This scoring approximation is not a battle simulator.

    end_value (V4): a function (our HP, their HP at the end of the turn) -> P(win). If given, an
    outcome is worth end_value(end) - start_value instead of HP traded plus the KO bonus.
    start_value is P(win) now; every row of a table must share it, so pass it explicitly when
    rows face different Pokemon (switches). It defaults to end_value at the starting HP.
    """
    numbers = (our_hp, opp_hp, our_first, *entry_losses, ko_bonus, *residuals, *substitutes)
    if not all(math.isfinite(x) for x in numbers):
        raise ValueError("Turn inputs must be finite")
    if not (0 < our_hp <= 1 and 0 < opp_hp <= 1 and 0 <= our_first <= 1):
        raise ValueError("Invalid starting HP or order probability")
    if min(*entry_losses, ko_bonus, *substitutes) < 0:
        raise ValueError("Entry damage, KO bonus and Substitute HP must be non-negative")
    value = 0.0
    if end_value is not None and start_value is None:
        start_value = end_value(our_hp, opp_hp)
    for first, order_probability in ((0, our_first), (1, 1 - our_first)):
        for our_outcome, opp_outcome in product(ours.outcomes(), theirs.outcomes()):
            chance = order_probability * our_outcome[0] * opp_outcome[0]
            if chance == 0:
                continue
            hp = [our_hp, opp_hp]
            lost = [min(our_hp, entry_losses[0]), min(opp_hp, entry_losses[1])]
            hp = [hp[i] - lost[i] for i in (0, 1)]
            recovered, protected = [0.0, 0.0], [False, False]
            landed = [False, False]
            sub = list(substitutes)
            results, effects = [our_outcome, opp_outcome], [ours, theirs]
            for actor in (first, 1 - first):
                target = 1 - actor
                if hp[actor] <= 0 or hp[target] <= 0:
                    continue
                _, damage, heal, protection, self_ko, hit = results[actor]
                protected[actor] = protection
                dealt = 0.0
                effect = effects[actor]
                blocked = protected[target] and effect.blocked_by_protect
                # A Substitute takes the hit instead of the Pokemon, and stops effects aimed at it.
                behind_sub = hit and sub[target] > 0 and effect.blocked_by_substitute
                landed[actor] = (
                    hit and not blocked and not (behind_sub and effect.future_needs_target)
                )
                if behind_sub and not blocked:
                    sub[target] = max(0.0, sub[target] - damage)
                elif not blocked:
                    dealt = min(hp[target], damage)
                hp[target] -= dealt
                lost[target] += dealt
                restored = min(1 - hp[actor], heal + effect.drain * dealt)
                hp[actor] += restored
                recovered[actor] += restored
                recoil = min(hp[actor], effect.recoil * dealt)
                hp[actor] -= recoil
                lost[actor] += recoil
                # Substitute: fails at 1/4 HP or below, or if one is already up.
                if effect.substitute and hit and sub[actor] == 0 and hp[actor] > 0.25:
                    hp[actor] -= 0.25
                    lost[actor] += 0.25
                    sub[actor] = 0.25
                if self_ko:
                    lost[actor] += hp[actor]
                    hp[actor] = 0.0
            for side in (0, 1):
                if hp[side] <= 0 or residuals[side] == 0:
                    continue
                if residuals[side] > 0:
                    restored = min(1 - hp[side], residuals[side])
                    hp[side] += restored
                    recovered[side] += restored
                else:
                    taken = min(hp[side], -residuals[side])
                    hp[side] -= taken
                    lost[side] += taken
            if end_value is not None:
                utility = end_value(hp[0], hp[1]) - start_value
            else:
                utility = lost[1] - lost[0] + recovered[0] - recovered[1]
                utility += ko_bonus * (int(hp[1] <= 0) - int(hp[0] <= 0))
            for actor, effect in ((0, ours), (1, theirs)):
                if not landed[actor] or effect.future == 0:
                    continue
                if effect.future_needs_user and hp[actor] <= 0:
                    continue  # e.g. our boosts are lost if we are knocked out this turn
                if effect.future_needs_target and hp[1 - actor] <= 0:
                    continue  # e.g. a status on a Pokemon that fainted does nothing
                utility += effect.future
            value += chance * utility
    return value
