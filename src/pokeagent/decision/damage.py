"""Approximate Gen 9 move damage using only our live view of the battle.

This is F3's input to the future payoff builder, not a complete move-ranking strategy.
Unknown opponent stats use the default Random Battle spread from foundation check 5.
Revealed items, abilities, boosts, types and field conditions stay with the calculation.
"""

import math
from copy import copy, deepcopy
from dataclasses import dataclass

from poke_env.battle import Battle
from poke_env.calc.damage_calc_gen9 import calculate_damage
from poke_env.data import GenData
from poke_env.stats import compute_raw_stats

_STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")


@dataclass(frozen=True)
class DamageEstimate:
    """Damage as percent of estimated maximum HP, before capping at remaining HP.

    expected_percent is the midpoint of the calculator's damage range, conditional on
    hitting without a critical hit. It is an approximation, not an average of all rolls.
    Misses, secondary effects and the value of status moves belong in F4's payoff policy.
    None values with an error mean calculation failed; zero means a calculated zero.
    """

    min_percent: float | None
    max_percent: float | None
    expected_percent: float | None
    error: str | None = None


def random_battle_stats(species: str, level: int) -> dict[str, int]:
    """Default Gen 9 spread: 85 EVs, 31 IVs, neutral nature, in HP/Atk/Def/SpA/SpD/Spe order.

    EVs and IVs are the game's stat adjustments. Some sets reduce Attack, Speed or HP;
    this helper deliberately keeps the approximation agreed in foundation check 5.
    """
    if not 1 <= level <= 100:
        raise ValueError("Pokemon level must be between 1 and 100")
    raw = compute_raw_stats(species, [85] * 6, [31] * 6, level, "serious", GenData.from_gen(9))
    return dict(zip(_STAT_NAMES, raw, strict=True))


def estimate_move_damage(battle: Battle) -> dict[str, DamageEstimate]:
    """Estimate each available move against the current opponent, keyed by move id.

    Return an empty mapping for forced replacements or missing/fainted active Pokemon.
    Only Gen 9 singles are supported. Unknown opponent stats are filled on a private copy;
    our known stats and all revealed opponent details are preserved. No peer/oracle battle,
    hidden set data or future log lines are read. A failed move gets an explicit error so
    callers can fall back without confusing failure with an immunity.
    """
    if not isinstance(battle, Battle) or battle.gen != 9:
        raise ValueError("Damage estimates require a Gen 9 singles Battle")
    attacker = battle.active_pokemon
    defender = battle.opponent_active_pokemon
    if (
        battle.force_switch
        or attacker is None
        or defender is None
        or attacker.fainted
        or defender.fainted
        or not battle.available_moves
    ):
        return {}

    estimates = {}
    try:
        attacker_key = next(key for key, mon in battle.team.items() if mon is attacker)
        defender_key = next(key for key, mon in battle.opponent_team.items() if mon is defender)
        # Copy only the active Pokemon and team mappings, not the growing replay log.
        calculation = copy(battle)
        calculation._team = dict(battle.team)
        calculation._opponent_team = dict(battle.opponent_team)
        calculation._team[attacker_key] = deepcopy(attacker)
        target = deepcopy(defender)
        calculation._opponent_team[defender_key] = target
        defaults = random_battle_stats(target.species, target.level)
        target.stats = dict(target.stats)
        for stat in _STAT_NAMES:
            if target.stats.get(stat) is None:
                target.stats[stat] = defaults[stat]
        max_hp = target.stats["hp"]
        if max_hp is None or not math.isfinite(max_hp) or max_hp <= 0:
            raise ValueError("Defender maximum HP must be positive")
    except Exception as error:
        # Invalid/unsupported state must remain visible to callers, without stalling a game.
        failure = DamageEstimate(None, None, None, f"{type(error).__name__}: {error}")
        return dict.fromkeys((move.id for move in battle.available_moves), failure)

    for move in battle.available_moves:
        try:
            low, high = calculate_damage(attacker_key, defender_key, move, calculation)
            if not math.isfinite(low) or not math.isfinite(high) or not 0 <= low <= high:
                raise ValueError("Calculator returned an invalid damage range")
            low_percent = 100 * low / max_hp
            high_percent = 100 * high / max_hp
            estimates[move.id] = DamageEstimate(
                low_percent, high_percent, (low_percent + high_percent) / 2
            )
        except Exception as error:
            estimates[move.id] = DamageEstimate(
                None, None, None, f"{type(error).__name__}: {error}"
            )
    return estimates
