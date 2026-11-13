"""Convert copied live state into inputs for the provisional one-turn scoring policy."""

import math
from copy import copy, deepcopy

from poke_env.battle import (
    Battle,
    Field,
    Move,
    MoveCategory,
    Pokemon,
    PokemonType,
    SideCondition,
    Status,
    Weather,
)
from poke_env.calc.damage_calc_gen9 import calculate_damage

from pokeagent.decision._turn import TurnEffect
from pokeagent.decision.damage import random_battle_stats


class PayoffUnavailable(ValueError):
    """A table cannot be scored honestly; callers need an explicit legal fallback."""


def scenario_copy(battle: Battle, our_key: str, opp_key: str | None) -> Battle:
    """Copy team objects, activate chosen combatants and fill only their missing stats."""
    result = copy(battle)
    result._team, result._opponent_team = deepcopy(battle.team), deepcopy(battle.opponent_team)
    for team, selected in ((result.team, our_key), (result.opponent_team, opp_key)):
        for key, mon in team.items():
            mon._active = key == selected
        if selected is None:
            continue
        mon = team[selected]
        defaults = random_battle_stats(mon.species, mon.level)
        mon.stats = {
            s: mon.stats.get(s) if mon.stats.get(s) is not None else v for s, v in defaults.items()
        }
        if any(not math.isfinite(v) or v <= 0 for v in mon.stats.values()):
            raise PayoffUnavailable("Active Pokemon stats must be finite and positive")
    return result


def entry_damage(mon: Pokemon, battle: Battle, conditions: dict) -> float:
    """Stealth Rock/Spikes damage with known boots/Magic Guard protection.

    Toxic Spikes, Sticky Web and switch-in abilities are omitted, as disclosed to callers.
    """
    if mon.item == "heavydutyboots" or mon.ability == "magicguard":
        return 0.0
    maximum, damage = mon.stats["hp"], 0
    if SideCondition.STEALTH_ROCK in conditions:
        multiplier = mon.damage_multiplier(PokemonType.ROCK)
        damage += max(1, math.floor(maximum * multiplier / 8)) if multiplier else 0
    if SideCondition.SPIKES in conditions:
        layers = conditions[SideCondition.SPIKES]
        if layers not in (1, 2, 3):
            raise PayoffUnavailable("Spikes must have one to three layers")
        grounded = (
            Field.GRAVITY in battle.fields
            or mon.item == "ironball"
            or (
                PokemonType.FLYING not in mon.types
                and mon.ability != "levitate"
                and mon.item != "airballoon"
            )
        )
        if grounded:
            damage += max(1, math.floor(maximum * (0, 1 / 8, 1 / 6, 1 / 4)[layers]))
    return damage / maximum


def effective_speed(mon: Pokemon, conditions: dict, battle: Battle) -> float:
    """Order approximation with stages, paralysis, Scarf, Tailwind and weather abilities."""
    stage = mon.boosts["spe"]
    speed = mon.stats["spe"] * max(2, 2 + stage) / max(2, 2 - stage)
    if mon.item == "choicescarf":
        speed *= 1.5
    if mon.status == Status.PAR:
        speed *= 0.5
    if SideCondition.TAILWIND in conditions:
        speed *= 2
    ability_weather = {
        "swiftswim": (Weather.RAINDANCE, Weather.PRIMORDIALSEA),
        "chlorophyll": (Weather.SUNNYDAY, Weather.DESOLATELAND),
        "sandrush": (Weather.SANDSTORM,),
        "slushrush": (Weather.SNOW, Weather.HAIL),
    }
    if any(w in battle.weather for w in ability_weather.get(mon.ability, ())):
        speed *= 2
    if mon.item in ("quickclaw", "laggingtail", "fullincense", "custapberry"):
        raise PayoffUnavailable("Unsupported action-order item")
    return speed


def first_probability(battle: Battle, ours: Move | None, theirs: Move | None) -> float:
    """Priority before Speed; ties branch equally. Switches have already occurred."""
    if ours is None or theirs is None:
        return 0.5
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon

    def priority(mon, move):
        value = 0 if move.id == "recharge" else move.priority
        return value + int(mon.ability == "prankster" and move.category == MoveCategory.STATUS)

    priorities = (priority(me, ours), priority(opp, theirs))
    if priorities[0] != priorities[1]:
        return float(priorities[0] > priorities[1])
    speeds = (
        effective_speed(me, battle.side_conditions, battle),
        effective_speed(opp, battle.opponent_side_conditions, battle),
    )
    if speeds[0] == speeds[1]:
        return 0.5
    faster = speeds[0] > speeds[1]
    return float(not faster if Field.TRICK_ROOM in battle.fields else faster)


def move_effect(
    battle: Battle, attacker_key: str, target_key: str | None, move: Move | None
) -> TurnEffect:
    """Immediate damage/recovery/protection estimate; unsupported calculations stay explicit."""
    if move is None or move.id == "recharge":
        return TurnEffect()
    attacker = battle.get_pokemon(attacker_key)
    if attacker.status in (Status.SLP, Status.FRZ):
        raise PayoffUnavailable("Sleep/freeze action availability is not modelled")
    can_act = 0.75 if attacker.status == Status.PAR else 1.0
    if move.is_protect_move:
        if move.id not in ("protect", "detect"):
            raise PayoffUnavailable(f"Unsupported protection side effect: {move.id}")
        return TurnEffect(protect=True, hit_chance=can_act / (3**attacker.protect_counter))
    if move.category == MoveCategory.STATUS:
        return TurnEffect(heal=move.heal, hit_chance=can_act)
    if move.entry.get("flags", {}).get("charge") or move.entry.get("selfdestruct"):
        raise PayoffUnavailable(f"Unsupported multi-turn or self-KO move: {move.id}")
    if target_key is None:
        return TurnEffect()  # Conservative zero credit for attacking an unseen switch target.
    target = battle.get_pokemon(target_key)
    try:
        low, high = calculate_damage(attacker_key, target_key, move, battle)
    except Exception as error:
        raise PayoffUnavailable(f"Damage calculation failed for {move.id}: {error}") from error
    if not all(math.isfinite(x) for x in (low, high)) or not 0 <= low <= high:
        raise PayoffUnavailable(f"Invalid damage range for {move.id}")
    if move.entry.get("accuracy") is True:
        accuracy = 1.0
    else:
        stage = max(-6, min(6, attacker.boosts["accuracy"] - target.boosts["evasion"]))
        accuracy = min(1, move.accuracy * max(3, 3 + stage) / max(3, 3 - stage))
    if attacker.ability == "noguard" or target.ability == "noguard":
        accuracy = 1.0
    return TurnEffect(
        damage=(low / target.stats["hp"], high / target.stats["hp"]),
        hit_chance=accuracy * can_act,
        blocked_by_protect=bool(move.entry.get("flags", {}).get("protect"))
        and not move.breaks_protect,
    )
