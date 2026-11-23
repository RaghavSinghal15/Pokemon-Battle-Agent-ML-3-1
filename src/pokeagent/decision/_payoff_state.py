"""Convert copied live state into inputs for the provisional one-turn scoring policy."""

import json
import math
import random
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache

from poke_env.battle import (
    Battle,
    Effect,
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
from poke_env.data import GenData, to_id_str

from pokeagent.config import (
    PAYOFF_FREEZE_ACT_CHANCE,
    PAYOFF_SLEEP_ACT_CHANCE,
    RANDOM_BATTLE_SETS_PATH,
)
from pokeagent.decision._turn import TurnEffect
from pokeagent.decision.damage import random_battle_stats


class PayoffUnavailable(ValueError):
    """A table cannot be scored honestly; callers need an explicit legal fallback."""


def scenario_copy(battle: Battle, our_key: str, opp_key: str | None) -> Battle:
    """Copy team objects, activate chosen combatants and fill only their missing stats.

    Only the two chosen Pokemon are deep-copied (their stats, HP and boosts may be changed on
    the copy). The others are shallow copies: we only flip their "active" flag and read them.
    Deep-copying both whole teams for every cell was most of the table's running time.
    """
    result = copy(battle)
    result._team = {
        key: deepcopy(mon) if key == our_key else copy(mon) for key, mon in battle.team.items()
    }
    result._opponent_team = {
        key: deepcopy(mon) if key == opp_key else copy(mon)
        for key, mon in battle.opponent_team.items()
    }
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
    can_act = 0.75 if attacker.status == Status.PAR else 1.0
    if attacker.status == Status.SLP and not move.entry.get("sleepUsable"):
        can_act = PAYOFF_SLEEP_ACT_CHANCE
    if attacker.status == Status.FRZ and not move.entry.get("flags", {}).get("defrost"):
        can_act = PAYOFF_FREEZE_ACT_CHANCE
    if move.is_protect_move:
        # Contact punishments and King's Shield's status exception are omitted in v1.
        return TurnEffect(
            protect=True, hit_chance=1 / (3**attacker.protect_counter), act_chance=can_act
        )
    if move.category == MoveCategory.STATUS:
        # Protect blocks status moves aimed at the target, not ones the user uses on itself.
        aimed_at_self = to_id_str(str(move.target).split()[0]) in _SELF_TARGETS
        accuracy = 1.0 if aimed_at_self or move.entry.get("accuracy") is True else move.accuracy
        return TurnEffect(
            heal=heal_fraction(battle, attacker_key, target_key, move),
            act_chance=can_act,
            hit_chance=accuracy,
            blocked_by_protect=not aimed_at_self,
            substitute=move.id == "substitute",
            blocked_by_substitute=_blocked_by_substitute(move, aimed_at_self),
        )
    if move.entry.get("flags", {}).get("charge"):
        # A conservative first-charge-turn approximation, even with weather/Power Herb.
        return TurnEffect()
    self_ko = bool(move.entry.get("selfdestruct"))
    if target_key is None:
        return TurnEffect(self_ko=self_ko, act_chance=can_act)
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
    # Drain and recoil are fractions of the damage dealt, which is measured in the target's HP;
    # convert to the user's HP. Magic Guard (when known) prevents recoil.
    hp_ratio = target.stats["hp"] / attacker.stats["hp"]
    recoil = 0.0 if attacker.ability == "magicguard" else move.recoil * hp_ratio
    return TurnEffect(
        damage=(low / target.stats["hp"], high / target.stats["hp"]),
        hit_chance=accuracy,
        act_chance=can_act,
        self_ko=self_ko,
        blocked_by_protect=bool(move.entry.get("flags", {}).get("protect"))
        and not move.breaks_protect,
        drain=move.drain * hp_ratio,
        recoil=recoil,
        blocked_by_substitute=_blocked_by_substitute(move, False),
    )


def _blocked_by_substitute(move: Move, aimed_at_self: bool) -> bool:
    """Substitute stops moves aimed at its owner, except sound moves (Boomburst, Torch Song)
    and moves that hit the whole side (hazards). Infiltrator is not modelled."""
    target = to_id_str(str(move.target).split()[0])
    if aimed_at_self or target in {"foeside", "allyside", "all"}:
        return False
    return not move.entry.get("flags", {}).get("sound")


# End-of-turn HP changes, as fractions of max HP (Showdown residual order is ignored).
_SAND_IMMUNE_TYPES = {PokemonType.ROCK, PokemonType.GROUND, PokemonType.STEEL}
_SAND_IMMUNE_ABILITIES = {"sandveil", "sandrush", "sandforce", "overcoat", "magicguard"}


def residual(mon: Pokemon | None, battle: Battle) -> float:
    """HP this Pokemon gains (+) or loses (-) at the end of a turn from its item, status and
    the weather. Only known items and abilities count. Grassy Terrain, Leech Seed, Salt Cure
    and other rarer effects are not modelled.
    """
    if mon is None:
        return 0.0
    change = 0.0
    item, ability = mon.item or "", mon.ability or ""
    if item == "leftovers":
        change += 1 / 16
    elif item == "blacksludge":
        change += 1 / 16 if PokemonType.POISON in mon.types else -1 / 8
    status = mon.status
    if ability == "poisonheal" and status in (Status.PSN, Status.TOX):
        change += 1 / 8
    elif status == Status.BRN:
        change -= 1 / 16
    elif status == Status.PSN:
        change -= 1 / 8
    elif status == Status.TOX:
        change -= min(15, mon.status_counter + 1) / 16
    sand = Weather.SANDSTORM in battle.weather
    if (
        sand
        and not _SAND_IMMUNE_TYPES & set(mon.types)
        and ability not in _SAND_IMMUNE_ABILITIES
        and item != "safetygoggles"
    ):
        change -= 1 / 16
    if ability == "magicguard":
        change = max(change, 0.0)  # Magic Guard: no indirect damage
    return change


def substitute_hp(mon: Pokemon | None) -> float:
    """HP left in a Substitute that is already up. The log does not say how damaged it is,
    so a full one (1/4 of max HP) is assumed."""
    return 0.25 if mon is not None and Effect.SUBSTITUTE in mon.effects else 0.0


# Healing moves whose amount the move data does not state (it depends on the weather or on the
# target), so poke-env's Move.heal reports 0 for them.
_WEATHER_HEAL = {"moonlight", "morningsun", "synthesis"}
_SUN = {Weather.SUNNYDAY, Weather.DESOLATELAND}


def heal_fraction(battle: Battle, user_key: str, target_key: str | None, move: Move) -> float:
    """Fraction of the user's max HP a status move restores this turn.

    Moonlight, Morning Sun and Synthesis heal 1/2, 2/3 in sun and 1/4 in other weather; Shore
    Up heals 1/2, 2/3 in sand; Rest heals fully (and puts the user to sleep, valued as a
    lasting effect); Strength Sap heals the target's Attack stat. Wish (next turn) is ignored.
    """
    weather = set(battle.weather)
    if move.id in _WEATHER_HEAL:
        if weather & _SUN:
            return 2 / 3
        return 1 / 4 if weather else 1 / 2
    if move.id == "shoreup":
        return 2 / 3 if Weather.SANDSTORM in weather else 1 / 2
    if move.id == "rest":
        return 1.0
    if move.id == "strengthsap" and target_key is not None:
        user, target = battle.get_pokemon(user_key), battle.get_pokemon(target_key)
        stage = target.boosts.get("atk", 0)
        attack = target.stats["atk"] * (2 + max(stage, 0)) / (2 + max(-stage, 0))
        return min(1.0, attack / user.stats["hp"])
    return move.heal


# ---------------------------------------------------------------------------------------
# Lasting effects (F8, F9, F10): what a move leaves behind for later turns.

_SELF_TARGETS = {"self", "allyside", "allies", "adjacentallyorself", "allyteam", "all"}
_HAZARD_LAYERS = {
    SideCondition.STEALTH_ROCK: 1,
    SideCondition.SPIKES: 3,
    SideCondition.TOXIC_SPIKES: 2,
    SideCondition.STICKY_WEB: 1,
}
# Moves that clear hazards: from the user's side only, or from both sides.
_HAZARD_REMOVAL = {"rapidspin": "own", "mortalspin": "own", "tidyup": "both", "defog": "both"}
# Types that can never get a status (Electric: paralysis; Fire: burn; Poison/Steel: poison).
_STATUS_IMMUNE_TYPES = {
    Status.PAR: {PokemonType.ELECTRIC},
    Status.BRN: {PokemonType.FIRE},
    Status.PSN: {PokemonType.POISON, PokemonType.STEEL},
    Status.TOX: {PokemonType.POISON, PokemonType.STEEL},
}
# Known abilities that stop a status move (only applied when the ability is known).
_STATUS_BLOCKING_ABILITIES = {"goodasgold", "magicbounce", "purifyingsalt", "comatose"}
_STATUS_ABILITY_IMMUNITY = {
    Status.PAR: {"limber"},
    Status.BRN: {"waterveil", "waterbubble", "thermalexchange"},
    Status.PSN: {"immunity", "pastelveil"},
    Status.TOX: {"immunity", "pastelveil"},
    Status.SLP: {"insomnia", "vitalspirit", "sweetveil"},
}


@dataclass(frozen=True)
class LastingEffect:
    """A copy of the battle after a move's lasting effect, and what the effect depends on."""

    battle: Battle
    needs_user: bool  # the user's own stat changes vanish if it faints
    needs_target: bool  # status or stat drops on the target vanish if it faints
    changes_pokemon: bool = True  # stat changes or status (valued by looking one turn ahead)
    changes_field: bool = False  # hazards set or removed (valued over the remaining Pokemon)


def _status_can_land(target: Pokemon, status: Status, move: Move) -> bool:
    if target.status is not None:
        return False
    types = set(target.types)
    if types & _STATUS_IMMUNE_TYPES.get(status, set()):
        return False
    if move.entry.get("flags", {}).get("powder") and PokemonType.GRASS in types:
        return False
    if (
        move.category == MoveCategory.STATUS
        and not move.entry.get("ignoreImmunity")
        and target.damage_multiplier(move.type) == 0
    ):
        return False  # Thunder Wave into a Ground type
    ability = target.ability
    return not (
        ability in _STATUS_BLOCKING_ABILITIES and move.category == MoveCategory.STATUS
    ) and ability not in _STATUS_ABILITY_IMMUNITY.get(status, set())


def _boost(mon: Pokemon, changes: dict | None) -> bool:
    changed = False
    for stat, delta in (changes or {}).items():
        if stat not in mon._boosts:
            continue
        new = max(-6, min(6, mon._boosts[stat] + delta))
        changed |= new != mon._boosts[stat]
        mon._boosts[stat] = new
    return changed


def lasting_effect(
    battle: Battle, user_key: str, target_key: str, move: Move
) -> LastingEffect | None:
    """Apply what `move` leaves behind to a copy of the battle; None if it leaves nothing.

    Covered: stat changes on the user (Swords Dance, Draco Meteor's drop, Flame Charge) or the
    target (Charm, Icy Wind); a status from a status move (Thunder Wave, Will-O-Wisp, Toxic,
    Spore), respecting immunities; entry hazards (Stealth Rock, Spikes ...); hazard removal.
    Only effects that are certain when the move hits (chance 100) are applied.
    """
    ours = user_key in battle.team
    result = copy(battle)
    result._team, result._opponent_team = deepcopy(battle.team), deepcopy(battle.opponent_team)
    result._side_conditions = dict(battle.side_conditions)
    result._opponent_side_conditions = dict(battle.opponent_side_conditions)
    user, target = result.get_pokemon(user_key), result.get_pokemon(target_key)
    own_side = result._side_conditions if ours else result._opponent_side_conditions
    foe_side = result._opponent_side_conditions if ours else result._side_conditions
    status_move = move.category == MoveCategory.STATUS
    aimed_at_self = to_id_str(str(move.target).split()[0]) in _SELF_TARGETS
    user_changed = target_changed = field_changed = False

    # Stat changes.
    if status_move and aimed_at_self:
        user_changed |= _boost(user, move.boosts)
    elif status_move:
        target_changed |= _boost(target, move.boosts)
    user_changed |= _boost(user, move.self_boost)
    for secondary in move.secondary or []:
        if secondary.get("chance", 0) < 100:
            continue
        user_changed |= _boost(user, secondary.get("self", {}).get("boosts"))
        target_changed |= _boost(target, secondary.get("boosts"))

    # Rest: the user sleeps (its full heal is counted this turn). It fails at full HP.
    if move.id == "rest" and user.status is None and user.current_hp_fraction < 1:
        user._status = Status.SLP
        user_changed = True

    # Status from a status move.
    if status_move and move.status is not None and _status_can_land(target, move.status, move):
        target._status = move.status
        target_changed = True

    # A Substitute stays up after this turn (its HP cost is paid this turn, in turn_value).
    if move.id == "substitute" and Effect.SUBSTITUTE not in user.effects:
        user._effects[Effect.SUBSTITUTE] = 0
        user_changed = True

    # Entry hazards on the foe's side, and hazard removal.
    condition = move.side_condition
    if condition in _HAZARD_LAYERS and foe_side.get(condition, 0) < _HAZARD_LAYERS[condition]:
        foe_side[condition] = foe_side.get(condition, 0) + 1
        field_changed = True
    sides = {"own": (own_side,), "both": (own_side, foe_side)}.get(_HAZARD_REMOVAL.get(move.id), ())
    for side in sides:
        for hazard in _HAZARD_LAYERS:
            if side.pop(hazard, None) is not None:
                field_changed = True

    if not (user_changed or target_changed or field_changed):
        return None
    return LastingEffect(
        result,
        needs_user=user_changed and not (target_changed or field_changed),
        needs_target=target_changed and not (user_changed or field_changed),
        changes_pokemon=user_changed or target_changed,
        changes_field=field_changed,
    )


# Entry damage for a Pokemon we have not seen yet: neutral to Stealth Rock, grounded.
def _unseen_entry_damage(conditions: dict) -> float:
    damage = 1 / 8 if SideCondition.STEALTH_ROCK in conditions else 0.0
    layers = conditions.get(SideCondition.SPIKES, 0)
    return damage + (0, 1 / 8, 1 / 6, 1 / 4)[min(layers, 3)]


def hazard_value(before: Battle, after: Battle, unseen_opponents: int) -> float:
    """Change in hazard damage still to come, as HP fractions summed over the Pokemon that
    have yet to switch in (positive = good for us).

    Each remaining reserve must switch in at least once to be used, so each pays the entry
    damage once. Unseen opponents are counted with neutral damage. Current HP caps the loss.
    """

    def total(battle: Battle, ours: bool) -> float:
        team = battle.team if ours else battle.opponent_team
        conditions = battle.side_conditions if ours else battle.opponent_side_conditions
        loss = 0.0
        for mon in team.values():
            if mon.active or mon.fainted or mon.max_hp <= 0 or mon.current_hp_fraction <= 0:
                continue
            probe = deepcopy(mon)
            defaults = random_battle_stats(probe.species, probe.level)
            probe.stats = {s: probe.stats.get(s) or v for s, v in defaults.items()}
            loss += min(probe.current_hp_fraction, entry_damage(probe, battle, conditions))
        if not ours:
            loss += unseen_opponents * _unseen_entry_damage(conditions)
        return loss

    return (total(after, False) - total(before, False)) - (total(after, True) - total(before, True))


UNSEEN_NAME_PREFIX = "Unseen-"  # names of stand-ins for Pokemon the opponent has not shown


@lru_cache(maxsize=2)
def _random_battle_levels(path) -> dict[str, int]:
    """Every Gen 9 Random Battle species with its level (from the public sets file).

    Empty if the file is missing (CI has no Showdown checkout): no stand-ins are drawn then.
    """
    try:
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
    except FileNotFoundError:
        return {}
    dex = GenData.from_gen(9).pokedex
    return {
        to_id_str(name): entry["level"]
        for name, entry in data.items()
        if to_id_str(name) in dex and "level" in entry
    }


def unseen_stand_ins(battle: Battle, count: int, seed: int) -> dict[str, Pokemon]:
    """Stand-in Pokemon for the opponent's unseen team members, keyed like team members.

    Random Battle teams are drawn from a pool of about 500 species with at most one of each
    base species per team, so an unseen Pokemon is (roughly) any pool species not already
    on their team. We draw `count` of them, the same ones for every turn of a battle, and
    give each the Random Battle level and full HP. Item and ability stay unknown, as for any
    unrevealed Pokemon. Only public information is used.
    """
    levels = _random_battle_levels(RANDOM_BATTLE_SETS_PATH)
    dex = GenData.from_gen(9).pokedex
    taken = set()
    for mon in battle.opponent_team.values():
        entry = dex.get(mon.species, {})
        taken.add(to_id_str(entry.get("baseSpecies", "")) or mon.species)
    pool = []
    for species in sorted(levels):
        base = to_id_str(dex[species].get("baseSpecies", "")) or species
        if base not in taken:
            pool.append(species)
    # A string seed gives the same draw on every platform and Python run.
    rng = random.Random(f"{seed}-{battle.battle_tag}")
    # Keys look like real ones ("p2: Name") because poke-env reads the side from the prefix.
    role = battle.opponent_role or "p2"
    stand_ins = {}
    for species in rng.sample(pool, min(count, len(pool))):
        mon = Pokemon(gen=9, species=species)
        mon._level = levels[species]
        mon._max_hp = mon._current_hp = 100  # opponent HP is shown in percent
        stand_ins[f"{role}: {UNSEEN_NAME_PREFIX}{species}"] = mon
    return stand_ins
