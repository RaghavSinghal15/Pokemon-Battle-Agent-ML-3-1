"""Build model inputs from the public state before either player acts."""

from functools import cache
from math import prod

from poke_env.data import GenData, to_id_str

from pokeagent.config import RANDOM_BATTLE_GENERATION
from pokeagent.interfaces import Action, BeliefState, FeatureRow, PokemonView, PublicState, SideView
from pokeagent.models.labels import action_class

FEATURE_NAMES = [
    "turn",
    "my_hp",
    "opp_hp",
    "my_remaining",
    "opp_remaining",
    "my_unrevealed",
    "opp_unrevealed",
    "my_boost_sum",
    "opp_boost_sum",
    "my_status",
    "opp_status",
    "my_paralyzed",
    "opp_paralyzed",
    "my_hazard_layers",
    "opp_hazard_layers",
    "my_tera_used",
    "opp_tera_used",
    "my_revealed_moves",
    "opp_revealed_moves",
    "my_type_matchup",
    "opp_type_matchup",
    "speed_advantage",
    "trick_room",
    "opp_choice_item_known",
    "opp_best_setup_move_probability",
    "opp_best_recovery_move_probability",
    "opp_last_attack",
    "opp_last_non_damaging",
    "opp_last_switch",
    # D7 (#90): why a player might use a non-damaging move NOW, from beliefs about its moves.
    "opp_expected_status_moves",
    "opp_status_move_useful",
    "opp_hazard_move_useful",
    "opp_hazard_removal_useful",
    "opp_protect_probability",
    "opp_substitute_probability",
    "opp_pivot_move_probability",
]

_HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
_STATS = {"atk", "def", "spa", "spd", "spe"}
_CHOICE_ITEMS = {"choiceband", "choicescarf", "choicespecs"}
# Hazard-setting moves and how many layers each hazard can have.
_HAZARD_MOVES = {"stealthrock": 1, "spikes": 3, "toxicspikes": 2, "stickyweb": 1}
_HAZARD_REMOVAL = {"rapidspin", "defog", "mortalspin", "tidyup", "courtchange"}
_PROTECT_MOVES = {
    "protect",
    "detect",
    "kingsshield",
    "spikyshield",
    "banefulbunker",
    "silktrap",
    "burningbulwark",
    "obstruct",
}
# Moves that let the user switch out while acting (damaging or not).
_PIVOT_MOVES = {
    "uturn",
    "voltswitch",
    "flipturn",
    "partingshot",
    "teleport",
    "chillyreception",
    "shedtail",
}
# Types that can never get a status: Electric can't be paralysed, Fire can't be burned,
# Poison and Steel can't be poisoned. (Grass ignores powder moves: checked separately.)
_STATUS_IMMUNE_TYPES = {
    "par": {"ELECTRIC"},
    "brn": {"FIRE"},
    "psn": {"POISON", "STEEL"},
    "tox": {"POISON", "STEEL"},
}
# Standard Random Battle spread used for the rough speed comparison.
_SPEED_EV = 85
_SPEED_IV = 31


def _side_features(side: SideView, active: PokemonView | None) -> tuple[float, ...]:
    status = to_id_str(active.status or "") if active else ""
    return (
        float(active.hp_fraction if active else 0),
        float(side.n_remaining),
        float(side.n_unrevealed),
        float(
            sum(value for stat, value in active.boosts.items() if stat in _STATS) if active else 0
        ),
        float(bool(status)),
        float(status == "par"),
        float(
            sum(value for key, value in side.side_conditions.items() if to_id_str(key) in _HAZARDS)
        ),
        float(side.tera_used),
        float(len(active.revealed_moves) if active else 0),
    )


def _base_types(dex: GenData, pokemon: PokemonView | None) -> tuple[str, ...]:
    if pokemon is None:
        return ()
    entry = dex.pokedex.get(to_id_str(pokemon.species), {})
    return tuple(name.upper() for name in entry.get("types", []))


def _type_matchup(
    dex: GenData, attacker: PokemonView | None, defender: PokemonView | None
) -> float:
    attack_types = _base_types(dex, attacker)
    defense_types = _base_types(dex, defender)
    if attacker and attacker.terastallized and attacker.tera_type:
        tera = attacker.tera_type.upper()
        if tera in dex.type_chart:
            attack_types += (tera,)
    if defender and defender.terastallized and defender.tera_type:
        tera = defender.tera_type.upper()
        if tera in dex.type_chart:
            defense_types = (tera,)
    if not attack_types or not defense_types:
        return 1.0
    # poke-env indexes the chart by defending type first.
    return max(prod(dex.type_chart[d][a] for d in defense_types) for a in attack_types)


def _speed(dex: GenData, pokemon: PokemonView | None) -> float | None:
    if pokemon is None or to_id_str(pokemon.species) not in dex.pokedex:
        return None
    base = dex.pokedex[to_id_str(pokemon.species)]["baseStats"]["spe"]
    speed = (2 * base + _SPEED_IV + _SPEED_EV // 4) * pokemon.level // 100 + 5
    stages = pokemon.boosts.get("spe", 0)
    speed = speed * (2 + max(stages, 0)) // (2 + max(-stages, 0))
    if to_id_str(pokemon.status or "") == "par":
        speed //= 2
    if not pokemon.item_removed and to_id_str(pokemon.revealed_item or "") == "choicescarf":
        speed = speed * 3 // 2
    return speed


@cache
def _move_signals(move_id: str) -> tuple[bool, bool]:
    move = GenData.from_gen(RANDOM_BATTLE_GENERATION).moves.get(move_id, {})
    boosts = move.get("boosts", {})
    setup = move.get("target") == "self" and any(boosts.get(stat, 0) > 0 for stat in _STATS)
    recovery = move.get("target") == "self" and bool(
        move.get("heal") or move.get("flags", {}).get("heal")
    )
    return setup, recovery


@cache
def _status_move(move_id: str) -> tuple[str, str, bool, bool] | None:
    """(status, move type, is powder, ignores type immunity) for a status-inflicting move."""
    move = GenData.from_gen(RANDOM_BATTLE_GENERATION).moves.get(move_id, {})
    if move.get("category") != "Status" or move.get("target") not in {"normal", "any"}:
        return None
    status = move.get("status") or ("slp" if move.get("volatileStatus") == "yawn" else None)
    if status is None:
        return None
    return (
        status,
        move.get("type", "").upper(),
        bool(move.get("flags", {}).get("powder")),
        bool(move.get("ignoreImmunity")),
    )


def _defending_types(dex: GenData, pokemon: PokemonView | None) -> tuple[str, ...]:
    if pokemon and pokemon.terastallized and pokemon.tera_type:
        tera = pokemon.tera_type.upper()
        if tera in dex.type_chart:
            return (tera,)
    return _base_types(dex, pokemon)


def _status_lands(dex: GenData, move_id: str, target: PokemonView | None) -> bool:
    """Could this status move affect the target now (no status yet, no type immunity)?"""
    info = _status_move(move_id)
    if info is None or target is None or target.status:
        return False
    status, move_type, powder, ignores_immunity = info
    types = set(_defending_types(dex, target))
    if types & _STATUS_IMMUNE_TYPES.get(status, set()):
        return False
    if powder and "GRASS" in types:
        return False
    if not ignores_immunity and types and all(t in dex.type_chart for t in types):
        # poke-env indexes the chart by defending type first; 0 means immune (Thunder Wave
        # into a Ground type).
        if prod(dex.type_chart[t].get(move_type, 1) for t in types) == 0:
            return False
    return True


def _threat_features(dex: GenData, state: PublicState, belief) -> tuple[float, ...]:
    """D7: how useful the acting player's non-damaging options are in this position.

    Uses beliefs (the probability it has each move), so moves not yet seen still count.
    "Useful" features take the most likely move that would actually work right now.
    """
    moves = belief.move_probs if belief else {}
    target = state.me.active  # the acting player's target is the observer's active Pokemon
    # Expected number of non-damaging moves in its set of four.
    expected_status = sum(p for m, p in moves.items() if _move_is_status(m))
    status_useful = max((p for m, p in moves.items() if _status_lands(dex, m, target)), default=0.0)
    their_hazards = {to_id_str(k): v for k, v in state.me.side_conditions.items()}
    hazard_useful = 0.0
    if state.me.n_remaining > 1:  # hazards only matter if we still have Pokemon to switch in
        hazard_useful = max(
            (
                p
                for m, p in moves.items()
                if m in _HAZARD_MOVES and their_hazards.get(m, 0) < _HAZARD_MOVES[m]
            ),
            default=0.0,
        )
    own_hazards = sum(v for k, v in state.opp.side_conditions.items() if to_id_str(k) in _HAZARDS)
    removal_useful = (
        max((p for m, p in moves.items() if m in _HAZARD_REMOVAL), default=0.0)
        if own_hazards
        else 0.0
    )
    return (
        float(expected_status),
        float(status_useful),
        float(hazard_useful),
        float(removal_useful),
        float(max((p for m, p in moves.items() if m in _PROTECT_MOVES), default=0.0)),
        float(moves.get("substitute", 0.0)),
        float(max((p for m, p in moves.items() if m in _PIVOT_MOVES), default=0.0)),
    )


@cache
def _move_is_status(move_id: str) -> bool:
    return GenData.from_gen(RANDOM_BATTLE_GENERATION).moves.get(move_id, {}).get("category") == (
        "Status"
    )


def _action_class(dex: GenData, action: Action | None) -> str | None:
    """The shared label rule (models/labels.py), or None when there is no usable action."""
    if action is None:
        return None
    try:
        return action_class(action)
    except ValueError:
        return None


def build_features(state: PublicState, beliefs: BeliefState) -> FeatureRow:
    """Return the ordered numeric columns documented in docs/features.md."""
    dex = GenData.from_gen(RANDOM_BATTLE_GENERATION)
    me, opp = state.me.active, state.opp.active
    my_speed, opp_speed = _speed(dex, me), _speed(dex, opp)
    speed_advantage = 0
    if my_speed is not None and opp_speed is not None:
        speed_advantage = int(my_speed > opp_speed) - int(my_speed < opp_speed)
        if state.field.trick_room:
            speed_advantage = -speed_advantage

    belief = beliefs.get(opp.species) if opp else None
    setup = recovery = 0.0
    if belief:
        for move, probability in belief.move_probs.items():
            is_setup, is_recovery = _move_signals(move)
            if is_setup:
                setup = max(setup, probability)
            if is_recovery:
                recovery = max(recovery, probability)

    last_class = _action_class(dex, state.opp_last_action)

    side_columns = tuple(
        value
        for pair in zip(_side_features(state.me, me), _side_features(state.opp, opp), strict=True)
        for value in pair
    )
    values = (
        float(state.turn),
        *side_columns,
        _type_matchup(dex, me, opp),
        _type_matchup(dex, opp, me),
        float(speed_advantage),
        float(state.field.trick_room),
        float(
            bool(
                opp and not opp.item_removed and to_id_str(opp.revealed_item or "") in _CHOICE_ITEMS
            )
        ),
        setup,
        recovery,
        float(last_class == "attack"),
        float(last_class == "non_damaging"),
        float(last_class == "switch"),
        *_threat_features(dex, state, belief),
    )
    return {name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=True)}
