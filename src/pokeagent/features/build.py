"""Build model inputs from the public state before either player acts."""

from functools import cache
from math import prod

from poke_env.data import GenData, to_id_str

from pokeagent.config import RANDOM_BATTLE_GENERATION
from pokeagent.interfaces import Action, BeliefState, FeatureRow, PokemonView, PublicState, SideView

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
]

_HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
_STATS = {"atk", "def", "spa", "spd", "spe"}
_CHOICE_ITEMS = {"choiceband", "choicescarf", "choicespecs"}
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


def _action_class(dex: GenData, action: Action | None) -> str | None:
    if action is None:
        return None
    if action.kind == "switch":
        return "switch"
    if action.kind != "move":
        return None
    category = dex.moves.get(to_id_str(action.name), {}).get("category")
    if category in {"Physical", "Special"}:
        return "attack"
    if category == "Status":
        return "non_damaging"
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
    )
    return {name: float(value) for name, value in zip(FEATURE_NAMES, values, strict=True)}
