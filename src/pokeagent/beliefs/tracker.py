"""Estimate hidden sets from the public Gen 9 Random Battle set definitions."""

import json
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import comb
from pathlib import Path

from poke_env.data import to_id_str

from pokeagent.config import RANDOM_BATTLE_MOVE_SLOTS, RANDOM_BATTLE_SETS_PATH
from pokeagent.interfaces import BeliefState, PokemonView, PublicState, SetBelief


@dataclass(frozen=True)
class _SetTemplate:
    role: str
    moves: frozenset[str]
    abilities: frozenset[str]
    tera_types: frozenset[str]

    @property
    def move_slots(self) -> int:
        return min(RANDOM_BATTLE_MOVE_SLOTS, len(self.moves))


@lru_cache(maxsize=1)
def _load_sets(path: Path) -> dict[str, tuple[_SetTemplate, ...]]:
    try:
        with path.open(encoding="utf-8") as source:
            data = json.load(source)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Random Battle sets not found at {path}. See docs/07-setup.md for the Showdown clone."
        ) from exc

    return {
        species: tuple(
            _SetTemplate(
                role=entry["role"],
                moves=frozenset(map(to_id_str, entry["movepool"])),
                abilities=frozenset(map(to_id_str, entry["abilities"])),
                tera_types=frozenset(map(to_id_str, entry["teraTypes"])),
            )
            for entry in definition["sets"]
        )
        for species, definition in data.items()
    }


def _distribution(values: frozenset[str], observed: str) -> dict[str, float]:
    if observed:
        return {observed: 1.0}
    return {value: 1 / len(values) for value in sorted(values)}


def _marginals(
    species: str,
    templates: list[tuple[_SetTemplate, float]],
    moves: frozenset[str] = frozenset(),
    ability: str = "",
    tera_type: str = "",
) -> SetBelief:
    roles: Counter[str] = Counter()
    move_probs: Counter[str] = Counter()
    abilities: Counter[str] = Counter()
    tera_types: Counter[str] = Counter()
    total_weight = sum(weight for _, weight in templates)
    for template, weight in templates:
        probability = weight / total_weight
        roles[template.role] += probability
        unseen = template.moves - moves
        slots_left = template.move_slots - len(moves)
        if unseen and slots_left:
            for move in sorted(unseen):
                move_probs[move] += probability * slots_left / len(unseen)
        for name, chance in _distribution(template.abilities, ability).items():
            abilities[name] += probability * chance
        for name, chance in _distribution(template.tera_types, tera_type).items():
            tera_types[name] += probability * chance

    move_probs.update(dict.fromkeys(sorted(moves), 1.0))
    return SetBelief(
        species=species,
        role_probs=dict(roles),
        move_probs=dict(move_probs),
        ability_probs=dict(abilities) if not ability else {ability: 1.0},
        tera_probs=dict(tera_types) if not tera_type else {tera_type: 1.0},
    )


def prior_belief(species: str) -> SetBelief:
    """Give each set template equal weight before seeing any battle evidence."""
    species = to_id_str(species)
    templates = _load_sets(RANDOM_BATTLE_SETS_PATH).get(species, ())
    return _marginals(species, [(template, 1.0) for template in templates])


def _likelihood(
    template: _SetTemplate, moves: frozenset[str], ability: str, tera_type: str
) -> float:
    if (
        not moves <= template.moves
        or len(moves) > template.move_slots
        or (ability and ability not in template.abilities)
        or (tera_type and tera_type not in template.tera_types)
    ):
        return 0.0

    # Count the equally likely movesets that contain every revealed move.
    pool_size = len(template.moves)
    slots = template.move_slots
    chance = comb(pool_size - len(moves), slots - len(moves)) / comb(pool_size, slots)
    if ability:
        chance /= len(template.abilities)
    if tera_type:
        chance /= len(template.tera_types)
    return chance


def update_belief(belief: SetBelief, seen: PokemonView) -> SetBelief:
    """Condition a prior on the cumulative public observations for this Pokémon.

    Use the species prior for each turn so earlier observations are counted once.
    Templates with the same role name share that role's prior weight equally.
    """
    species = to_id_str(seen.species)
    if to_id_str(belief.species) != species:
        raise ValueError(f"Belief for {belief.species} cannot describe {seen.species}")

    moves = frozenset(map(to_id_str, seen.revealed_moves))
    ability = to_id_str(seen.revealed_ability or "")
    tera_type = to_id_str(seen.tera_type or "") if seen.terastallized else ""
    item = to_id_str(seen.revealed_item or "")

    templates = _load_sets(RANDOM_BATTLE_SETS_PATH).get(species, ())
    role_counts = Counter(template.role for template in templates)
    weights = []
    for template in templates:
        prior_weight = belief.role_probs.get(template.role, 0.0) / role_counts[template.role]
        weight = prior_weight * _likelihood(template, moves, ability, tera_type)
        if weight > 0:
            weights.append((template, weight))

    result = _marginals(species, weights, moves, ability, tera_type)
    if not weights:
        # A copied move, changed ability or older set can contradict every template.
        # Keep the role prior and the observations without guessing unseen details.
        result.role_probs = belief.role_probs.copy()
    if item and item != "unknownitem" and not seen.item_removed:
        result.item_probs = {item: 1.0}
    return result


def beliefs_for(state: PublicState) -> BeliefState:
    """Rebuild each revealed opponent's belief from this start-of-turn snapshot."""
    return {p.species: update_belief(prior_belief(p.species), p) for p in state.opp.pokemon}
