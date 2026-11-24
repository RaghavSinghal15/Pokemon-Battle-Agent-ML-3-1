"""Item beliefs for unrevealed opponent items (#119).

Gen 9 Random Battle sets (sets.json) list moves, abilities and tera types, but not items: the
team generator (teams.ts) picks the item from the moveset and the Pokemon's role. So instead of
re-implementing those rules, `python -m pokeagent.beliefs.sample_items` samples many teams from
the local simulator and saves how often each (species, moves, ability, item) combination comes
up. The item belief is then Bayes' rule by counting: among sampled sets of this species that
agree with everything revealed so far (every revealed move, the ability if shown), how often
does each item appear?

This only feeds the payoff table (speed order and damage). The opponent model's features do
not read item beliefs, so the trained models are unaffected.
"""

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from poke_env.data import to_id_str

from pokeagent.config import RANDOM_BATTLE_ITEMS_PATH


@lru_cache(maxsize=2)
def _load(path: Path) -> dict[str, list[tuple[frozenset[str], str, str, int]]]:
    """species -> [(moves, ability, item, count)]; empty if the sample was never made."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    table: dict[str, list] = {}
    for key, count in raw.items():
        species, moves, ability, item = key.split("|")
        table.setdefault(species, []).append(
            (frozenset(moves.split(",")) if moves else frozenset(), ability, item, count)
        )
    return table


def item_probs(
    species: str,
    revealed_moves=(),
    revealed_ability: str = "",
    path: Path = RANDOM_BATTLE_ITEMS_PATH,
) -> dict[str, float]:
    """P(item) for this species given what has been revealed. {} if there is no sample.

    If nothing sampled agrees with the evidence (a copied move or ability, say), the ability is
    dropped first and then the moves, as the set tracker does.
    """
    sets = _load(path).get(to_id_str(species), [])
    moves = frozenset(map(to_id_str, revealed_moves))
    ability = to_id_str(revealed_ability or "")
    for need_moves, need_ability in ((moves, ability), (moves, ""), (frozenset(), "")):
        counts = Counter()
        for set_moves, set_ability, item, count in sets:
            if need_moves <= set_moves and (not need_ability or set_ability == need_ability):
                counts[item] += count
        total = sum(counts.values())
        if total:
            return {item: n / total for item, n in counts.most_common()}
    return {}
