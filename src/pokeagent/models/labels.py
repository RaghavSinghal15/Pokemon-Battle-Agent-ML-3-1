"""Turning an Action into a training label (one of interfaces.ACTION_CLASSES).

This is the only place the rule is written, because three parts of the project must agree:

- features/export.py: the label column of features.parquet, i.e. what the opponent model learns.
- features/build.py: the feature "class of the opponent's last action".
- decision/payoff.py: which of the opponent's possible moves count as "attack" and which as
  "non_damaging" when it spreads the model's class probabilities over those moves.

If they used different rules, the model would predict one grouping and the payoff table would
read its output as another.

The rule labels the CHOICE the player made, using the move's official category from the Gen 9
move data. It says nothing about how good the choice is; valuing moves is the payoff table's job.
"""

from functools import cache

from poke_env.data import GenData, to_id_str

from pokeagent.config import RANDOM_BATTLE_GENERATION
from pokeagent.interfaces import Action


@cache
def _move_data() -> dict:
    """Showdown's Gen 9 move table (via poke-env), loaded once."""
    return GenData.from_gen(RANDOM_BATTLE_GENERATION).moves


def move_class(move_name: str) -> str:
    """ "attack" for a Physical or Special move, "non_damaging" for a Status move.

    Accepts a display name ("U-turn") or an id ("uturn"). Edge cases follow the category, on
    purpose (counts and reasons in docs/decisions.md, 2026-11-22):
    U-turn, Volt Switch, Seismic Toss and Struggle are attacks; Swords Dance, Parting Shot,
    Leech Seed, Pain Split and Sleep Talk are non-damaging.
    """
    category = _move_data().get(to_id_str(move_name), {}).get("category")
    if category in ("Physical", "Special"):
        return "attack"
    if category == "Status":
        return "non_damaging"
    raise ValueError(f"Unknown move {move_name!r}: it has no category in the Gen 9 move data")


def action_class(action: Action) -> str:
    """Map an action to "attack", "non_damaging" or "switch" (interfaces.ACTION_CLASSES).

    Terastallizing does not change the label: the label is about the move or the switch.
    """
    if action.kind == "switch":
        return "switch"
    if action.kind == "move":
        return move_class(action.name)
    raise ValueError(f"Unknown action kind {action.kind!r}")
