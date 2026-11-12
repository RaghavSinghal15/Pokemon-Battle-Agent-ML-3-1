"""Turning an Action into a training label (one of interfaces.ACTION_CLASSES)."""

from pokeagent.interfaces import Action


def action_class(action: Action) -> str:
    """Map an action to "attack", "non_damaging" or "switch".

    TODO (roles D and E): a move is "attack" if its category is Physical or Special and
    "non_damaging" if its category is Status. poke-env's Move(move_id, gen=9).category gives
    this without any Pokémon knowledge.
    """
    raise NotImplementedError
