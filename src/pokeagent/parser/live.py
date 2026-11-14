"""Convert live poke-env messages to the same public states used for training."""

from collections.abc import Iterable, Iterator

from poke_env.battle import AbstractBattle

from pokeagent.interfaces import Player, PublicState
from pokeagent.parser.replay import iter_public_states


def iter_live_states(
    battle_id: str, lines: Iterable[str], perspective: Player
) -> Iterator[PublicState]:
    """Yield one side's public states from its received message stream."""
    for state in iter_public_states(battle_id, lines):
        if state.perspective == perspective:
            yield state


def state_from_battle(battle: AbstractBattle) -> PublicState:
    """Return the current turn's initial snapshot from public protocol events."""
    perspective = battle.player_role
    if perspective not in ("p1", "p2") or battle.turn < 1:
        raise ValueError("The battle must have started a decision turn")
    lines = ("|".join(message) for message in battle._replay_data)
    for state in iter_live_states(battle.battle_tag, lines, perspective):
        if state.turn == battle.turn:
            return state
    raise ValueError(f"Public messages for turn {battle.turn} are missing")
