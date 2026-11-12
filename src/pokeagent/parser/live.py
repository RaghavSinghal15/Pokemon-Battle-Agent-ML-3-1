"""Live poke-env battle -> PublicState (used when our agent is actually playing)."""

from poke_env.battle import AbstractBattle

from pokeagent.interfaces import PublicState


def state_from_battle(battle: AbstractBattle) -> PublicState:
    """Convert poke-env's live Battle object into the same PublicState the replay parser makes.

    TODO (role C):
    - Our own side must contain only PUBLIC information (what has been revealed in the battle
      so far), even though poke-env knows our full team. See the PublicState docstring.
    - Validation: play games on the local server with save_replays=True, parse the saved log
      with parser.replay, and check both functions produce the same state for every turn.
      That test is the proof that training inputs match live inputs.
    """
    raise NotImplementedError
