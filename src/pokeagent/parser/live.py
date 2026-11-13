"""Live poke-env battle -> PublicState (used when our agent is actually playing)."""

from poke_env.battle import AbstractBattle

from pokeagent.interfaces import PublicState


def state_from_battle(battle: AbstractBattle) -> PublicState:
    """Convert poke-env's live Battle object into the same PublicState the replay parser makes.

    TODO (role C, task C3 in TASKS.md):
    - Don't read the live battle object directly: it contains private information about our
      own team (exact HP, unrevealed moves) and even leaks it into the opponent's side (Ditto).
      Instead, re-parse the messages our bot received (`battle._replay_data`) with the same
      code as parser.replay, so training and live play share one code path. Checks 2 and 3 in
      docs/08-foundation-checks.md showed this gives exactly the spectator view.
    - Validation (task C4): on local games, this function on the live stream and parser.replay
      on the spectator log must give the same state for every turn. That test is the proof
      that training inputs match live inputs.
    """
    raise NotImplementedError
