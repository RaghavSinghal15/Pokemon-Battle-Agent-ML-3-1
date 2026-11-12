"""Replay log -> list of DecisionPoints."""

from pokeagent.interfaces import DecisionPoint


def parse_battle(
    battle_id: str, log: str, winner: str, ratings: dict[str, int | None]
) -> list[DecisionPoint]:
    """Walk through a battle log line by line and emit two DecisionPoints per turn.

    A log is plain text, one event per line, fields separated by "|", for example:

        |turn|3
        |move|p1a: Gholdengo|Make It Rain|p2a: Garchomp
        |-damage|p2a: Garchomp|61/100

    TODO (role C):
    1. First spike (do this before writing your own parser): try feeding the log lines into a
       poke-env Battle object (`Battle.parse_message`). If poke-env can track a spectator's view
       correctly, reuse it and just convert its state into PublicState. That could save a lot
       of work. Write down what you found either way in docs/decisions.md.
    2. At each "|turn|N" line, snapshot the current state (before any move of turn N).
    3. Record what each player chose during turn N (their "|move|" or "|switch|" line) as
       that turn's Action. Skip forced switches after a faint (see DecisionPoint docstring).
    4. Leakage test: the state for turn N must be identical whether or not you cut the log
       right after the "|turn|N" line. tests/test_parser.py should check this on real logs.
    """
    raise NotImplementedError
