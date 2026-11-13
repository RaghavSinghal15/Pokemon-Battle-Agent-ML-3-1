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

    TODO (role C, tasks C1 and C2 in TASKS.md):
    1. Feed the log lines into a poke-env Battle object (`Battle.parse_message`). This is
       already shown to work on public logs: see scripts/spikes/replay_parse.py and checks 2
       and 3 in docs/08-foundation-checks.md. Convert its state into PublicState.
    2. At each "|turn|N" line, snapshot the current state (before any move of turn N).
    3. Record what each player chose during turn N as that turn's Action, or skip the turn if
       the choice is hidden (fainted first, couldn't move, locked into a move). The rules are
       in scripts/spikes/label_visibility.py and check 4 of docs/08-foundation-checks.md.
    4. Leakage test: the state for turn N must be identical whether or not you cut the log
       right after the "|turn|N" line. tests/test_parser.py should check this.
    """
    raise NotImplementedError
