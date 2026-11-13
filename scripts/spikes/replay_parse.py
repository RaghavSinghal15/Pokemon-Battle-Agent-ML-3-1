"""Rebuild a battle from a public log with poke-env's own battle tracker (foundation check 2).

The idea: poke-env's `Battle.parse_message` already understands every Showdown log event,
because that is how a live bot keeps track of a game. So instead of writing a parser from
scratch, we create an empty `Battle`, pretend to be one of the two players (the
"perspective"), and feed it the replay line by line. Every time a `|turn|N` line arrives, the
battle is at "the start of turn N, before anyone chooses", which is exactly a decision point.
"""

import logging

from poke_env.battle import Battle

# Lines a player's client handles itself instead of passing to Battle.parse_message
# (mirrors poke_env.player.Player._handle_battle_message).
SKIP = {"request", "showteam", "bigerror", "error", "t:", "j", "J", "l", "L", "c", "raw"}

_logger = logging.getLogger("replay_parse")
_logger.setLevel(logging.ERROR)


def parse_log(
    battle_id: str,
    log_lines: list[str],
    perspective_username: str,
    on_turn=None,
    errors=None,
):
    """Feed a public log into a fresh Battle seen from `perspective_username`'s side.

    `on_turn(battle)` is called right after each `|turn|N` line. Returns the final battle.
    If `errors` (a Counter) is given, lines poke-env can't handle are counted there by message
    type and skipped, instead of raising. Real human replays contain lines bots never produce
    (chat, timer notices, messages from older server versions), so this tells us what's missing.
    """
    battle = Battle(battle_id, perspective_username, _logger, gen=9, save_replays=False)
    for line in log_lines:
        parts = line.split("|")
        if len(parts) < 2 or parts[1] in SKIP:
            continue
        if parts[1] == "":
            continue
        if parts[1] == "win":
            battle.won_by(parts[2])
            continue
        if parts[1] == "tie":
            battle.tied()
            continue
        if errors is None:
            battle.parse_message(parts)
        else:
            try:
                battle.parse_message(parts)
            except Exception as e:  # noqa: BLE001 (count and keep going)
                errors[f"{type(e).__name__} on |{parts[1]}|"] += 1
        if parts[1] == "turn" and on_turn is not None:
            on_turn(battle)
    return battle
