"""Play on the public Showdown ladder against people and record ratings (A8, #82).

The ladder rates each account with Elo. After a rated game the server posts a line such as
"Rag Bot's rating: 1043 &rarr; <strong>1061</strong>". poke-env keeps only the first number
(the rating BEFORE the game), so we read the line ourselves to get the rating AFTER it.
"""

import re
from collections.abc import Iterable

from poke_env.data.normalize import to_id_str

from pokeagent.agent.player import ExplainableAgent
from pokeagent.eval.stats import win_rate_ci

RATING_LINE = re.compile(r"^(?P<name>.+)'s rating: (?P<before>\d+) &rarr; <strong>(?P<after>\d+)")


def parse_rating_change(raw: str) -> tuple[str, int, int] | None:
    """Return (user id, rating before, rating after) from a ladder |raw| line, else None."""
    match = RATING_LINE.match(raw)
    if match is None:
        return None
    return to_id_str(match["name"]), int(match["before"]), int(match["after"])


class LadderAgent(ExplainableAgent):
    """The normal agent, plus a record of the rating changes the server announces."""

    def __init__(self, *args, **kwargs):
        # battle tag -> user id -> (before, after)
        self.rating_changes: dict[str, dict[str, tuple[int, int]]] = {}
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages):
        tag = split_messages[0][0].lstrip(">").strip()
        for message in split_messages[1:]:
            if len(message) > 2 and message[1] == "raw":
                change = parse_rating_change(message[2])
                if change is not None:
                    user, before, after = change
                    self.rating_changes.setdefault(tag, {})[user] = (before, after)
        await super()._handle_battle_message(split_messages)


def game_record(battle, changes: dict[str, tuple[int, int]], our_id: str) -> dict:
    """One finished ladder game. Holds the opponent's name, so it is kept out of reports/."""
    opponent_id = to_id_str(battle.opponent_username or "")
    ours = changes.get(our_id, (None, None))
    theirs = changes.get(opponent_id, (None, None))
    return {
        "battle": battle.battle_tag,
        "opponent": opponent_id,
        "won": bool(battle.won),
        "turns": battle.turn,
        "rating_before": ours[0],
        "rating_after": ours[1],
        "opponent_rating_before": theirs[0],
    }


def summarise(records: Iterable[dict]) -> dict:
    """Win rate with a Wilson interval, the rating path and opponent ratings, no usernames."""
    records = list(records)
    if not records:
        raise ValueError("No finished ladder games to summarise")
    wins = sum(record["won"] for record in records)
    rate, low, high = win_rate_ci(wins, len(records))
    path = [record["rating_after"] for record in records if record["rating_after"] is not None]
    opponents = [
        record["opponent_rating_before"]
        for record in records
        if record["opponent_rating_before"] is not None
    ]
    return {
        "games": len(records),
        "wins": wins,
        "win_rate": rate,
        "win_rate_95": [low, high],
        "rating_start": records[0]["rating_before"],
        "rating_end": path[-1] if path else None,
        "rating_peak": max(path) if path else None,
        "rating_path": path,
        "opponent_rating_mean": sum(opponents) / len(opponents) if opponents else None,
        "mean_turns": sum(record["turns"] for record in records) / len(records),
    }
