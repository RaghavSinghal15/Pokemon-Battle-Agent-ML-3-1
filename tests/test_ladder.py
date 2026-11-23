"""Ladder bookkeeping (A8, #82): rating lines, per-game records and the summary."""

import asyncio
from types import SimpleNamespace

import pytest

from pokeagent.eval.ladder import LadderAgent, game_record, parse_rating_change, summarise


def test_parse_rating_change_reads_before_and_after():
    line = "Rag Bot-1's rating: 1043 &rarr; <strong>1061</strong><br />(+18 for winning)"
    assert parse_rating_change(line) == ("ragbot1", 1043, 1061)
    # Below 1000: poke-env's own parser reads only the first four characters; ours doesn't.
    assert parse_rating_change("x's rating: 987 &rarr; <strong>970</strong>") == ("x", 987, 970)
    assert parse_rating_change("Ladder updating...") is None


def battle(won, opponent="Some One", turns=20):
    return SimpleNamespace(
        battle_tag="b", opponent_username=opponent, won=won, turns=turns, turn=turns
    )


def test_game_record_matches_ratings_by_user_id():
    changes = {"ragbot": (1000, 1020), "someone": (1100, 1080)}
    record = game_record(battle(True), changes, "ragbot")
    assert record["won"] and record["rating_before"] == 1000 and record["rating_after"] == 1020
    assert record["opponent_rating_before"] == 1100
    unrated = game_record(battle(None), {}, "ragbot")  # unrated or still updating
    assert not unrated["won"] and unrated["rating_after"] is None


def test_summary_has_wilson_interval_rating_path_and_no_usernames():
    records = [
        game_record(battle(True), {"ragbot": (1000, 1020), "someone": (1100, 1)}, "ragbot"),
        game_record(battle(False), {"ragbot": (1020, 1005)}, "ragbot"),
        game_record(battle(True), {}, "ragbot"),
    ]
    summary = summarise(records)
    assert summary["wins"] == 2 and summary["games"] == 3
    assert summary["win_rate_95"][0] < 2 / 3 < summary["win_rate_95"][1]
    assert summary["rating_path"] == [1020, 1005] and summary["rating_end"] == 1005
    assert summary["rating_start"] == 1000 and summary["opponent_rating_mean"] == 1100
    assert "someone" not in str(summary)
    with pytest.raises(ValueError):
        summarise([])


def test_summary_leaves_out_games_against_excluded_accounts():
    records = [
        game_record(battle(True, opponent="Some One"), {"ragbot": (1000, 1020)}, "ragbot"),
        game_record(battle(True, opponent="Our Bot 2"), {"ragbot": (1020, 1030)}, "ragbot"),
        game_record(battle(False, opponent="Other"), {"ragbot": (1030, 1015)}, "ragbot"),
    ]
    summary = summarise(records, exclude=["OurBot2"])
    assert summary["games"] == 2 and summary["wins"] == 1 and summary["excluded_games"] == 1
    assert summary["rating_path"] == [1020, 1015]


class FakeClient:
    def __init__(self):
        self.sent = []

    async def send_message(self, message, room=""):
        self.sent.append((room, message))


def test_ladder_agent_greets_once_per_battle(monkeypatch):
    agent = LadderAgent.__new__(LadderAgent)  # skip the network setup in __init__
    agent.rating_changes, agent.greeting, agent._greeted = {}, "hi, I'm a bot", set()
    client = FakeClient()
    monkeypatch.setattr(LadderAgent, "ps_client", client, raising=False)

    async def parent(self, split_messages):
        return None

    monkeypatch.setattr(LadderAgent.__mro__[1], "_handle_battle_message", parent)
    for tag in (">battle-gen9randombattle-1", ">battle-gen9randombattle-1", ">battle-x-2"):
        asyncio.run(agent._handle_battle_message([[tag], ["", "turn", "1"]]))
    assert client.sent == [
        ("battle-gen9randombattle-1", "hi, I'm a bot"),
        ("battle-x-2", "hi, I'm a bot"),
    ]
