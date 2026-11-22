"""Ladder bookkeeping (A8, #82): rating lines, per-game records and the summary."""

from types import SimpleNamespace

import pytest

from pokeagent.eval.ladder import game_record, parse_rating_change, summarise


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
