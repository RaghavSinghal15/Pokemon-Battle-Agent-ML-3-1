"""Opposite orientations, draws, provenance and partial reports cannot be mixed silently."""

from copy import deepcopy

import pytest

from pokeagent.eval.baseline_table import render_baseline_report, summarise_matchups


def record(first, second, wins, losses, ties=0):
    return {
        "player_class": first,
        "opponent_class": second,
        "games": wins + losses + ties,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "seed": 464,
        "source": {"commit": "source-sha", "tracked_changes": False},
        "server_source": {"commit": "server-sha", "tracked_changes": False},
        "format": "gen9randombattle",
    }


def pair():
    return [
        record("RandomPlayer", "MaxBasePowerPlayer", 6, 3, 1),
        record("MaxBasePowerPlayer", "RandomPlayer", 4, 5, 1),
    ]


def test_reverses_outcomes_and_keeps_ties_in_denominator():
    original = pair()
    before = deepcopy(original)
    summary = summarise_matchups(original)[0]
    assert (
        summary["player"],
        summary["wins"],
        summary["losses"],
        summary["ties"],
        summary["games"],
    ) == ("max_power", 7, 11, 2, 20)
    assert summary["win_rate"] == pytest.approx(0.35)
    assert summary["opponent_win_rate"] == pytest.approx(0.55)
    assert original == before


def test_partial_unbalanced_and_duplicate_orientations_are_rejected():
    records = pair()
    for bad in ([], records[:1], [records[0], records[0]], records + records):
        with pytest.raises(ValueError):
            summarise_matchups(bad)
    records[1]["games"] = 11
    records[1]["wins"] += 1
    with pytest.raises(ValueError, match="equal game counts"):
        summarise_matchups(records)


@pytest.mark.parametrize("change", ["source", "server", "counts", "format", "dirty"])
def test_incompatible_records(change):
    records = pair()
    if change == "source":
        records[1]["source"]["commit"] = "different"
    elif change == "server":
        records[1]["server_source"] = None
    elif change == "counts":
        records[1]["wins"] = 100
    elif change == "format":
        records[1]["format"] = "other"
    else:
        records[0]["source"]["tracked_changes"] = True
    with pytest.raises(ValueError):
        summarise_matchups(records)


def test_report_needs_all_pairs():
    with pytest.raises(ValueError, match="all three pairs"):
        render_baseline_report(pair(), ["one.json", "two.json"])


def test_complete_report_has_all_three_pairs_and_six_record_links():
    records = pair()
    for opponent in ("RandomPlayer", "MaxBasePowerPlayer"):
        records.extend(
            [
                record("SimpleHeuristicsPlayer", opponent, 8, 2),
                record(opponent, "SimpleHeuristicsPlayer", 2, 8),
            ]
        )
    links = [f"run-{i}.json" for i in range(6)]
    text = render_baseline_report(records, links)
    assert "source-sha" in text and "server-sha" in text
    assert all(f"]({link})" in text for link in links)
    assert "35.0%" in text and "55.0%" in text
    assert "heuristics" in text and "max_power" in text and "random" in text
