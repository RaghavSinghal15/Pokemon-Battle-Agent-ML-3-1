"""Headline evaluation report (A7, #83)."""

import json

from pokeagent.eval.headline import markdown, summarise


def test_headline_adds_chunks_and_checks_the_criteria(tmp_path):
    chunks = [
        ("best_response", "random", 990),
        ("best_response", "max_power", 900),
        ("best_response", "heuristics", 700),
        ("lp", "heuristics", 480),
    ]
    for i, (strategy, opponent, wins) in enumerate(chunks):
        record = {
            "strategy": strategy,
            "opponent": opponent,
            "games": 1000,
            "wins": wins,
            "source": {"commit": "abcdef123"},
            "started_utc": "2026-11-23T01:00:00",
        }
        (tmp_path / f"{i}.json").write_text(json.dumps(record), encoding="utf-8")
    rows = summarise(tmp_path)
    assert [(r["strategy"], r["opponent"]) for r in rows] == [
        ("best_response", "random"),
        ("best_response", "max_power"),
        ("best_response", "heuristics"),
        ("lp", "heuristics"),
    ]
    text = markdown(rows)
    assert "beats random and max power: yes" in text
    assert "beats SimpleHeuristicsPlayer: yes" in text
    assert "abcdef1" in text and "2026-11-23" in text
