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
    rows = summarise(tmp_path, ablation_dir=None)
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


def test_heuristics_pairing_reuses_the_full_and_lp_ablation_chunks(tmp_path):
    headline, ablations = tmp_path / "headline", tmp_path / "ablations"
    headline.mkdir()
    ablations.mkdir()
    base = {"games": 250, "source": {"commit": "abcdef123"}, "started_utc": "2026-11-23T01:00"}
    record = {**base, "strategy": "best_response", "opponent": "random", "wins": 249}
    (headline / "br-random.json").write_text(json.dumps(record), encoding="utf-8")
    for i, (variant, wins) in enumerate(
        [("full", 200), ("full", 210), ("lp", 160), ("no_tera", 1)]
    ):
        record = {**base, "variant": variant, "wins": wins}
        (ablations / f"{variant}-{i}.json").write_text(json.dumps(record), encoding="utf-8")
    rows = {
        (r["strategy"], r["opponent"]): (r["games"], r["wins"])
        for r in summarise(headline, ablations)
    }
    assert rows == {
        ("best_response", "random"): (250, 249),
        ("best_response", "heuristics"): (500, 410),
        ("lp", "heuristics"): (250, 160),
    }
