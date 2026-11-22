"""Exact-move prediction (E6, #79): candidates, pair features and the metrics."""

import json

import numpy as np
import pandas as pd
import pytest
from test_interfaces import make_state

from pokeagent.beliefs import tracker
from pokeagent.models.exact_move import (
    FLOOR,
    _normalise,
    candidate_rows,
    evaluate_moves,
)


@pytest.fixture(autouse=True)
def public_sets(tmp_path, monkeypatch):
    """Gholdengo's only set: Make It Rain, Shadow Ball, Nasty Plot, Recover."""
    role = {
        "role": "Bulky Support",
        "movepool": ["Make It Rain", "Shadow Ball", "Nasty Plot", "Recover"],
        "abilities": ["Good as Gold"],
        "teraTypes": ["Steel"],
    }
    path = tmp_path / "sets.json"
    path.write_text(json.dumps({"gholdengo": {"sets": [role]}}), encoding="utf-8")
    monkeypatch.setattr(tracker, "RANDOM_BATTLE_SETS_PATH", path)
    tracker._load_sets.cache_clear()
    yield
    tracker._load_sets.cache_clear()


def test_candidates_describe_each_move_against_the_target():
    state = make_state()  # Gholdengo (me) against Garchomp (Ground/Dragon)
    rows = {row["move"]: row for row in candidate_rows(state, "makeitrain")}
    assert set(rows) == {"makeitrain", "shadowball", "nastyplot", "recover"}
    rain = rows["makeitrain"]
    assert rain["revealed"] == 1 and rain["used_last_turn"] == 1 and rain["stab"] == 1
    assert rain["effectiveness"] == 1  # Steel is neutral on Ground/Dragon
    assert rows["nastyplot"]["is_setup"] == 1 and rows["nastyplot"]["is_attack"] == 0
    assert rows["recover"]["is_recovery"] == 1
    assert all(row["n_candidates"] == 4 for row in rows.values())


def test_metrics_count_a_move_outside_the_candidates_as_a_miss():
    table = pd.DataFrame(
        {
            "decision": ["a", "a", "a", "b", "b", "c", "c"],
            "chosen": [False, True, False, True, False, False, True],
            "missing": [False, False, False, False, False, False, True],
        }
    )
    probs = _normalise(table, np.array([0.2, 0.6, 0.2, 0.9, 0.1, 1.0, 0.0]))
    assert probs[:3].sum() == pytest.approx(1) and probs[6] == FLOOR
    result = evaluate_moves(table, probs)
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["top1"] == pytest.approx(2 / 3)  # a and b are top-ranked; c is a miss
    expected = -(np.log(0.6) + np.log(0.9) + np.log(FLOOR)) / 3
    assert result["log_loss"] == pytest.approx(expected)


def test_effectiveness_from_the_type_chart():
    from pokeagent.interfaces import PokemonView
    from pokeagent.models.exact_move import _effectiveness

    gholdengo = PokemonView("gholdengo", 77)  # Steel/Ghost
    assert _effectiveness("GROUND", gholdengo) == 2
    assert _effectiveness("NORMAL", gholdengo) == 0
    assert _effectiveness("ELECTRIC", PokemonView("garchomp", 77)) == 0
