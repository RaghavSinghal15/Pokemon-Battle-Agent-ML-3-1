"""DL3 (#78): the game comparison and the significance test."""

import json

import pytest

from pokeagent.dl import compare


def test_two_proportion_p_value():
    assert compare.two_proportion_p(500, 1000, 500, 1000) == pytest.approx(1.0)
    # 81.3% vs 79.7% over 1000 games each: not significant (p about 0.37).
    assert compare.two_proportion_p(813, 1000, 797, 1000) == pytest.approx(0.366, abs=0.01)


def test_game_results_add_up_chunks(tmp_path):
    games, ablations = tmp_path / "games", tmp_path / "ablations"
    games.mkdir()
    ablations.mkdir()
    for i, wins in enumerate((200, 210)):
        record = {"games": 250, "wins": wins}
        (games / f"mlp-{i}.json").write_text(json.dumps(record), encoding="utf-8")
        (ablations / f"full-{i}.json").write_text(json.dumps(record), encoding="utf-8")
    (ablations / "no_tera-0.json").write_text(json.dumps({"games": 250, "wins": 1}), "utf-8")
    result = compare.game_results(games, ablations)
    assert result["mlp"]["games"] == 500 and result["mlp"]["wins"] == 410
    assert result[compare.REFERENCE]["wins"] == 410
    assert "naive_hidden" not in result
    assert result["mlp_vs_reference_p"] == pytest.approx(1.0)
