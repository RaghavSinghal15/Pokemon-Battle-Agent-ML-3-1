"""Value-model utility (V4, #74): cells score the change in predicted P(win)."""

import numpy as np
import pandas as pd
import pytest
from test_interfaces import make_state

from pokeagent.decision._turn import TurnEffect, turn_value
from pokeagent.decision.value_utility import ValueScorer
from pokeagent.eval.puzzles import PUZZLES, solve
from pokeagent.features.value import LABEL, VALUE_FEATURE_NAMES
from pokeagent.models.value import train_value_logistic

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def value_model():
    """A logistic value model on synthetic rows: more HP and more Pokemon left -> more wins."""
    rng = np.random.default_rng(0)
    # Other features are small noise, so the model leans on HP and Pokemon remaining.
    noise = rng.normal(scale=0.01, size=(3000, len(VALUE_FEATURE_NAMES)))
    table = pd.DataFrame(noise, columns=VALUE_FEATURE_NAMES)
    for side in ("my", "opp"):
        table[f"{side}_team_hp"] = rng.uniform(0, 6, 3000)
        table[f"{side}_hp"] = rng.uniform(0, 1, 3000)
        table[f"{side}_remaining"] = rng.integers(1, 7, 3000)
    lead = sum(
        sign * (table[f"{side}_team_hp"] + table[f"{side}_hp"] + table[f"{side}_remaining"])
        for side, sign in (("my", 1), ("opp", -1))
    )
    table[LABEL] = rng.random(3000) < 1 / (1 + np.exp(-0.2 * lead))
    model, _ = train_value_logistic(table, table, c_grid=(1.0,))
    return model


def test_unstandardised_weights_reproduce_the_model(value_model):
    scorer = ValueScorer(value_model, make_state())
    row = scorer._row(None, None)
    expected = value_model.win_probability(row)
    assert scorer.win_probability() == pytest.approx(expected)


def test_end_value_moves_with_hp_and_knockouts(value_model):
    scorer = ValueScorer(value_model, make_state())
    value = scorer.end_value(None, None)
    now = scorer._row(None, None)
    start = value(now["my_hp"], now["opp_hp"])
    assert start == pytest.approx(scorer.win_probability())
    assert value(now["my_hp"], now["opp_hp"] / 2) > start > value(now["my_hp"] / 2, now["opp_hp"])
    # Knocking out their active Pokemon is worth more than leaving it on a sliver of HP.
    assert value(now["my_hp"], 0) > value(now["my_hp"], 0.01)
    # Losing our last Pokemon decides the game.
    scorer.state.me.team_size = 1
    scorer._rows.clear()
    assert scorer.end_value(None, None)(0, 1) == pytest.approx(0, abs=1e-9)


def test_turn_value_reports_change_from_a_shared_start():
    def end(ours, theirs):
        return 0.5 + (ours - theirs) / 4

    hit = TurnEffect((0.4, 0.4))
    assert turn_value(1, 1, hit, TurnEffect(), 1, end_value=end) == pytest.approx(0.1)
    # A switch row facing a different start position still measures against P(win) now.
    assert turn_value(1, 1, TurnEffect(), TurnEffect(), 1, end_value=end, start_value=0.4) == (
        pytest.approx(0.1)
    )


def test_puzzles_run_in_value_mode(value_model):
    puzzle = next(p for p in PUZZLES if p.name == "take_the_ko")
    result = solve(puzzle, value_model=value_model)
    assert result["best_response"] == "earthquake"
    assert all(-1 <= v <= 1 for v in result["expected"].values())


def test_agent_needs_a_value_model_for_value_utility(tmp_path, monkeypatch):
    from pokeagent.agent import player
    from pokeagent.agent.player import ExplainableAgent

    path = tmp_path / "sets.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(player, "RANDOM_BATTLE_SETS_PATH", path)
    with pytest.raises(ValueError, match="value_model"):
        ExplainableAgent(start_listening=False, utility="value")
