"""F13 (#81): the per-turn trace and the forecast's feature contributions."""

import json

import numpy as np
import pandas as pd
from test_interfaces import make_state

from pokeagent.agent.explain import turn_record
from pokeagent.dashboard.contributions import forecast_contributions, top_contributions
from pokeagent.features.build import FEATURE_NAMES
from pokeagent.interfaces import Action, Decision, PayoffTable, SetBelief
from pokeagent.models.opponent import OpponentModel, train_lightgbm


def test_turn_record_is_plain_json_with_the_table_and_choice():
    state = make_state()
    opp = state.opp.active.species
    beliefs = {opp: SetBelief(opp, role_probs={"A": 0.7, "B": 0.3}, item_probs={"lifeorb": 1})}
    table = PayoffTable(
        [Action("move", "tackle"), Action("switch", "x")],
        [Action("move", "ember")],
        [[0.1], [-0.2]],
        [1.0],
    )
    decision = Decision(Action("move", "tackle"), "best_response", {"move:tackle": 1.0}, "why")
    forecast = {"attack": 0.5, "non_damaging": 0.2, "switch": 0.3}
    record = turn_record(state, beliefs, forecast, table, decision, 0.6, {"turn": 1.0})
    assert json.loads(json.dumps(record)) == record
    assert record["chosen"] == "move:tackle" and record["table"]["values"] == [[0.1], [-0.2]]
    assert record["beliefs"]["roles"] == {"A": 0.7, "B": 0.3}


def test_contributions_add_up_to_the_raw_score():
    rng = np.random.default_rng(0)
    table = pd.DataFrame(rng.normal(size=(600, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    table["action_class"] = np.where(table["opp_type_matchup"] < -0.3, "switch", "attack")
    table.loc[table.index % 5 == 0, "action_class"] = "non_damaging"
    model, _ = train_lightgbm(table, table, leaves_grid=(7,))
    row = table.iloc[0][FEATURE_NAMES].to_dict()
    contributions = forecast_contributions(model, row)
    raw = model.estimator.booster.predict(pd.DataFrame([row]), raw_score=True)[0]
    for index, name in enumerate(model.classes):
        assert (
            sum(contributions[name].values()) == np.float64(raw[index]).round(6)
            or abs(sum(contributions[name].values()) - raw[index]) < 1e-6
        )
    assert len(top_contributions(contributions["switch"], 3)) == 3
    assert (
        forecast_contributions(OpponentModel("flat", class_frequencies=np.ones(3) / 3), row) == {}
    )
