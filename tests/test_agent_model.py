"""F6 wiring checks with an explicit test predictor, never a claimed trained model."""

from copy import deepcopy
from io import StringIO

import pytest
from test_agent import public_sets as public_sets
from test_damage import make_battle
from test_payoff import add_reserve

from pokeagent.agent import player
from pokeagent.agent.player import ExplainableAgent
from pokeagent.beliefs.tracker import beliefs_for
from pokeagent.config import BATTLE_FORMAT
from pokeagent.features.build import FEATURE_NAMES
from pokeagent.interfaces import Action, PayoffTable, PokemonView, PublicState, SideView


class TestPredictor:
    """Fixed probabilities for plumbing tests; no training or performance claim."""

    __test__ = False

    def __init__(self, probabilities=None):
        self.rows = []
        self.probabilities = probabilities or {"attack": 0.6, "non_damaging": 0.1, "switch": 0.3}

    def predict_proba(self, row):
        self.rows.append(deepcopy(row))
        return dict(self.probabilities)


def public_turn():
    battle = make_battle("Blissey", "Blissey")
    battle._turn = 1
    battle._replay_data = [
        line.split("|")
        for line in [
            "|player|p1|Alice|",
            "|player|p2|Bob|",
            "|teamsize|p1|6",
            "|teamsize|p2|6",
            "|switch|p1a: Attacker|Blissey, L100|100/100",
            "|switch|p2a: Defender|Blissey, L100|100/100",
            "|turn|1",
        ]
    ]
    return battle


@pytest.mark.parametrize("strategy", ["best_response", "lp"])
def test_public_pipeline_passes_model_forecast_and_logs_real_strategy(strategy, monkeypatch):
    model, log = TestPredictor(), StringIO()
    battle = public_turn()
    forecasts = []

    def table_builder(battle, beliefs, forecast, unseen_samples=0):
        forecasts.append(forecast)
        # Stand-ins for unseen switch-ins are used by minimax only (F13).
        assert bool(unseen_samples) == (strategy == "lp")
        assert beliefs["blissey"].move_probs
        return PayoffTable(
            [Action("move", "seismictoss")], [Action("move", "splash")], [[0.2]], [1]
        )

    monkeypatch.setattr(player, "build_payoff_table", table_builder)
    agent = ExplainableAgent(
        battle_format=BATTLE_FORMAT,
        mode="model",
        opponent_model=model,
        strategy=strategy,
        explanation_stream=log,
        start_listening=False,
    )
    assert agent.choose_move(battle).order.id == "seismictoss"
    assert list(model.rows[0]) == FEATURE_NAMES
    assert model.rows[0]["my_revealed_moves"] == 0
    assert forecasts[0] == model.probabilities
    assert agent.last_decision.strategy == strategy
    assert "Opponent forecast: attack 60%" in log.getvalue()
    assert "Battle battle-test | Turn 1" in log.getvalue()
    if strategy == "lp":
        assert "Minimax ignores the opponent forecast" in log.getvalue()
    else:
        assert "Expected row payoffs" in log.getvalue()
    # Private live changes and future public results must not reach this turn's features.
    battle.active_pokemon.stats["spe"] = 99999
    battle._replay_data += [
        line.split("|")
        for line in [
            "|move|p1a: Attacker|Seismic Toss|p2a: Defender",
            "|-damage|p2a: Defender|50/100",
            "|win|Alice",
        ]
    ]
    agent.choose_move(battle)
    assert model.rows[0] == model.rows[1]


def test_invalid_model_distribution_has_a_labelled_fallback():
    model, log = TestPredictor({"attack": 1}), StringIO()
    agent = ExplainableAgent(
        battle_format=BATTLE_FORMAT,
        mode="model",
        opponent_model=model,
        explanation_stream=log,
        start_listening=False,
    )
    assert agent.choose_move(public_turn()).order.id == "seismictoss"
    assert agent.last_decision.strategy == "baseline"
    assert "Forecast must contain exactly" in log.getvalue()


def test_forced_replacement_does_not_query_model_and_still_logs():
    model, log = TestPredictor(), StringIO()
    battle = public_turn()
    reserve = add_reserve(battle, "p1")
    battle._force_switch, battle._available_switches = True, [reserve]
    agent = ExplainableAgent(
        battle_format=BATTLE_FORMAT,
        mode="model",
        opponent_model=model,
        explanation_stream=log,
        start_listening=False,
    )
    assert agent.choose_move(battle).order is reserve
    assert not model.rows
    assert "forced replacement" in log.getvalue()
    assert "switch:snorlax" in log.getvalue()


def test_cosmetic_form_conditions_base_prior_on_public_revealed_move():
    seen = PokemonView("pikachuhoenn", 90, active=True, revealed_moves=["surf"])
    state = PublicState("test", 1, "p1", SideView("p1"), SideView("p2", [seen]))
    before = deepcopy(state)
    belief = beliefs_for(state)["pikachuhoenn"]
    assert belief.species == "pikachuhoenn"
    assert belief.move_probs["surf"] == 1
    assert state == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "model"},
        {"opponent_model": TestPredictor()},
        {"mode": "unknown"},
    ],
)
def test_missing_or_ignored_model_is_a_configuration_error(kwargs):
    with pytest.raises(ValueError, match="requires an opponent_model"):
        ExplainableAgent(start_listening=False, **kwargs)
