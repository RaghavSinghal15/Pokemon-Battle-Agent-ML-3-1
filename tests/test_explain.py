"""Explanation arithmetic and the distinction between best response and minimax."""

import numpy as np
from test_interfaces import make_state

from pokeagent.agent.explain import format_turn
from pokeagent.decision.strategies import best_response, minimax_lp
from pokeagent.interfaces import Action, PayoffTable, SetBelief


def example():
    table = PayoffTable(
        [Action("move", "makeitrain"), Action("switch", "corviknight")],
        [Action("move", "earthquake"), Action("move", "swordsdance")],
        [[0.2, 0.4], [0.3, -0.1]],
        [0.25, 0.75],
    )
    beliefs = {"garchomp": SetBelief("garchomp", role_probs={"Setup Sweeper": 0.8, "Fast": 0.2})}
    return table, beliefs, {"attack": 0.2, "non_damaging": 0.3, "switch": 0.5}


def test_best_response_uses_feasible_column_weights_not_raw_class_mass():
    table, beliefs, forecast = example()
    text = format_turn(make_state(), beliefs, forecast, table, best_response(table))
    assert "gholdengo (78%) vs garchomp (100%)" in text
    assert "Setup Sweeper 80%, Fast 20%" in text
    assert "move:makeitrain +0.350" in text
    assert "switch:corviknight -0.000" in text or "switch:corviknight +0.000" in text
    assert "not win probability" in text


def test_minimax_explains_mixture_and_worst_case_instead_of_forecast_expectation():
    table, beliefs, forecast = example()
    decision = minimax_lp(table, rng=np.random.default_rng(12))
    text = format_turn(make_state(), beliefs, forecast, table, decision)
    assert "Minimax ignores the opponent forecast" in text
    assert "Mixture:" in text
    assert "Row worst cases" in text
    assert "worst-case expected payoff" in text
    assert "Expected row payoffs" not in text
