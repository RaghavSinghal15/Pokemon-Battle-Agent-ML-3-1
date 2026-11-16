"""Hand-calculated payoffs check best response independently of the unfinished agent."""

from copy import deepcopy

import pytest

from pokeagent.decision.strategies import best_response
from pokeagent.interfaces import Action, PayoffTable


def make_table(values: list[list[float]], opp_probs: list[float] | None) -> PayoffTable:
    """Give each row and column a distinct action so the chosen row is easy to check."""
    return PayoffTable(
        our_actions=[Action("move", f"row{i}") for i in range(len(values))],
        opp_actions=[Action("move", f"col{j}") for j in range(len(values[0]))],
        values=values,
        opp_probs=opp_probs,
    )


@pytest.mark.parametrize("probabilities, chosen_index", [([0.9, 0.1], 0), ([0.1, 0.9], 1)])
def test_forecast_changes_the_best_response(probabilities, chosen_index):
    # Row 0 has expected payoff 2.6 or -8.6; row 1 always returns 1.
    table = make_table([[4, -10], [1, 1]], probabilities)
    decision = best_response(table)
    assert decision.chosen is table.our_actions[chosen_index]
    assert decision.strategy == "best_response"
    assert list(decision.mixed_strategy.values()) == [float(i == chosen_index) for i in range(2)]


def test_rectangular_game_and_explanation():
    # Row averages are -2*0.2 + 4*0.5 + 1*0.3 = 1.9 and 3*0.2 + 2*0.3 = 1.2.
    table = make_table([[-2, 4, 1], [3, 0, 2]], [0.2, 0.5, 0.3])
    decision = best_response(table)
    assert decision.chosen is table.our_actions[0]
    assert "+1.900" in decision.explanation
    assert "move:row0" in decision.explanation


def test_all_negative_payoffs_still_choose_the_least_bad_row():
    table = make_table([[-5, -4], [-2, -3]], [0.5, 0.5])
    assert best_response(table).chosen is table.our_actions[1]


def test_zero_probability_column_does_not_affect_choice():
    table = make_table([[2, -100], [1, 100]], [1, 0])
    assert best_response(table).chosen is table.our_actions[0]


def test_exact_ties_choose_first_row():
    table = make_table([[2, 0], [0, 2]], [0.5, 0.5])
    assert best_response(table).chosen is table.our_actions[0]


def test_single_action_table():
    table = make_table([[-7]], [1])
    decision = best_response(table)
    assert decision.chosen is table.our_actions[0]
    assert decision.mixed_strategy == {"move:row0": 1.0}


def test_small_rounding_error_is_normalised_without_mutating_table():
    table = make_table([[3, 3], [1, 1]], [0.4999999, 0.5])
    original = deepcopy(table)
    decision = best_response(table)
    assert "+3.000" in decision.explanation
    assert table == original


def test_labels_keep_moves_switches_and_tera_distinct():
    table = make_table([[1], [2], [3]], [1])
    table.our_actions = [
        Action("move", "same"),
        Action("switch", "same"),
        Action("move", "same", True),
    ]
    decision = best_response(table)
    assert decision.chosen is table.our_actions[2]
    assert decision.mixed_strategy == {"move:same": 0.0, "switch:same": 0.0, "move:same:tera": 1.0}


@pytest.mark.parametrize(
    "probabilities",
    [
        None,
        [],
        [1],
        [0.2, 0.3, 0.5],
        [[0.5, 0.5]],
        [-0.1, 1.1],
        [0, 0],
        [1, 1],
        [0.4, 0.4],
        [float("nan"), 1],
        [float("inf"), 0],
    ],
)
def test_rejects_missing_or_invalid_opponent_probabilities(probabilities):
    table = make_table([[1, 2], [3, 4]], probabilities)
    with pytest.raises(ValueError):
        best_response(table)


@pytest.mark.parametrize(
    "table",
    [
        PayoffTable([], [Action("move", "a")], [], [1]),
        PayoffTable([Action("move", "a")], [], [[]], []),
        PayoffTable([Action("move", "a")], [Action("move", "b")], [[1, 2]], [1]),
        PayoffTable([Action("move", "a")], [Action("move", "b")], [1], [1]),
        make_table([[1, 2], [3]], [0.5, 0.5]),
        make_table([[float("nan")]], [1]),
        make_table([[float("inf")]], [1]),
        PayoffTable([Action("move", "a")] * 2, [Action("move", "b")], [[1], [2]], [1]),
    ],
)
def test_rejects_invalid_tables(table):
    with pytest.raises(ValueError):
        best_response(table)
