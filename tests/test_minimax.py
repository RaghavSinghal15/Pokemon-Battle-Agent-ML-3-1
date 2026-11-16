"""Known small games check the minimax maths without a server or Pokémon knowledge."""

from types import SimpleNamespace

import numpy as np
import pytest

from pokeagent.config import SEED
from pokeagent.decision import strategies
from pokeagent.decision.strategies import minimax_lp
from pokeagent.interfaces import Action, PayoffTable


def make_table(values: list[list[float]], opp_probs: list[float] | None = None) -> PayoffTable:
    """Give each row and column a distinct legal-looking action."""
    return PayoffTable(
        our_actions=[Action("move", f"row{i}") for i in range(len(values))],
        opp_actions=[Action("move", f"col{j}") for j in range(len(values[0]))],
        values=values,
        opp_probs=opp_probs,
    )


def test_rock_paper_scissors_is_uniform():
    table = make_table([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    decision = minimax_lp(table, rng=np.random.default_rng(SEED))
    probabilities = list(decision.mixed_strategy.values())
    assert probabilities == pytest.approx([1 / 3] * 3)
    assert sum(probabilities) == pytest.approx(1)
    assert np.asarray(probabilities) @ table.values == pytest.approx([0, 0, 0])
    assert decision.chosen in table.our_actions
    assert decision.strategy == "lp"
    assert "worst-case expected payoff" in decision.explanation


@pytest.mark.parametrize("offset", [0, -10])
def test_rectangular_game_has_known_nonuniform_mix(offset):
    # Against the first two columns: 3*x = 2*(1-x), so x = 2/5 and value = 6/5.
    # The third column cannot lower the payoff; shifting every cell leaves the mix unchanged.
    table = make_table([[3 + offset, offset, 4 + offset], [offset, 2 + offset, 4 + offset]])
    decision = minimax_lp(table)
    probabilities = list(decision.mixed_strategy.values())
    assert probabilities == pytest.approx([0.4, 0.6])
    assert min(np.asarray(probabilities) @ table.values) == pytest.approx(1.2 + offset)


def test_strictly_dominant_row_gets_all_probability():
    table = make_table([[2, 3], [0, 1], [-1, -2]])
    decision = minimax_lp(table)
    assert list(decision.mixed_strategy.values()) == pytest.approx([1, 0, 0])
    assert decision.chosen is table.our_actions[0]


@pytest.mark.parametrize("values, expected", [([[-4]], [1]), ([[2], [7]], [0, 1])])
def test_single_action_dimensions(values, expected):
    decision = minimax_lp(make_table(values))
    assert list(decision.mixed_strategy.values()) == pytest.approx(expected)


def test_ignores_opponent_model_probabilities():
    values = [[1, -1], [-1, 1]]
    first = minimax_lp(make_table(values, [1, 0]), rng=np.random.default_rng(SEED))
    second = minimax_lp(make_table(values, [0, 1]), rng=np.random.default_rng(SEED))
    assert first == second
    assert list(first.mixed_strategy.values()) == pytest.approx([0.5, 0.5])


def test_seeded_sampling_repeats_sequence_without_repeating_one_draw():
    table = make_table([[1, -1], [-1, 1]])
    first_rng = np.random.default_rng(SEED)
    second_rng = np.random.default_rng(SEED)
    first = [minimax_lp(table, rng=first_rng).chosen.name for _ in range(20)]
    second = [minimax_lp(table, rng=second_rng).chosen.name for _ in range(20)]
    assert first == second
    assert set(first) == {"row0", "row1"}


def test_default_sampling_advances_a_single_generator(monkeypatch):
    monkeypatch.setattr(strategies, "_RNG", np.random.default_rng(SEED))
    table = make_table([[1, -1], [-1, 1]])
    choices = {minimax_lp(table).chosen.name for _ in range(20)}
    assert choices == {"row0", "row1"}


def test_labels_keep_moves_switches_and_tera_distinct():
    table = make_table([[1], [1], [1]])
    table.our_actions = [
        Action("move", "same"),
        Action("switch", "same"),
        Action("move", "same", True),
    ]
    decision = minimax_lp(table)
    assert set(decision.mixed_strategy) == {"move:same", "switch:same", "move:same:tera"}


@pytest.mark.parametrize(
    "table",
    [
        PayoffTable([], [Action("move", "a")], []),
        PayoffTable([Action("move", "a")], [], [[]]),
        PayoffTable([Action("move", "a")], [Action("move", "b")], [[1, 2]]),
        PayoffTable([Action("move", "a")], [Action("move", "b")], [1]),
        make_table([[1, 2], [3]]),
        make_table([[float("nan")]]),
        make_table([[float("inf")]]),
        PayoffTable([Action("move", "a")] * 2, [Action("move", "b")], [[1], [2]]),
    ],
)
def test_rejects_invalid_tables(table):
    with pytest.raises(ValueError):
        minimax_lp(table)


def test_solver_failure_is_explicit(monkeypatch):
    def fail(*args, **kwargs):
        return SimpleNamespace(success=False, x=None, message="numerical difficulty")

    monkeypatch.setattr(strategies, "linprog", fail)
    with pytest.raises(RuntimeError, match="numerical difficulty"):
        minimax_lp(make_table([[1]]))
