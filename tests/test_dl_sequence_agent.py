"""DL3 (#78): the live GRU agent rebuilds the opponent's history like the training data."""

import numpy as np
import pytest

pytest.importorskip("torch")  # dl/ needs torch; CI installs only the core extras

from pokeagent.dl.sequence_agent import history_rows, sequence_forecast  # noqa: E402

LOG = """|player|p1|Alice|1|1500
|player|p2|Bob|2|1600
|teamsize|p1|6
|teamsize|p2|6
|switch|p1a: Sparky|Pikachu, L90|100/100
|switch|p2a: Corviknight|Corviknight, L80|100/100
|turn|1
|move|p1a: Sparky|Thunderbolt|p2a: Corviknight
|-damage|p2a: Corviknight|51/100
|move|p2a: Corviknight|Roost|p2a: Corviknight
|-heal|p2a: Corviknight|100/100
|upkeep
|turn|2
|switch|p2a: Garchomp|Garchomp, L80|100/100
|move|p1a: Sparky|Thunderbolt|p2a: Garchomp
|-immune|p2a: Garchomp
|upkeep
|turn|3
|move|p1a: Sparky|Thunderbolt|p2a: Garchomp
|-immune|p2a: Garchomp
|move|p2a: Garchomp|Earthquake|p1a: Sparky
|-damage|p1a: Sparky|0 fnt
|faint|p1a: Sparky
|upkeep
|switch|p1a: Golem|Golem, L90|100/100
|turn|4"""


def test_history_has_one_labelled_row_per_finished_opponent_turn():
    rows = history_rows(LOG, "a", "p2")
    assert [(row["turn"], row["action_class"]) for row in rows] == [
        (1, "non_damaging"),  # Roost
        (2, "switch"),
        (3, "attack"),  # Earthquake
    ]
    # Turn 4 has no choices yet, so it is not in the history; Alice's turns never are.
    assert all(row["turn"] < 4 for row in rows)


def test_history_never_reads_a_turn_that_has_not_finished():
    cut = LOG.split("|turn|3", 1)[0] + "|turn|3"
    assert [row["turn"] for row in history_rows(cut, "a", "p2")] == [1, 2]


class _LastRowModel:
    """Returns row index-based probabilities, to check which row is reported."""

    def predict_proba_table(self, table):
        n = len(table)
        probs = np.zeros((n, 3))
        probs[:, 0] = 1.0
        probs[-1] = [0.2, 0.3, 0.5]
        return probs


def test_forecast_is_the_current_turns_row():
    history = history_rows(LOG, "a", "p2")
    current = {key: value for key, value in history[-1].items() if key != "action_class"}
    forecast = sequence_forecast(_LastRowModel(), history, current)
    assert forecast == {"attack": 0.2, "non_damaging": 0.3, "switch": 0.5}
