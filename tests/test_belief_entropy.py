"""Check the role-uncertainty scale used in the battle figure."""

import pytest

from pokeagent.beliefs.plot_entropy import role_entropy


@pytest.mark.parametrize(
    "probabilities,expected",
    [({"a": 1.0, "b": 0.0}, 0.0), ({"a": 0.5, "b": 0.5}, 1.0), (dict.fromkeys("abcd", 0.25), 2.0)],
)
def test_role_entropy_in_bits(probabilities, expected):
    assert role_entropy(probabilities) == expected
