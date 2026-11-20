"""Known intervals and boundary cases for the win-probability statistic."""

import pytest

from pokeagent.eval.stats import win_rate_ci


def test_fifty_wins_in_one_hundred():
    assert win_rate_ci(50, 100) == pytest.approx((0.5, 0.40382982859014716, 0.5961701714098528))


def test_zero_and_all_wins_still_have_uncertainty():
    assert win_rate_ci(0, 100) == pytest.approx((0, 0, 0.03699480747600191))
    assert win_rate_ci(100, 100) == pytest.approx((1, 0.9630051925239981, 1))
    assert 0 < win_rate_ci(0, 1)[2] < 1
    assert 0 < win_rate_ci(1, 1)[1] < 1


def test_more_games_narrow_the_interval():
    small = win_rate_ci(50, 100)
    large = win_rate_ci(500, 1000)
    assert large[1] > small[1] and large[2] < small[2]


def test_complement_and_z_level():
    low = win_rate_ci(20, 100)
    high = win_rate_ci(80, 100)
    assert low[1] == pytest.approx(1 - high[2])
    assert low[2] == pytest.approx(1 - high[1])
    assert win_rate_ci(20, 100, z=2.576)[1] < low[1]


@pytest.mark.parametrize(
    "wins,games",
    [
        (0, 0),
        (1, 0),
        (-1, 10),
        (11, 10),
        (1, -1),
        (True, 10),
        (1, False),
        (1.0, 10),
        (1, 10.0),
        ("1", 10),
    ],
)
def test_invalid_counts(wins, games):
    with pytest.raises(ValueError):
        win_rate_ci(wins, games)


@pytest.mark.parametrize("z", [0, -1, float("nan"), float("inf"), 1e308, True, "1.96"])
def test_invalid_z(z):
    with pytest.raises(ValueError):
        win_rate_ci(1, 10, z)
