"""Win-rate statistics."""


def win_rate_ci(wins: int, games: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (win rate, lower bound, upper bound) of a 95% confidence interval.

    TODO (role A): use the Wilson score interval, and be able to explain in the viva why a
    win rate from 500 games is uncertain by roughly plus or minus 4 percentage points.
    """
    raise NotImplementedError
