"""Win-rate statistics."""

import math
from numbers import Integral, Real


def win_rate_ci(wins: int, games: int, z: float = 1.96) -> tuple[float, float, float]:
    """Return (win rate, lower, upper) using the Wilson score interval.

    The default z=1.96 gives the usual nominal 95% interval. Count a tie as a non-win
    if it is included in games; half-credit scoring is a different statistic. Reject zero
    games instead of reporting an invented win rate. Bounds remain useful at 0 or all wins.
    """
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in (wins, games)):
        raise ValueError("wins and games must be integer counts")
    if games <= 0 or not 0 <= wins <= games:
        raise ValueError("Require games > 0 and 0 <= wins <= games")
    if isinstance(z, bool) or not isinstance(z, Real) or not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    squared = float(z) * float(z)
    if not math.isfinite(squared):
        raise ValueError("z is too large")
    rate = wins / games
    denominator = 1 + squared / games
    centre = (rate + squared / (2 * games)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / games + squared / (4 * games**2)) / denominator
    return rate, max(0.0, centre - radius), min(1.0, centre + radius)
