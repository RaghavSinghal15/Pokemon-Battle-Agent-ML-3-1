"""Clean and filter battles."""

import pandas as pd


def filter_battles(df: pd.DataFrame, min_rating: int) -> pd.DataFrame:
    """Return only the battles worth learning from.

    TODO (role B):
    - Drop duplicate battle ids.
    - Keep only finished games with a winner (no ties, no games that ended by forfeit on turn 1).
    - Keep battles where BOTH players are rated at least `min_rating`. Choose the threshold from
      the rating distribution in the data report (e.g. the top 30 to 40 percent), not a guess,
      and record why in the report.
    - Log how many battles each step removed; this goes into the data report.
    """
    raise NotImplementedError
