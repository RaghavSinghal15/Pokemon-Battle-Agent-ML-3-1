"""Clean and filter battles.

Not every recorded game is worth learning from. We drop, in this order:

1. duplicate battle ids (the same game stored twice would be counted twice, and could even end
   up in both the training and the test set);
2. games without a winner (ties and games that never finished have no clean ending);
3. very short games (fewer than config.MIN_TURNS turns, mostly early forfeits);
4. games where either player is unrated or rated below a threshold, so the model learns from
   reasonably skilled players.

The rating threshold is not a guess: `choose_min_rating` picks it from the data, as the rating
that keeps the top config.RATING_KEEP_FRACTION of games when ranked by their LOWER-rated
player. Each step's count is returned so the data report can show how much every rule removed.
"""

from pathlib import Path

import pandas as pd

from pokeagent.config import BATTLES_PATH, MIN_TURNS, RATING_KEEP_FRACTION, RAW_BATTLES_PATH


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset="battle_id", keep="first")


def _has_winner(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["winner"].isin(["p1", "p2"])]


def _long_enough(df: pd.DataFrame, min_turns: int) -> pd.DataFrame:
    return df[df["turns"] >= min_turns]


def _both_rated_at_least(df: pd.DataFrame, min_rating: int) -> pd.DataFrame:
    # A missing rating counts as "not rated at least min_rating", so unrated games go too.
    ok = (df["rating_p1"] >= min_rating) & (df["rating_p2"] >= min_rating)
    return df[ok.fillna(False).astype(bool)]


def usable_battles(df: pd.DataFrame, min_turns: int = MIN_TURNS) -> pd.DataFrame:
    """Steps 1 to 3: the battles left before the rating rule."""
    return _long_enough(_has_winner(_drop_duplicates(df)), min_turns)


def lower_rating(df: pd.DataFrame) -> pd.Series:
    """The lower of the two players' ratings in each battle (missing if either is unrated)."""
    both = df[["rating_p1", "rating_p2"]].astype("Float64")
    return both.min(axis=1, skipna=False)


def choose_min_rating(
    df: pd.DataFrame, keep_fraction: float = RATING_KEEP_FRACTION, min_turns: int = MIN_TURNS
) -> int:
    """The rating threshold that keeps the top `keep_fraction` of usable, rated battles.

    Battles are ranked by their lower-rated player, because "both players rated at least X"
    is the same as "the weaker player is rated at least X".
    """
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be between 0 and 1")
    lower = lower_rating(usable_battles(df, min_turns)).dropna()
    if lower.empty:
        raise ValueError("No battle has both players rated, so no threshold can be chosen")
    # The (1 - keep_fraction) quantile: e.g. keeping the top 35% means cutting at the 65th
    # percentile. Rounded down so that ties at the cut-off are kept.
    return int(lower.quantile(1 - keep_fraction))


def filter_steps(
    df: pd.DataFrame, min_rating: int, min_turns: int = MIN_TURNS
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Apply every rule in order; return the kept battles and the count left after each rule."""
    counts = [("downloaded", len(df))]
    df = _drop_duplicates(df)
    counts.append(("after removing duplicate ids", len(df)))
    df = _has_winner(df)
    counts.append(("after keeping games with a winner", len(df)))
    df = _long_enough(df, min_turns)
    counts.append((f"after keeping games of {min_turns}+ turns", len(df)))
    df = _both_rated_at_least(df, min_rating)
    counts.append((f"after keeping games with both players rated {min_rating}+", len(df)))
    return df, counts


def filter_battles(df: pd.DataFrame, min_rating: int, min_turns: int = MIN_TURNS) -> pd.DataFrame:
    """Return only the battles worth learning from (see the module docstring for the rules)."""
    return filter_steps(df, min_rating, min_turns)[0]


def filter_battle_file(
    raw_path: Path = RAW_BATTLES_PATH,
    out_path: Path = BATTLES_PATH,
    min_rating: int | None = None,
    keep_fraction: float = RATING_KEEP_FRACTION,
) -> dict:
    """Filter the downloaded battle file into data/interim/battles.parquet.

    The rules are decided on the small columns only (the logs are large), then the kept rows,
    logs included, are copied across in batches. If `min_rating` is None it is chosen from the
    data with `choose_min_rating`. Returns the threshold and the per-step counts.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    small = pd.read_parquet(
        raw_path, columns=["battle_id", "rating_p1", "rating_p2", "winner", "turns"]
    )
    chosen_from_data = min_rating is None
    if chosen_from_data:
        min_rating = choose_min_rating(small, keep_fraction)
    kept, counts = filter_steps(small, min_rating)
    keep_ids = set(kept["battle_id"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    copied: set[str] = set()  # so a duplicated id is copied only once
    with pq.ParquetFile(raw_path) as source, pq.ParquetWriter(tmp_path, source.schema_arrow) as out:
        for batch in source.iter_batches(batch_size=2_000):
            rows = []
            for row in batch.to_pylist():
                battle_id = row["battle_id"]
                if battle_id in keep_ids and battle_id not in copied:
                    copied.add(battle_id)
                    rows.append(row)
            if rows:
                out.write_table(pa.Table.from_pylist(rows, schema=source.schema_arrow))
    tmp_path.replace(out_path)
    return {
        "min_rating": min_rating,
        "keep_fraction": keep_fraction if chosen_from_data else None,
        "counts": counts,
    }
