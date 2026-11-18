"""Build the whole dataset in one command: download, filter, split, summarise.

    python -m pokeagent.data.build --max-battles 5000      # quick development sample
    python -m pokeagent.data.build --max-battles 200000    # the size we train on

Writes data/raw/gen9randombattle.parquet, data/interim/battles.parquet and data/splits.json
(all git-ignored), plus reports/results/data_summary.json: counts, rating and game-length
distributions and split sizes, used for reports/data_report.md. The summary holds only counts
and statistics, no player names or logs, so it is safe to commit and paste.
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pokeagent.config import (
    BATTLES_PATH,
    MIN_TURNS,
    RATING_KEEP_FRACTION,
    RAW_BATTLES_PATH,
    REPORTS_DIR,
    SPLITS_PATH,
)
from pokeagent.data.filter import filter_battle_file, lower_rating, usable_battles
from pokeagent.data.ingest import download_replays, load_battles
from pokeagent.data.splits import describe_splits, make_splits, save_splits

SUMMARY_PATH = REPORTS_DIR / "results" / "data_summary.json"
SMALL_COLUMNS = ["battle_id", "upload_time", "rating_p1", "rating_p2", "winner", "tie", "turns"]


def histogram(values: pd.Series, bin_width: int, upper: int | None = None) -> dict[str, int]:
    """Counts per bin, e.g. {"1500": 812} for ratings 1500 to 1549. Values above `upper` are
    counted in one last bin, so a few very long games do not stretch the plot."""
    values = values.dropna().astype(int)
    if upper is not None:
        values = values.clip(upper=upper)
    bins = (values // bin_width) * bin_width
    return {str(k): int(v) for k, v in bins.value_counts().sort_index().items()}


def quantiles(values: pd.Series) -> dict[str, float]:
    values = values.dropna().astype(float)
    if values.empty:
        return {}
    qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    return {f"p{int(q * 100)}": float(np.quantile(values, q)) for q in qs}


def day(seconds) -> str | None:
    if pd.isna(seconds):
        return None
    return datetime.fromtimestamp(int(seconds), tz=timezone.utc).date().isoformat()


def summarise(raw: pd.DataFrame, kept: pd.DataFrame, filtering: dict, splits_info: dict) -> dict:
    usable = usable_battles(raw)
    both_rated = raw["rating_p1"].notna() & raw["rating_p2"].notna()
    player_ratings = pd.concat([usable["rating_p1"], usable["rating_p2"]])
    return {
        "downloaded": {
            "battles": len(raw),
            "first_upload": day(raw["upload_time"].min()),
            "last_upload": day(raw["upload_time"].max()),
            "share_both_players_rated": round(float(both_rated.mean()), 4),
            "share_with_winner": round(float(raw["winner"].isin(["p1", "p2"]).mean()), 4),
            "share_ties": round(float(raw["tie"].mean()), 4),
            "duplicate_ids": int(raw["battle_id"].duplicated().sum()),
            "turns": quantiles(raw["turns"]),
        },
        "ratings_before_rating_filter": {
            "note": f"usable games (winner, {MIN_TURNS}+ turns), every rated player counted",
            "player_rating_histogram_50": histogram(player_ratings, 50),
            "player_rating_quantiles": quantiles(player_ratings),
            "lower_rating_quantiles": quantiles(lower_rating(usable)),
        },
        "filtering": {
            "min_turns": MIN_TURNS,
            "rating_keep_fraction": filtering["keep_fraction"],
            "min_rating": filtering["min_rating"],
            "counts": filtering["counts"],
        },
        "kept": {
            "battles": len(kept),
            "first_upload": day(kept["upload_time"].min()),
            "last_upload": day(kept["upload_time"].max()),
            "share_p1_wins": round(float((kept["winner"] == "p1").mean()), 4),
            "turns": quantiles(kept["turns"]),
            "turns_histogram_5": histogram(kept["turns"], 5, upper=100),
            "lower_rating_quantiles": quantiles(lower_rating(kept)),
        },
        "splits": splits_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download, filter and split the replay data.")
    parser.add_argument("--max-battles", type=int, default=5_000, help="battles to download")
    parser.add_argument(
        "--skip-download", action="store_true", help="reuse data/raw/ from an earlier run"
    )
    parser.add_argument(
        "--min-rating", type=int, help="use this threshold instead of choosing one from the data"
    )
    parser.add_argument(
        "--overwrite-splits",
        action="store_true",
        help="replace an existing data/splits.json (only if the team agreed)",
    )
    args = parser.parse_args()

    if not args.skip_download:
        download_replays(RAW_BATTLES_PATH, max_battles=args.max_battles)
    raw = load_battles(RAW_BATTLES_PATH, columns=SMALL_COLUMNS)

    print("Filtering ...")
    filtering = filter_battle_file(RAW_BATTLES_PATH, BATTLES_PATH, min_rating=args.min_rating)
    for step, count in filtering["counts"]:
        print(f"  {step}: {count:,}")
    kept = load_battles(BATTLES_PATH, columns=SMALL_COLUMNS)

    print("Splitting ...")
    splits = make_splits(kept)
    splits_info = describe_splits(kept, splits)
    save_splits(splits, SPLITS_PATH, overwrite=args.overwrite_splits)

    summary = summarise(raw, kept, filtering, splits_info)
    summary["settings"] = {
        "max_battles": None if args.skip_download else args.max_battles,
        "rating_keep_fraction": RATING_KEEP_FRACTION,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {SPLITS_PATH} and {SUMMARY_PATH}. Summary (safe to paste):\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
