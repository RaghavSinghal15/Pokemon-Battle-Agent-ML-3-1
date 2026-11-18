"""Download Gen 9 Random Battle replays from Hugging Face and store them locally as Parquet.

The dataset (config.HF_REPLAY_DATASET) holds about 34 million Showdown replays of every format,
stored as one or more Parquet files per format. We only read the Gen 9 Random Battle files and,
inside them, only rows whose `formatid` is exactly `gen9randombattle` (the same files can hold
close variants such as Blitz).

For each battle we keep the raw log (role C parses it) plus a few summary columns read from the
log, so that filtering and splitting never need to parse the log again:

    battle_id     the replay id, e.g. "gen9randombattle-2123456789"
    upload_time   when the replay was uploaded, in Unix seconds (used for the time-based split)
    rating_p1     player 1's ladder rating at the start of the game, or missing if unrated
    rating_p2     same for player 2
    ratings       the same two ratings as JSON, {"p1": 1534, "p2": null}; role C's exporter
                  reads this column
    winner        "p1" or "p2", or missing if the game was a tie or never finished
    tie           True if the game ended in a tie
    turns         number of turns played (the highest "|turn|N" line)
    log           the full public battle log

Player names are used only to work out which side won and are not stored as separate columns.
They do still appear inside the log text, which is one reason the data is never shared.
"""

import json
import re
from pathlib import Path

import pandas as pd

from pokeagent.config import (
    BATTLE_FORMAT,
    HF_REPLAY_DATASET,
    HF_REPLAY_FILE_PREFIX,
    HF_REPLAY_REVISION,
    RAW_BATTLES_PATH,
)

# The columns of the output table, in order. Tests and later steps rely on these names.
BATTLE_COLUMNS = [
    "battle_id",
    "upload_time",
    "rating_p1",
    "rating_p2",
    "ratings",
    "winner",
    "tie",
    "turns",
    "log",
]

# Dataset columns we read. Everything else (views, the players list) is skipped.
SOURCE_COLUMNS = ["id", "formatid", "uploadtime", "log"]

# A player line looks like "|player|p1|Ash|avatar-name|1534". The rating is missing in unrated
# games, and Showdown also repeats "|player|p1|" with no name when someone reconnects.
PLAYER_LINE = re.compile(r"^\|player\|(p[12])\|([^|\n]*)\|[^|\n]*\|?(\d*)", re.M)
WIN_LINE = re.compile(r"^\|win\|(.*)$", re.M)
TIE_LINE = re.compile(r"^\|tie\s*$", re.M)
TURN_LINE = re.compile(r"^\|turn\|(\d+)", re.M)


def to_id(name: str) -> str:
    """Showdown's way of comparing names: lowercase letters and digits only ("Ash K." -> "ashk")."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def summarise_log(log: str) -> dict:
    """Read the ratings, the winner and the number of turns from one battle log.

    Only lines a spectator sees are used, so nothing here is private information.
    """
    names: dict[str, str] = {}
    ratings: dict[str, int | None] = {"p1": None, "p2": None}
    for side, name, rating in PLAYER_LINE.findall(log):
        if name and side not in names:  # the first line with a name is the real one
            names[side] = name
            ratings[side] = int(rating) if rating else None

    winner = None
    win = WIN_LINE.search(log)
    if win:
        # The win line names the player, so match it back to "p1" or "p2".
        winner_id = to_id(win.group(1))
        for side, name in names.items():
            if to_id(name) == winner_id:
                winner = side

    turns = [int(n) for n in TURN_LINE.findall(log)]
    return {
        "rating_p1": ratings["p1"],
        "rating_p2": ratings["p2"],
        "winner": winner,
        "tie": TIE_LINE.search(log) is not None,
        "turns": max(turns) if turns else 0,
    }


def battles_from_rows(rows: list[dict]) -> pd.DataFrame:
    """Turn raw dataset rows into our battle table, keeping only Gen 9 Random Battles.

    Each row is a dict with the dataset's columns (`id`, `formatid`, `uploadtime`, `log`).
    """
    out = []
    for row in rows:
        if row.get("formatid") != BATTLE_FORMAT:
            continue
        log = row.get("log")
        if not isinstance(log, str) or not log:
            continue
        summary = summarise_log(log)
        out.append(
            {
                "battle_id": row["id"],
                "upload_time": row.get("uploadtime"),
                "rating_p1": summary["rating_p1"],
                "rating_p2": summary["rating_p2"],
                "ratings": json.dumps({"p1": summary["rating_p1"], "p2": summary["rating_p2"]}),
                "winner": summary["winner"],
                "tie": summary["tie"],
                "turns": summary["turns"],
                "log": log,
            }
        )
    df = pd.DataFrame(out, columns=BATTLE_COLUMNS)
    # Fixed column types, so every batch written to the Parquet file has the same schema.
    # "Int64" (capital I) is pandas' whole-number type that allows missing values.
    return df.astype(
        {
            "battle_id": "string",
            "upload_time": "Int64",
            "rating_p1": "Int64",
            "rating_p2": "Int64",
            "ratings": "string",
            "winner": "string",
            "tie": "bool",
            "turns": "int64",
            "log": "string",
        }
    )


def list_source_files(revision: str = HF_REPLAY_REVISION) -> list[str]:
    """The dataset's Gen 9 Random Battle files, in part order (part1, part2, ..., part10)."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(HF_REPLAY_DATASET, repo_type="dataset", revision=revision)
    ours = [f for f in files if f.startswith(HF_REPLAY_FILE_PREFIX) and f.endswith(".parquet")]

    def part_number(name: str) -> int:
        match = re.search(r"_part(\d+)\.parquet$", name)
        return int(match.group(1)) if match else 0

    return sorted(ours, key=part_number)


def download_replays(
    out_path: Path = RAW_BATTLES_PATH,
    max_battles: int | None = None,
    revision: str = HF_REPLAY_REVISION,
) -> Path:
    """Stream Gen 9 Random Battles from Hugging Face into one local Parquet file.

    "Streaming" means we read the remote files a few thousand rows at a time and write each
    batch straight to disk, so the full dataset never has to fit in memory.

    `max_battles` stops early, e.g. 5,000 battles for quick development. Battles are taken in
    file order. Returns the path written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    files = list_source_files(revision)
    if not files:
        raise RuntimeError(f"No files starting with {HF_REPLAY_FILE_PREFIX!r} in the dataset")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")  # renamed only when complete
    fs = HfFileSystem()
    writer = None
    written = 0
    try:
        for name in files:
            remote = f"datasets/{HF_REPLAY_DATASET}@{revision}/{name}"
            print(f"Reading {name} ...")
            with fs.open(remote, "rb") as handle:
                source = pq.ParquetFile(handle)
                for batch in source.iter_batches(batch_size=2_000, columns=SOURCE_COLUMNS):
                    battles = battles_from_rows(batch.to_pylist())
                    if max_battles is not None:
                        battles = battles.head(max_battles - written)
                    if battles.empty:
                        continue
                    table = pa.Table.from_pandas(battles, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema)
                    writer.write_table(table)
                    written += len(battles)
                    if max_battles is not None and written >= max_battles:
                        break
            print(f"  {written:,} battles so far")
            if max_battles is not None and written >= max_battles:
                break
    finally:
        if writer is not None:
            writer.close()
    if written == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"No {BATTLE_FORMAT} battles found")
    tmp_path.replace(out_path)
    print(f"Wrote {written:,} battles to {out_path}")
    return out_path


def load_battles(path: Path = RAW_BATTLES_PATH, columns: list[str] | None = None) -> pd.DataFrame:
    """Load a Parquet file of battles. Pass `columns` to skip the (large) log column."""
    return pd.read_parquet(path, columns=columns)
