"""Train / validation / test split, made ONCE and shared by everyone.

Rules (these are viva questions, so understand why):

- Split by BATTLE, never by turn. All turns of one game land in the same set. Turns of one game
  are nearly identical (same Pokémon, same players), so splitting a game would test the model on
  something it has almost seen in training, and the score would look better than it really is.
- Split by TIME. Battles are sorted by upload time: the oldest become training, the next block
  validation, and the newest test. The agent will play games in the future, after everything it
  learned from, and the test set checks exactly that. A random split would let the model learn
  from games played after some test games (a quiet form of leakage) and would hide any drift in
  how people play over time.
- Validation sits between train and test in time. It is used for tuning and calibration, and
  because it is "the future" relative to training, the tuning choices are judged the same way
  the test set will judge the final model.
- The split is saved to config.SPLITS_PATH and never regenerated after models are trained;
  everyone reads battle ids from that file.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from pokeagent.config import SPLITS_PATH

SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    battles: pd.DataFrame, test_frac: float = 0.15, val_frac: float = 0.15
) -> dict[str, list[str]]:
    """Split battle ids into train, validation and test sets by upload time.

    `battles` needs a `battle_id` and an `upload_time` column, one row per battle. With the
    default fractions the oldest 70% is train, the next 15% validation and the newest 15% test.
    """
    if not (0 < test_frac < 1 and 0 < val_frac < 1 and test_frac + val_frac < 1):
        raise ValueError("test_frac and val_frac must be positive and add up to less than 1")
    if battles["battle_id"].duplicated().any():
        raise ValueError("Each battle must appear once; run the filter first")
    if battles["upload_time"].isna().any():
        raise ValueError("Every battle needs an upload_time to be placed in time")

    # Oldest first. The battle id breaks ties between equal times, so the result never depends
    # on the order rows happened to be stored in.
    ordered = battles.sort_values(["upload_time", "battle_id"], kind="stable")
    ids = list(ordered["battle_id"])
    n = len(ids)
    n_test = round(n * test_frac)
    n_val = round(n * val_frac)
    n_train = n - n_val - n_test
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def describe_splits(battles: pd.DataFrame, splits: dict[str, list[str]]) -> dict:
    """Size and date range of each split, for the data report and the splits file."""
    time_of = battles.set_index("battle_id")["upload_time"]

    def day(seconds) -> str:
        return datetime.fromtimestamp(int(seconds), tz=timezone.utc).date().isoformat()

    out = {}
    for name in SPLIT_NAMES:
        times = time_of.loc[splits[name]]
        out[name] = {
            "battles": len(times),
            "first_upload": day(times.min()) if len(times) else None,
            "last_upload": day(times.max()) if len(times) else None,
        }
    return out


def save_splits(
    splits: dict[str, list[str]], path: Path = SPLITS_PATH, overwrite: bool = False
) -> Path:
    """Write the split file. Refuses to replace an existing one unless `overwrite=True`.

    The file holds exactly three keys, {"train": [...], "val": [...], "test": [...]}, because
    role D's feature exporter checks for exactly these. Sizes and dates go in the data summary.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. The split is made once and shared; pass overwrite=True "
            "only if the team has agreed to remake it (and retrain every model)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: splits[name] for name in SPLIT_NAMES}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def load_splits(path: Path = SPLITS_PATH) -> dict[str, list[str]]:
    """Read the shared split file: {"train": [...], "val": [...], "test": [...]}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {name: data[name] for name in SPLIT_NAMES}
