"""Download replays from Hugging Face and store them locally as Parquet."""

from pathlib import Path

import pandas as pd

from pokeagent.config import HF_REPLAY_DATASET, RAW_DIR


def download_replays(out_dir: Path = RAW_DIR, max_battles: int | None = None) -> Path:
    """Download Gen 9 Random Battle replays from HF_REPLAY_DATASET into `out_dir`.

    TODO (role B):
    - Inspect the dataset first (columns, size, which split holds Gen 9 Random Battles).
      The full dataset is large; prefer streaming (`datasets.load_dataset(..., streaming=True)`)
      and keep only the `gen9randombattle` format.
    - `max_battles` lets people grab a small sample for development (e.g. 5,000 battles).
    - Write Parquet, not CSV: it is compressed and loads much faster.
    - Never commit anything under data/ (it is git-ignored) and never re-upload the dataset.
    """
    raise NotImplementedError(f"Download from {HF_REPLAY_DATASET} not implemented yet")


def load_battles(path: Path) -> pd.DataFrame:
    """Load a Parquet file of battles into a DataFrame."""
    return pd.read_parquet(path)
