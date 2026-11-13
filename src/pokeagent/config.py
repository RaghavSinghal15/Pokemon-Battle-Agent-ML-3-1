"""Project-wide constants and paths.

Everything that more than one module needs to agree on (file locations, the battle format,
the random seed) lives here, so nobody hard-codes a path or a number in their own module.
"""

from pathlib import Path

# The repo root is two levels above this file: src/pokeagent/config.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Data folders (all git-ignored; see data/README.md).
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # untouched downloads
INTERIM_DIR = DATA_DIR / "interim"  # filtered battles, one row per battle
PROCESSED_DIR = DATA_DIR / "processed"  # decision points and feature tables
SPLITS_PATH = DATA_DIR / "splits.json"  # which battle ids are train / validation / test

# Trained models and other large outputs (git-ignored).
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

# Small results that ARE committed: figures, result tables, the report.
REPORTS_DIR = REPO_ROOT / "reports"
MATCHUP_RESULTS_DIR = REPORTS_DIR / "results" / "matchups"
BASELINE_RESULTS_DIR = REPORTS_DIR / "results" / "baselines"
BASELINE_REPORT_PATH = REPORTS_DIR / "results" / "baselines.md"

# The Showdown format we play. "gen9randombattle" = Generation 9, singles, random teams.
BATTLE_FORMAT = "gen9randombattle"

# Hugging Face dataset with recorded human games.
HF_REPLAY_DATASET = "HolidayOugi/pokemon-showdown-replays"

# One seed for everything random (splits, model training, sampling), so results reproduce.
SEED = 464

# Local Showdown server started with `node pokemon-showdown start --no-security`.
LOCAL_SERVER_URL = "ws://localhost:8000/showdown/websocket"
