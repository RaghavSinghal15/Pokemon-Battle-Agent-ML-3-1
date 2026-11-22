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
AGENT_GAMES_DIR = INTERIM_DIR / "agent_games"  # local public tournament replays
PROCESSED_DIR = DATA_DIR / "processed"  # decision points and feature tables
SPLITS_PATH = DATA_DIR / "splits.json"  # which battle ids are train / validation / test
DECISION_POINTS_PATH = PROCESSED_DIR / "decision_points.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"
# Turns whose choice the log hides (fainted first, couldn't move): used only to correct the
# label bias in the opponent model (E4), never as labelled examples.
HIDDEN_POINTS_PATH = PROCESSED_DIR / "hidden_points.parquet"
HIDDEN_FEATURES_PATH = PROCESSED_DIR / "hidden_features.parquet"
# Exact-move model (E6): one row per (decision where a move was used, candidate move).
MOVE_PAIRS_PATH = PROCESSED_DIR / "move_pairs.parquet"
# Value model (V1): one row per decision point, labelled with whether that player won.
VALUE_FEATURES_PATH = PROCESSED_DIR / "value_features.parquet"

# Trained models and other large outputs (git-ignored).
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OPPONENT_MODEL_PATH = ARTIFACTS_DIR / "opponent_model.pkl"  # the model the agent loads
VALUE_MODEL_PATH = ARTIFACTS_DIR / "value_model.pkl"  # P(win) from a position (V2)

# Small results that ARE committed: figures, result tables, the report.
REPORTS_DIR = REPO_ROOT / "reports"
MATCHUP_RESULTS_DIR = REPORTS_DIR / "results" / "matchups"
BASELINE_RESULTS_DIR = REPORTS_DIR / "results" / "baselines"
BASELINE_REPORT_PATH = REPORTS_DIR / "results" / "baselines.md"
AGENT_V0_RESULTS_DIR = REPORTS_DIR / "results" / "agent-v0"
AGENT_PIPELINE_RESULTS_DIR = REPORTS_DIR / "results" / "agent-pipeline-smoke"
# Public ladder games (A8). Per-game logs name other players, so they stay in data/
# (git-ignored); only the summary with no usernames goes to reports/.
LADDER_GAMES_DIR = INTERIM_DIR / "ladder"
LADDER_RESULTS_DIR = REPORTS_DIR / "results" / "ladder"

# The Showdown format we play. "gen9randombattle" = Generation 9, singles, random teams.
BATTLE_FORMAT = "gen9randombattle"
RANDOM_BATTLE_GENERATION = 9
RANDOM_BATTLE_MOVE_SLOTS = 4

# Use the same Showdown checkout that runs local battles.
RANDOM_BATTLE_SETS_PATH = (
    REPO_ROOT.parent / "pokemon-showdown" / "data" / "random-battles" / "gen9" / "sets.json"
)

# Hugging Face dataset with recorded human games.
HF_REPLAY_DATASET = "HolidayOugi/pokemon-showdown-replays"
# The exact dataset version (a commit of the dataset repo) we download. Pinning it means the
# owner adding or changing files later cannot silently change our data. Same revision the
# parser was checked against (docs/08, C5).
HF_REPLAY_REVISION = "15a0a455dbcb7f5c07c2dd670f9252f275d4ddc8"
# The dataset stores each format in its own Parquet files; ours start with this name
# ("[Gen 9] RANDOMBATTLE_part1.parquet", "_part2", ...).
HF_REPLAY_FILE_PREFIX = "[Gen 9] RANDOMBATTLE"

# Where the downloaded Gen 9 Random Battle table is written (one row per battle, unfiltered).
RAW_BATTLES_PATH = RAW_DIR / f"{BATTLE_FORMAT}.parquet"
# Filtering (role B, data/filter.py). Games shorter than this many turns are dropped: they are
# mostly early forfeits and teach the model almost nothing.
MIN_TURNS = 2
# We keep the battles whose LOWER-rated player is in the top this-fraction of the rating
# distribution, so both players are reasonably skilled. The rating cut-off this produces is
# printed by the data build and recorded in reports/data_report.md.
RATING_KEEP_FRACTION = 0.35

# The filtered table every later step reads (role C's exporter reads this path by default).
BATTLES_PATH = INTERIM_DIR / "battles.parquet"

# One seed for everything random (splits, model training, sampling), so results reproduce.
SEED = 464

# Provisional one-turn decision policy, not fitted model parameters.
PAYOFF_KO_BONUS = 0.25
PAYOFF_SLEEP_ACT_CHANCE = 1 / 3  # Approximation; the hidden sleep timer is not inferred.
PAYOFF_FREEZE_ACT_CHANCE = 0.2
# Lasting effects (stat changes, status, hazards) are valued as the change they make to next
# turn's table value, times this many turns. Tuned in games (docs/decisions.md, F8).
PAYOFF_EFFECT_TURNS = 1.0
# Entry hazards are valued as the entry damage they will cost every remaining Pokemon on that
# side (each must switch in at least once to be used), times this discount for those that
# never come in before the game ends.
PAYOFF_HAZARD_WEIGHT = 0.5
RANDOM_BATTLE_TEAM_SIZE = 6

# Local Showdown server started with `node pokemon-showdown start --no-security`.
LOCAL_SERVER_URL = "ws://localhost:8000/showdown/websocket"
