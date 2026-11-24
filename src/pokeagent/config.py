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
EXACT_MOVE_MODEL_PATH = ARTIFACTS_DIR / "exact_move_model.pkl"  # E6, used by the payoff table
VALUE_MODEL_PATH = ARTIFACTS_DIR / "value_model.pkl"  # P(win) from a position (V2)
# The payoff utility (V4) uses the logistic value model: fast to apply to every outcome.
VALUE_UTILITY_MODEL_PATH = ARTIFACTS_DIR / "value_model_logistic_regression.pkl"

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
# Sent once at the start of every public game, so opponents know they are playing a bot.
LADDER_GREETING = "Hi! I'm a registered bot for a university machine-learning project. glhf!"

# The Showdown format we play. "gen9randombattle" = Generation 9, singles, random teams.
BATTLE_FORMAT = "gen9randombattle"
RANDOM_BATTLE_GENERATION = 9
RANDOM_BATTLE_MOVE_SLOTS = 4

# Use the same Showdown checkout that runs local battles.
RANDOM_BATTLE_SETS_PATH = (
    REPO_ROOT.parent / "pokemon-showdown" / "data" / "random-battles" / "gen9" / "sets.json"
)

# Where Random Battle items come from: items are not in sets.json (teams.ts picks them from
# the moveset), so beliefs.sample_items samples this many teams from the local simulator and
# saves (species, moves, ability, item) counts here. Item beliefs (#119) read that file.
RANDOM_BATTLE_SHOWDOWN_DIR = REPO_ROOT.parent / "pokemon-showdown"
RANDOM_BATTLE_ITEMS_PATH = INTERIM_DIR / "random_battle_items.json"
RANDOM_BATTLE_ITEM_TEAMS = 50_000

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
# Entry hazards are valued as the entry damage one switch-in costs every remaining Pokemon on
# that side, times this weight: how many more times each of them is expected to come in. In our
# training replays a Pokemon that still has to enter does so 1.6 times after turn 1 (1.3 after
# turn 20); Heavy-Duty Boots and hazard removal (Rapid Spin, Defog) cancel some of that, so we
# use 1.0. (Before measuring, this was a guessed 0.5.)
PAYOFF_HAZARD_WEIGHT = 1.0
# When the opponent may switch to a Pokemon we have not seen yet, our attack can be scored
# against this many stand-in species drawn from the Random Battle pool (fixed per battle by
# SEED). Without them, an attack into an unseen switch-in does no damage (F13). Measured in
# games: stand-ins lift minimax from 32% to 62% wins but lower best response from 74.5% to
# 70%, so they are used for these strategies only (docs/decisions.md, F13).
PAYOFF_UNSEEN_SAMPLES = 8
PAYOFF_STAND_INS_FOR = ("lp",)
# F15: within the opponent's attacks, a move's weight is multiplied by
# exp(expected damage to us / this), so they are expected to pick the attacks that hurt most
# (a 0.1 temperature makes a move doing 30% more of our HP about 20 times as likely).
# 0 turns it off: attacks are then weighted only by how likely they have them.
PAYOFF_ATTACK_TEMPERATURE = 0.1
# Terastallizing can be done once per battle, by any one of our Pokemon. Using it now gives up
# using it later, which the one-turn look ahead cannot see. Tera rows pay this cost, in
# fractions of one Pokemon's HP (converted to P(win) like hazards). Tuned in games (F14).
PAYOFF_TERA_COST = 0.25
# #119: unrevealed items and abilities that change the turn, weighted by their belief.
# Damage multipliers of the opponent's possible items (others count as 1).
PAYOFF_HIDDEN_ITEM_DAMAGE = {
    "lifeorb": {"physical": 1.3, "special": 1.3},
    "choiceband": {"physical": 1.5, "special": 1.0},
    "choicespecs": {"physical": 1.0, "special": 1.5},
}
# #121: if this many turns in a row change nobody's HP and every row scores within
# PAYOFF_STALL_EPSILON of zero, the one-turn table cannot see a way forward: switch instead.
PAYOFF_STALL_TURNS = 3
PAYOFF_STALL_EPSILON = 5e-4
# F12 (#70): two-turn search. Expand our best rows x their most likely moves; each expanded
# cell gains DISCOUNT * (next turn's table value from the end-of-turn HP - from the HP now).
PAYOFF_SEARCH = False  # the agent's default; the "two_turn" ablation turns it on
PAYOFF_SEARCH_ROWS = 3
PAYOFF_SEARCH_COLUMNS = 2
PAYOFF_SEARCH_DISCOUNT = 1.0
RANDOM_BATTLE_TEAM_SIZE = 6

# Local Showdown server started with `node pokemon-showdown start --no-security`.
LOCAL_SERVER_URL = "ws://localhost:8000/showdown/websocket"
