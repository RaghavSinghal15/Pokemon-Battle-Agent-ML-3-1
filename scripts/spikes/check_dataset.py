"""Check that the replay dataset is what we think it is (foundation check 1).

Run this on a laptop with internet (it reads from Hugging Face). It downloads NOTHING big: it
reads the file list, then streams a sample of Gen 9 Random Battle games and reports:

  - what files and columns the dataset has, and its licence
  - how many Gen 9 Random Battle games there are (when the file format lets us count cheaply)
  - whether each game has a full log, ratings and a date
  - whether logs are the public (spectator) view or contain private details
  - whether poke-env can parse every log (and which log lines it can't handle)
  - how many turns give a usable label for the opponent model
  - how long parsing takes, so we know if parsing every game is feasible

It prints only counts and summaries (no player names, no log text), so the output is safe to
paste into the group chat or commit to reports/results/.

Usage:
    pip install -e ".[dev]"
    python scripts/spikes/check_dataset.py                    # sample 2,000 games from Hugging Face
    python scripts/spikes/check_dataset.py --sample 10000
    python scripts/spikes/check_dataset.py --local data/interim/local_games.jsonl   # offline test

Paste the whole output back to Raghav / Claude.
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from label_visibility import LABELS, visibility  # noqa: E402
from replay_parse import parse_log  # noqa: E402

from pokeagent.config import BATTLE_FORMAT, HF_REPLAY_DATASET  # noqa: E402

PRIVATE_LOOKING = ("player", "user", "name", "p1", "p2", "winner", "loser")


# ---------------------------------------------------------------------------------------------
# Reading rows
# ---------------------------------------------------------------------------------------------


def describe_repo(repo: str) -> list[str]:
    """Print the file list and licence; return paths of files that look like Gen 9 randbats."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(repo, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    print(f"Dataset: {repo}")
    card = info.card_data.to_dict() if info.card_data else {}
    print(f"  licence (from dataset card): {card.get('license', 'NOT STATED')}")
    print(f"  last modified: {info.last_modified}")
    print(f"  files: {len(files)}, total size: {sum(s for _, s in files) / 1e9:.2f} GB")
    exts = Counter(Path(f).suffix or "(none)" for f, _ in files)
    print(f"  file types: {dict(exts.most_common(6))}")
    tops = Counter(f.split("/")[0] if "/" in f else "(root)" for f, _ in files)
    print(f"  top-level folders: {dict(tops.most_common(8))}")
    matches = [(f, s) for f, s in files if BATTLE_FORMAT in f.lower().replace("_", "")]
    print(f"  files with '{BATTLE_FORMAT}' in the name: {len(matches)}")
    for f, s in matches[:10]:
        print(f"    {f}  ({s / 1e6:.1f} MB)")
    return [f for f, _ in matches if not f.endswith((".md", ".gitattributes"))]


def count_parquet_rows(repo: str, paths: list[str]) -> int | None:
    """Row count from parquet footers (cheap: no full download)."""
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        total = 0
        for p in paths:
            if not p.endswith(".parquet"):
                return None
            with fs.open(f"datasets/{repo}/{p}") as f:
                total += pq.ParquetFile(f).metadata.num_rows
        return total
    except Exception as e:  # noqa: BLE001
        print(f"  (could not count rows from parquet metadata: {type(e).__name__}: {e})")
        return None


def hf_rows(repo: str, files: list[str]):
    from datasets import load_dataset

    if files:
        ds = load_dataset(repo, data_files=files, split="train", streaming=True)
    else:
        print("  no per-format files found: streaming the whole dataset and filtering")
        ds = load_dataset(repo, split="train", streaming=True)
    yield from ds


def local_rows(path: Path):
    for line in path.open():
        rec = json.loads(line)
        yield {"id": rec["battle_id"], "format": BATTLE_FORMAT, "log": rec["spectator_log"]}


# ---------------------------------------------------------------------------------------------
# Understanding a row without knowing the column names in advance
# ---------------------------------------------------------------------------------------------


def as_text(value) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        return "\n".join(value)
    return None


def find_log_column(row: dict) -> str | None:
    for k, v in row.items():
        t = as_text(v)
        if t and "|turn|" in t:
            return k
    return None


def is_gen9_randbats(row: dict, log: str) -> bool:
    for v in row.values():
        if isinstance(v, str) and len(v) < 200:
            low = v.lower()
            if low == BATTLE_FORMAT or low.startswith(BATTLE_FORMAT + "-"):
                return True
            if "battle-" + BATTLE_FORMAT + "-" in low:
                return True
    return "|tier|[Gen 9] Random Battle\n" in log or log.rstrip().endswith(
        "|tier|[Gen 9] Random Battle"
    )


def show_schema(row: dict, log_col: str | None) -> None:
    print("\nColumns (first matching row):")
    for k, v in row.items():
        if k == log_col:
            shown = f"<battle log, {len(as_text(v) or '')} characters>"
        elif any(p in k.lower() for p in PRIVATE_LOOKING):
            shown = "<hidden: may be a username>"
        else:
            shown = repr(v)[:70]
        print(f"  {k:<22} {type(v).__name__:<8} {shown}")


RATING_RE = re.compile(r"^\|player\|(p[12])\|([^|]*)\|[^|]*\|(\d+)?", re.M)
PLAYER_RE = re.compile(r"^\|player\|(p[12])\|([^|]+)\|", re.M)
TIME_RE = re.compile(r"^\|t:\|(\d+)", re.M)


def analyse(rows, sample: int, max_scan: int) -> None:
    n_scanned = 0
    first_row_shown = False
    log_col = None
    ids: Counter = Counter()
    ratings: list[int] = []
    rated_games = 0
    rating_cols: Counter = Counter()
    times: list[int] = []
    finished = 0
    turns: list[int] = []
    private = Counter()
    parse_errors: Counter = Counter()
    games_with_errors = 0
    parse_seconds = 0.0
    labels: Counter = Counter()
    games = 0

    for row in rows:
        n_scanned += 1
        if n_scanned > max_scan:
            break
        if log_col is None:
            log_col = find_log_column(row)
            if log_col is None:
                continue
        log = as_text(row.get(log_col))
        if not log or not is_gen9_randbats(row, log):
            continue
        if not first_row_shown:
            show_schema(row, log_col)
            first_row_shown = True
        games += 1

        for k, v in row.items():
            if isinstance(v, str) and len(v) < 100 and re.fullmatch(r"[a-z0-9]+-\d+", v):
                ids[v] += 1
            if "rating" in k.lower() or "elo" in k.lower():
                rating_cols[k] += 1

        game_ratings = [int(r) for _, _, r in RATING_RE.findall(log) if r]
        if game_ratings:
            rated_games += 1
            ratings.extend(game_ratings)
        ts = TIME_RE.findall(log)
        if ts:
            times.append(int(ts[0]))
        if re.search(r"^\|(win|tie)\b", log, re.M):
            finished += 1
        n_turns = len(re.findall(r"^\|turn\|", log, re.M))
        turns.append(n_turns)
        if "|split|" in log:
            private["has |split| lines (private view)"] += 1
        if re.search(r"\|p[12]a: [^|]+\|\d+/(?!100\b)\d+", log):
            private["has exact HP (x/NNN not x/100)"] += 1

        lines = log.split("\n")
        players = dict(PLAYER_RE.findall(log))
        errs: Counter = Counter()
        t0 = time.perf_counter()
        for role in ("p1", "p2"):
            if role in players:
                try:
                    parse_log("check", lines, players[role], errors=errs)
                except Exception as e:  # noqa: BLE001
                    errs[f"crashed: {type(e).__name__}"] += 1
        parse_seconds += time.perf_counter() - t0
        if errs:
            games_with_errors += 1
            parse_errors.update(errs)
        labels.update(visibility(lines))

        if games >= sample:
            break

    print(f"\nScanned {n_scanned} rows, found {games} {BATTLE_FORMAT} games with a log")
    if games == 0:
        print("  !! No matching games found. Paste this output back so we can adjust the script.")
        return
    print(f"  duplicate ids in sample: {sum(v - 1 for v in ids.values() if v > 1)}")
    print(f"  finished (has |win| or |tie|): {finished}/{games} ({100 * finished / games:.1f}%)")
    print(
        f"  turns per game: median {statistics.median(turns):.0f}, "
        f"10th pct {sorted(turns)[len(turns) // 10]}, "
        f"90th pct {sorted(turns)[9 * len(turns) // 10]}"
    )
    print(f"\nRatings: {rated_games}/{games} games have a rating in their |player| lines")
    if rating_cols:
        print(f"  rating-like columns: {dict(rating_cols)}")
    if ratings:
        q = statistics.quantiles(ratings, n=10)
        print(
            f"  player ratings: min {min(ratings)}, "
            f"deciles {[int(x) for x in q]}, max {max(ratings)}"
        )
        for cut in (1100, 1300, 1500, 1700):
            share = sum(r >= cut for r in ratings) / len(ratings)
            print(f"    share of players rated >= {cut}: {100 * share:.1f}%")
    if times:
        lo = datetime.fromtimestamp(min(times), tz=timezone.utc).date()
        hi = datetime.fromtimestamp(max(times), tz=timezone.utc).date()
        print(f"\nDates (from |t:| lines): {lo} to {hi}, {len(times)}/{games} games have one")
    else:
        print("\nDates: no |t:| timestamps in logs; look for a date column above")
    print("\nPublic view?")
    if private:
        for k, v in private.items():
            print(f"  !! {v}/{games} games {k}")
    else:
        print("  yes: no |split| lines and all HP shown as x/100")
    print(
        f"\nParsing with poke-env: {games - games_with_errors}/{games} games parsed with no "
        f"errors ({100 * (games - games_with_errors) / games:.1f}%)"
    )
    for k, v in parse_errors.most_common(12):
        print(f"  {v:>6}  {k}")
    per_game = parse_seconds / games
    print(f"  time: {1000 * per_game:.1f} ms per game (both perspectives)")
    print(f"  => 100,000 games would take ~{100_000 * per_game / 60:.0f} min on one core")
    total = sum(labels.values())
    usable = sum(labels[k] for k in LABELS)
    print(
        f"\nOpponent-model labels: {usable}/{total} (player, turn) pairs usable "
        f"({100 * usable / total:.1f}%)"
    )
    for k, v in labels.most_common():
        print(f"  {k:<14} {v:>8}  ({100 * v / total:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=2000, help="games to analyse")
    parser.add_argument("--max-scan", type=int, default=2_000_000, help="rows to look through")
    parser.add_argument("--local", type=Path, help="analyse a local jsonl instead (offline test)")
    args = parser.parse_args()

    if args.local:
        analyse(local_rows(args.local), args.sample, args.max_scan)
        return
    files = describe_repo(HF_REPLAY_DATASET)
    n = count_parquet_rows(HF_REPLAY_DATASET, files) if files else None
    if n is not None:
        print(f"  rows in those files: {n:,}")
    analyse(hf_rows(HF_REPLAY_DATASET, files), args.sample, args.max_scan)


if __name__ == "__main__":
    main()
