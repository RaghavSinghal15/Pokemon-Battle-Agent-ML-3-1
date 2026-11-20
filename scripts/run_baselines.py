"""Play 1,000 games per baseline pair, balanced by challenge direction, and write the report."""

import argparse
import asyncio
import os
from itertools import combinations
from pathlib import Path

from pokeagent.config import BASELINE_REPORT_PATH, BASELINE_RESULTS_DIR, BATTLE_FORMAT, SEED
from pokeagent.eval.baseline_table import render_baseline_report
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.tournament import run_matchup, source_revision


async def main(args: argparse.Namespace) -> None:
    """Run a fixed batch without overwriting or publishing a partial report."""
    source, server = source_revision(), source_revision(args.server_dir)
    if source["tracked_changes"] or server["tracked_changes"]:
        raise ValueError("Commit tracked source/server changes before formal baselines")
    if args.report.exists() or args.out_dir.exists():
        raise FileExistsError("Use fresh report/output paths; existing results are preserved")
    args.out_dir.mkdir(parents=True)
    records, paths = [], []
    for pair_index, (first, second) in enumerate(combinations(BASELINES, 2)):
        for orientation, (p1, p2) in enumerate(((first, second), (second, first))):
            if source_revision() != source or source_revision(args.server_dir) != server:
                raise RuntimeError("Source changed during the baseline batch")
            player = BASELINES[p1](
                battle_format=BATTLE_FORMAT, max_concurrent_battles=args.concurrent
            )
            opponent = BASELINES[p2](
                battle_format=BATTLE_FORMAT, max_concurrent_battles=args.concurrent
            )
            path = args.out_dir / f"{p1}-vs-{p2}.json"
            try:
                record = await run_matchup(
                    player,
                    opponent,
                    args.games_per_pair // 2,
                    seed=args.seed + pair_index * 2 + orientation,
                    output_path=path,
                    timeout=args.timeout,
                    server_revision=server,
                )
            finally:
                await player.ps_client.stop_listening()
                await opponent.ps_client.stop_listening()
            records.append(record)
            paths.append(os.path.relpath(path, args.report.parent).replace(os.sep, "/"))
            print(
                f"{p1} vs {p2}: {record['wins']} wins, {record['losses']} losses, "
                f"{record['ties']} ties in {record['games']} games",
                flush=True,
            )
    if source_revision() != source or source_revision(args.server_dir) != server:
        raise RuntimeError("Source changed during the baseline batch")
    report = render_baseline_report(records, paths)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write(report)
    print(f"Report saved to {args.report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-dir", type=Path, required=True)
    parser.add_argument("--games-per-pair", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--out-dir", type=Path, default=BASELINE_RESULTS_DIR)
    parser.add_argument("--report", type=Path, default=BASELINE_REPORT_PATH)
    args = parser.parse_args()
    if args.games_per_pair <= 0 or args.games_per_pair % 2 or args.concurrent <= 0:
        parser.error("--games-per-pair must be positive and even; --concurrent must be positive")
    if not 0 <= args.seed < 2**32 - 5:
        parser.error("--seed must allow six consecutive NumPy seeds")
    asyncio.run(main(args))
