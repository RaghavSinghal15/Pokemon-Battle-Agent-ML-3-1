"""Run a baseline matchup and save counts, uncertainty and provenance as JSON."""

import argparse
import asyncio
import json
from pathlib import Path

from pokeagent.config import BATTLE_FORMAT, SEED
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.tournament import run_matchup, source_revision


async def main(args: argparse.Namespace) -> dict:
    """Create and close the two clients around one recorded matchup."""
    server = source_revision(args.server_dir) if args.server_dir else None
    player = BASELINES[args.p1](battle_format=BATTLE_FORMAT, max_concurrent_battles=args.concurrent)
    opponent = BASELINES[args.p2](
        battle_format=BATTLE_FORMAT, max_concurrent_battles=args.concurrent
    )
    try:
        return await run_matchup(
            player,
            opponent,
            args.games,
            seed=args.seed,
            output_path=args.out,
            timeout=args.timeout,
            server_revision=server,
        )
    finally:
        await player.ps_client.stop_listening()
        await opponent.ps_client.stop_listening()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p1", choices=BASELINES, default="heuristics")
    parser.add_argument("--p2", choices=BASELINES, default="random")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--server-dir", type=Path, help="local Showdown checkout for source provenance"
    )
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.games <= 0 or args.concurrent <= 0:
        parser.error("--games and --concurrent must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32-1")
    print(json.dumps(asyncio.run(main(args)), indent=2))
