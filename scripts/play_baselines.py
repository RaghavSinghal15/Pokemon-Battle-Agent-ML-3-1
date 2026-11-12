"""Make two baseline bots play each other on the local Showdown server.

This is the M0 smoke test: if it prints win counts, the server, poke-env and your Python
environment all work.

Usage (with the local server running, see docs/07-setup.md):
    python scripts/play_baselines.py --games 20
    python scripts/play_baselines.py --p1 heuristics --p2 random --games 100
"""

import argparse
import asyncio
import time

from pokeagent.config import BATTLE_FORMAT
from pokeagent.eval.baselines import BASELINES


async def main(p1_name: str, p2_name: str, games: int) -> None:
    p1 = BASELINES[p1_name](battle_format=BATTLE_FORMAT, max_concurrent_battles=10)
    p2 = BASELINES[p2_name](battle_format=BATTLE_FORMAT, max_concurrent_battles=10)

    start = time.time()
    await p1.battle_against(p2, n_battles=games)
    elapsed = time.time() - start

    print(f"{games} games of {BATTLE_FORMAT} in {elapsed:.1f}s")
    print(f"  {p1_name:>10}: {p1.n_won_battles} wins")
    print(f"  {p2_name:>10}: {p2.n_won_battles} wins")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p1", choices=BASELINES, default="max_power")
    parser.add_argument("--p2", choices=BASELINES, default="random")
    parser.add_argument("--games", type=int, default=20)
    args = parser.parse_args()
    asyncio.run(main(args.p1, args.p2, args.games))
