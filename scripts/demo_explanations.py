"""Explanation demo (M7b, #85): play one game against the heuristics bot and show every
decision's reasoning.

With the local server running:
    python scripts/demo_explanations.py
    python scripts/demo_explanations.py --games 100 --quiet --out data/interim/analysis

Prints each turn's explanation (matchup, beliefs about the opponent's set, the opponent
forecast, every option's score and the choice) and saves it, with the game's replay, to
reports/examples/. Needs the trained models in artifacts/ (see README).
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex

from poke_env import AccountConfiguration

from pokeagent.config import BATTLE_FORMAT, REPORTS_DIR
from pokeagent.eval.ablations import build_agent
from pokeagent.eval.baselines import BASELINES

EXAMPLES_DIR = REPORTS_DIR / "examples"


class _Tee:
    """Write the explanation log to the screen and a file at once."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


async def main(games: int, out: Path, quiet: bool = False) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    suffix = token_hex(3)  # unique Showdown names for runs started together
    log_path = out / f"demo-{stamp}-{suffix}.log"
    trace_path = out / f"demo-{stamp}-{suffix}.jsonl"  # for the dashboard (F13)
    with (
        log_path.open("w", encoding="utf-8") as log,
        trace_path.open("w", encoding="utf-8") as trace,
    ):
        agent = build_agent(
            "full",
            account_configuration=AccountConfiguration(f"demo{suffix}", None),
            battle_format=BATTLE_FORMAT,
            explanation_stream=log if quiet else _Tee(sys.stdout, log),
            trace_stream=trace,
            save_replays=str(out),
        )
        opponent = BASELINES["heuristics"](
            account_configuration=AccountConfiguration(f"demoopp{suffix}", None),
            battle_format=BATTLE_FORMAT,
        )
        await agent.battle_against(opponent, n_battles=games)
    print(
        f"\nWon {agent.n_won_battles}/{games}. Explanations: {log_path}; dashboard data:"
        f" {trace_path}; replay in {out}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--out", type=Path, default=EXAMPLES_DIR)
    parser.add_argument("--quiet", action="store_true", help="write the log without printing")
    args = parser.parse_args()
    asyncio.run(main(args.games, args.out, args.quiet))
