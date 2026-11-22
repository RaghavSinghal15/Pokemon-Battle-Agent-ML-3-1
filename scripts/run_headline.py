"""Headline evaluation (A7, #83): the full agent, with each strategy, against every baseline.

With the local server running:
    python scripts/run_headline.py --all --games 1000                 # every pairing
    python scripts/run_headline.py --pairing lp:heuristics --games 250  # one chunk
    python scripts/run_headline.py --report                           # write the table

Chunks are saved to reports/results/headline/<strategy>-<opponent>-<time>.json with provenance
(commit, seed, date, versions); reports/results/headline.md adds them up and checks the
success criteria fixed in docs/09-evaluation-plan.md.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone

from poke_env import AccountConfiguration

from pokeagent.config import BATTLE_FORMAT, SEED
from pokeagent.eval.ablations import build_agent
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.headline import (
    HEADLINE_DIR,
    HEADLINE_REPORT,
    PAIRINGS,
    STRATEGIES,
    markdown,
    summarise,
)
from pokeagent.eval.tournament import run_matchup

CONCURRENT = 8


async def play(pairing: str, games: int) -> None:
    strategy, opponent_name = pairing.split(":")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = f"{strategy[:2]}{opponent_name[:3]}{stamp[-6:]}"
    agent = build_agent(
        STRATEGIES[strategy],
        account_configuration=AccountConfiguration(f"hl{tag}", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=CONCURRENT,
        seed=SEED,
    )
    opponent = BASELINES[opponent_name](
        account_configuration=AccountConfiguration(f"ho{tag}", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=CONCURRENT,
    )
    HEADLINE_DIR.mkdir(parents=True, exist_ok=True)
    path = HEADLINE_DIR / f"{strategy}-{opponent_name}-{stamp}.json"
    try:
        result = await run_matchup(agent, opponent, games, output_path=path)
    finally:
        await agent.ps_client.stop_listening()
        await opponent.ps_client.stop_listening()
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(strategy=strategy, opponent=opponent_name, decisions=dict(agent.decision_counts))
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{pairing}: {result['wins']}/{games}")


async def main(args) -> None:
    pairings = PAIRINGS if args.all else args.pairing or []
    for pairing in pairings:
        await play(pairing, args.games)
    if args.report or args.all:
        rows = summarise()
        HEADLINE_REPORT.write_text(markdown(rows), encoding="utf-8")
        print(markdown(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairing", action="append", choices=PAIRINGS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--games", type=int, default=250)
    parser.add_argument("--report", action="store_true")
    asyncio.run(main(parser.parse_args()))
