"""Run the ablations (A6, #35) against SimpleHeuristicsPlayer and write one table.

With the local server running:
    python scripts/run_ablations.py --all --games 1000     # every variant, one after another
    python scripts/run_ablations.py --variant lp --games 250   # one chunk of one variant
    python scripts/run_ablations.py --report               # table from every saved chunk

Each chunk is saved to reports/results/ablations/<variant>-<time>.json with provenance;
reports/results/ablations.md adds them up.
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone

from poke_env import AccountConfiguration

from pokeagent.config import BATTLE_FORMAT, SEED
from pokeagent.eval.ablations import (
    ABLATION_REPORT_PATH,
    ABLATION_RESULTS_DIR,
    VARIANTS,
    build_agent,
    lasting_effects,
    markdown,
    summarise,
)
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.tournament import run_matchup

CONCURRENT = 8


async def play(variant: str, games: int) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    tag = f"{variant[:8]}{stamp[-6:]}"
    agent = build_agent(
        variant,
        account_configuration=AccountConfiguration(f"abl{tag}", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=CONCURRENT,
        seed=SEED,
    )
    opponent = BASELINES["heuristics"](
        account_configuration=AccountConfiguration(f"opp{tag}", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=CONCURRENT,
    )
    ABLATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = ABLATION_RESULTS_DIR / f"{variant}-{stamp}.json"
    enabled = VARIANTS[variant][1].get("effects", True)
    try:
        with lasting_effects(enabled):
            result = await run_matchup(agent, opponent, games, output_path=path)
    finally:
        await agent.ps_client.stop_listening()
        await opponent.ps_client.stop_listening()
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(
        variant=variant,
        decisions=dict(agent.decision_counts),
        fallbacks={k[:90]: v for k, v in agent.fallback_reasons.items()},
    )
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    low, high = result["ci_lower"], result["ci_upper"]
    print(f"{variant}: {result['wins']}/{games} ({low:.1%} to {high:.1%})")
    return record


def report() -> None:
    rows = summarise()
    ABLATION_REPORT_PATH.write_text(markdown(rows), encoding="utf-8")
    print(markdown(rows))


async def main(args) -> None:
    variants = list(VARIANTS) if args.all else args.variant or []
    for variant in variants:
        await play(variant, args.games)
    if args.report or args.all:
        report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", choices=list(VARIANTS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--games", type=int, default=250)
    parser.add_argument("--report", action="store_true")
    asyncio.run(main(parser.parse_args()))
