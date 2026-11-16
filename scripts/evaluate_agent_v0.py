"""Measure the flat-forecast F5 agent against random and max-power baselines."""

import argparse
import asyncio
import json
from pathlib import Path

from pokeagent.agent.player import ExplainableAgent
from pokeagent.config import AGENT_V0_RESULTS_DIR, BATTLE_FORMAT, RANDOM_BATTLE_SETS_PATH, SEED
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.tournament import run_matchup, source_revision


async def main(args: argparse.Namespace) -> dict:
    """Run both comparisons, saving provenance and fallback diagnostics."""
    if args.server_dir.resolve() != RANDOM_BATTLE_SETS_PATH.parents[3].resolve():
        raise ValueError("--server-dir must match the checkout supplying RANDOM_BATTLE_SETS_PATH")
    source, server = source_revision(), source_revision(args.server_dir)
    if source["tracked_changes"] or server["tracked_changes"]:
        raise ValueError("Commit tracked evaluator/server changes before measurement")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.out_dir / f"vs-{name}.json" for name in ("random", "max_power")]
    summary_path = args.out_dir / "summary.json"
    if any(path.exists() for path in [*targets, summary_path]):
        raise FileExistsError("Choose a fresh output directory; prior results are retained")
    summary = {"mode": "flat", "strategy": args.strategy, "source": source, "matchups": []}
    for index, name in enumerate(("random", "max_power")):
        if source_revision() != source or source_revision(args.server_dir) != server:
            raise RuntimeError("Source changed during the experiment")
        seed = args.seed + index
        agent = ExplainableAgent(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=args.concurrent,
            strategy=args.strategy,
            seed=seed,
        )
        opponent = BASELINES[name](
            battle_format=BATTLE_FORMAT, max_concurrent_battles=args.concurrent
        )
        try:
            result = await run_matchup(
                agent,
                opponent,
                args.games,
                seed=seed,
                output_path=targets[index],
                timeout=args.timeout,
                server_revision=server,
            )
            summary["matchups"].append(
                {
                    "opponent": name,
                    "record": targets[index].name,
                    "wins": result["wins"],
                    "games": result["games"],
                    "ci_lower": result["ci_lower"],
                    "ci_upper": result["ci_upper"],
                    "decision_counts": dict(agent.decision_counts),
                    "fallback_reasons": dict(agent.fallback_reasons),
                }
            )
            print(json.dumps(summary["matchups"][-1]), flush=True)
        finally:
            await agent.ps_client.stop_listening()
            await opponent.ps_client.stop_listening()
    if source_revision() != source or source_revision(args.server_dir) != server:
        raise RuntimeError("Source changed during the experiment")
    summary["beats_both_with_interval_above_half"] = all(
        run["ci_lower"] > 0.5 for run in summary["matchups"]
    )
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--strategy", choices=("best_response", "lp"), default="best_response")
    parser.add_argument("--out-dir", type=Path, default=AGENT_V0_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--concurrent", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.games <= 0 or args.concurrent <= 0 or not 0 <= args.seed < 2**32 - 1:
        parser.error("games/concurrent must be positive and seed in [0, 2**32-2]")
    asyncio.run(main(args))
