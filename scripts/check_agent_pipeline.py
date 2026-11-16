"""Smoke-check F6 with an explicitly untrained fixed forecast, not an E5 result."""

import argparse
import asyncio
import io
import json
import logging
from pathlib import Path

from pokeagent.agent.player import ExplainableAgent
from pokeagent.config import (
    AGENT_PIPELINE_RESULTS_DIR,
    BATTLE_FORMAT,
    RANDOM_BATTLE_SETS_PATH,
    SEED,
)
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.tournament import run_matchup, source_revision
from pokeagent.interfaces import ACTION_CLASSES


class SmokeForecast:
    """Exercise the model API with equal class probabilities; no model is trained."""

    def __init__(self):
        self.calls = 0

    def predict_proba(self, row: dict[str, float]) -> dict[str, float]:
        """Count a model invocation and return the declared smoke-test forecast."""
        self.calls += 1
        return dict.fromkeys(ACTION_CLASSES, 1 / len(ACTION_CLASSES))


class DemoLog(io.TextIOBase):
    """Count all explanations while keeping only one table and one fallback example."""

    def __init__(self):
        self.turns = 0
        self.table = self.fallback = ""

    def write(self, text: str) -> int:
        """Keep a bounded readable demo, without retaining every battle log."""
        self.turns += 1
        if "Opponent forecast:" in text and not self.table:
            self.table = text
        elif "Fallback:" in text and not self.fallback:
            self.fallback = text
        return len(text)


class ErrorLog(logging.Handler):
    """Count unexpected client errors without storing usernames or protocol logs."""

    def __init__(self):
        super().__init__(logging.ERROR)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


async def main(args: argparse.Namespace) -> dict:
    """Run both strategies through the real public pipeline and save smoke diagnostics."""
    if args.server_dir.resolve() != RANDOM_BATTLE_SETS_PATH.parents[3].resolve():
        raise ValueError("Server provenance must match the checkout supplying public sets")
    source, server = source_revision(), source_revision(args.server_dir)
    if source["tracked_changes"] or server["tracked_changes"]:
        raise ValueError("Commit tracked evaluator/server changes before running")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if any(args.out_dir.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    summary = {
        "forecast": "untrained fixed thirds; plumbing only, not E5",
        "source": source,
        "strategies": [],
    }
    for offset, strategy in enumerate(("best_response", "lp")):
        model, log, errors = SmokeForecast(), DemoLog(), ErrorLog()
        agent = ExplainableAgent(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=args.concurrent,
            strategy=strategy,
            seed=args.seed + offset,
            mode="model",
            opponent_model=model,
            explanation_stream=log,
        )
        opponent = BASELINES["random"](
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=args.concurrent,
        )
        agent.logger.addHandler(errors)
        opponent.logger.addHandler(errors)
        try:
            await run_matchup(
                agent,
                opponent,
                args.games,
                seed=args.seed + offset,
                output_path=args.out_dir / f"{strategy}.json",
                timeout=args.timeout,
                server_revision=server,
            )
            entry = {
                "strategy": strategy,
                "games": agent.n_finished_battles,
                "model_calls": model.calls,
                "explanations": log.turns,
                "decision_counts": dict(agent.decision_counts),
                "fallback_reasons": dict(agent.fallback_reasons),
                "client_errors": errors.count,
            }
            if errors.count or not agent.decision_counts[strategy] or not log.table:
                raise RuntimeError(f"Pipeline smoke check failed: {entry}")
            if log.turns != sum(agent.decision_counts.values()):
                raise RuntimeError("An action did not produce exactly one explanation")
            summary["strategies"].append(entry)
            (args.out_dir / f"{strategy}-demo.txt").write_text(
                (
                    "UNTRAINED FIXED-FORECAST SMOKE TEST; NOT E5\n\n" + log.table + log.fallback
                ).rstrip()
                + "\n",
                encoding="utf-8",
            )
            print(json.dumps(entry), flush=True)
        finally:
            agent.logger.removeHandler(errors)
            opponent.logger.removeHandler(errors)
            await agent.ps_client.stop_listening()
            await opponent.ps_client.stop_listening()
        if source_revision() != source or source_revision(args.server_dir) != server:
            raise RuntimeError("Source changed during the smoke check")
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, default=AGENT_PIPELINE_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--concurrent", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800)
    args = parser.parse_args()
    if args.games <= 0 or args.concurrent <= 0 or not 0 <= args.seed < 2**32 - 1:
        parser.error("games/concurrent must be positive and seed in [0, 2**32-2]")
    asyncio.run(main(args))
