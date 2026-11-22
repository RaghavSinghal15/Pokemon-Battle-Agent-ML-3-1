"""Play the agent on the public Pokemon Showdown ladder against people (A8, #82).

The account name and password come from environment variables, never from the repo:
    SHOWDOWN_USERNAME   the bot account (create it on play.pokemonshowdown.com first)
    SHOWDOWN_PASSWORD   its password

Usage:
    python scripts/play_ladder.py --games 20 --mode model
    python scripts/play_ladder.py --games 2 --local     # rehearsal on the local server

Games are played one at a time. Per-game records, the explanation log and HTML replays go to
data/interim/ladder/<run>/ (git-ignored: they name other players). The summary, with no
usernames, goes to reports/results/ladder/<run>.json.
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.data.normalize import to_id_str

from pokeagent.config import (
    BATTLE_FORMAT,
    LADDER_GAMES_DIR,
    LADDER_RESULTS_DIR,
    OPPONENT_MODEL_PATH,
)
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.ladder import LadderAgent, game_record, summarise
from pokeagent.eval.tournament import source_revision


def agent_options(mode: str) -> dict:
    if mode == "flat":
        return {"mode": "flat"}
    if not OPPONENT_MODEL_PATH.is_file():
        sys.exit(f"No trained opponent model at {OPPONENT_MODEL_PATH}; run pokeagent.models.train")
    with OPPONENT_MODEL_PATH.open("rb") as stream:
        return {"mode": "model", "opponent_model": pickle.load(stream)}


async def main(games: int, strategy: str, mode: str, local: bool) -> None:
    run = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{mode}-{strategy}"
    folder = LADDER_GAMES_DIR / run
    folder.mkdir(parents=True, exist_ok=True)
    options = {
        "battle_format": BATTLE_FORMAT,
        "max_concurrent_battles": 1,  # one game at a time on the public ladder
        "save_replays": str(folder / "replays"),
        "strategy": strategy,
        **agent_options(mode),
    }
    if local:
        # Rehearsal: two local accounts search the local ladder together (no passwords).
        agent = LadderAgent(
            account_configuration=AccountConfiguration("ladderagent", None), **options
        )
        sparring = BASELINES["heuristics"](
            account_configuration=AccountConfiguration("laddersparring", None),
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=1,
        )
    else:
        username, password = (
            os.environ.get("SHOWDOWN_USERNAME"),
            os.environ.get("SHOWDOWN_PASSWORD"),
        )
        if not username or not password:
            sys.exit("Set SHOWDOWN_USERNAME and SHOWDOWN_PASSWORD for the bot account first.")
        agent = LadderAgent(
            account_configuration=AccountConfiguration(username, password),
            server_configuration=ShowdownServerConfiguration,
            **options,
        )
        sparring = None

    start = time.time()
    with (folder / "explanations.log").open("w", encoding="utf-8") as stream:
        agent.explanation_stream = stream
        if sparring is None:
            await agent.ladder(games)
        else:
            await asyncio.gather(agent.ladder(games), sparring.ladder(games))

    our_id = to_id_str(agent.username)
    records = [
        game_record(battle, agent.rating_changes.get(tag, {}), our_id)
        for tag, battle in agent.battles.items()
        if battle.finished
    ]
    with (folder / "games.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    summary = {
        "run": run,
        "server": "local" if local else "showdown",
        "format": BATTLE_FORMAT,
        "mode": mode,
        "strategy": strategy,
        "source": source_revision(),
        "minutes": round((time.time() - start) / 60, 1),
        "decisions": dict(agent.decision_counts),
        "fallbacks": dict(agent.fallback_reasons),
        **summarise(records),
    }
    LADDER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (LADDER_RESULTS_DIR / f"{run}.json").write_text(json.dumps(summary, indent=2), "utf-8")

    low, high = summary["win_rate_95"]
    print(f"{summary['wins']}/{summary['games']} won ({low:.0%} to {high:.0%}, 95% Wilson)")
    print(f"Rating {summary['rating_start']} -> {summary['rating_end']}")
    print(f"Games and logs: {folder}\nSummary: {LADDER_RESULTS_DIR / (run + '.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--strategy", choices=("best_response", "lp"), default="best_response")
    parser.add_argument("--mode", choices=("flat", "model"), default="model")
    parser.add_argument("--local", action="store_true", help="rehearse on localhost:8000")
    args = parser.parse_args()
    asyncio.run(main(args.games, args.strategy, args.mode, args.local))
