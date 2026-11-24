"""DL3 (#78): the full agent with DL1's MLP as its opponent model, against the heuristics bot.

    python -m pokeagent.dl.play_games --games 250

Everything else is the full agent of the ablation runs (same payoff table, beliefs, value
utility, tera), so its win rate compares directly with the "full" ablation row, which uses
LightGBM. Lives in dl/ because unpickling the MLP needs torch.

The GRU (DL2) is not played: it needs the opponent's earlier actions as a sequence, and the
live agent only builds one turn's features (docs/decisions.md, DL3).
"""

import argparse
import asyncio
import json
import pickle
from datetime import datetime, timezone
from secrets import token_hex

from poke_env import AccountConfiguration

from pokeagent.config import ARTIFACTS_DIR, BATTLE_FORMAT, REPORTS_DIR, SEED
from pokeagent.eval.ablations import build_agent, switched_off
from pokeagent.eval.baselines import BASELINES
from pokeagent.eval.tournament import run_matchup

MLP_PATH = ARTIFACTS_DIR / "opponent_model_mlp.pkl"
GAMES_DIR = REPORTS_DIR / "results" / "dl_games"
CONCURRENT = 8


async def play(games: int) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = token_hex(3)  # unique Showdown names for chunks started together
    agent = build_agent(
        "full",
        account_configuration=AccountConfiguration(f"dlmlp{suffix}", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=CONCURRENT,
        seed=SEED,
    )
    with MLP_PATH.open("rb") as stream:
        agent.opponent_model = pickle.load(stream)  # swap LightGBM for the MLP, nothing else
    opponent = BASELINES["heuristics"](
        account_configuration=AccountConfiguration(f"dlopp{suffix}", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=CONCURRENT,
    )
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    path = GAMES_DIR / f"mlp-{stamp}-{suffix}.json"
    try:
        with switched_off("full"):
            result = await run_matchup(agent, opponent, games, output_path=path)
    finally:
        await agent.ps_client.stop_listening()
        await opponent.ps_client.stop_listening()
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update(variant="mlp", opponent_model=agent.opponent_model.name)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"mlp: {result['wins']}/{games}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--games", type=int, default=250)
    asyncio.run(play(parser.parse_args().games))
