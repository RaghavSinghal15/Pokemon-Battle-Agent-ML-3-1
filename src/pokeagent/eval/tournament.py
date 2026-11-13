"""Run and record fresh games between two idle players on the local server."""

import asyncio
import json
import math
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from numbers import Integral
from pathlib import Path

import numpy as np
from poke_env.player import Player

from pokeagent.config import MATCHUP_RESULTS_DIR, REPO_ROOT, SEED
from pokeagent.eval.stats import win_rate_ci


def source_revision(directory: Path = REPO_ROOT) -> dict:
    """Record the exact Git revision and whether tracked files were modified.

    Fail rather than fabricate provenance if Git is unavailable. Ignored/untracked result
    files do not dirty the recorded source; source changes should be committed before runs.
    """
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=directory, text=True, stderr=subprocess.DEVNULL
    ).strip()
    tracked = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=directory, text=True
    ).strip()
    return {"commit": commit, "tracked_changes": bool(tracked)}


async def run_matchup(
    player: Player,
    opponent: Player,
    n_games: int,
    *,
    seed: int = SEED,
    output_path: Path | None = None,
    timeout: float | None = None,
    server_revision: dict | None = None,
) -> dict:
    """Play new games, save one JSON record and return counts plus a Wilson win interval.

    Players must be idle, distinct and use the same format; do not share them with another
    concurrent matchup. Earlier completed battles are excluded. Ties count as non-wins,
    not half-wins. This seeds Python/NumPy choices, not the server's random team generator.
    The caller owns the clients and is responsible for closing them after use.
    """
    if isinstance(n_games, bool) or not isinstance(n_games, Integral) or n_games <= 0:
        raise ValueError("n_games must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer between 0 and 2**32-1")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite and positive")
    if player is opponent or player.format != opponent.format:
        raise ValueError("Use two distinct players with the same battle format")
    for bot in (player, opponent):
        if any(not battle.finished for battle in bot.battles.values()):
            raise ValueError("Players must have no unfinished battles")
    before = [set(bot.battles) for bot in (player, opponent)]
    source = source_revision()
    started = datetime.now(timezone.utc)
    if output_path is None:
        output_path = MATCHUP_RESULTS_DIR / f"matchup-{started.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Result already exists: {output_path}")
    random.seed(int(seed))
    np.random.seed(int(seed))
    start = time.perf_counter()
    await asyncio.wait_for(player.battle_against(opponent, n_battles=int(n_games)), timeout)
    elapsed = time.perf_counter() - start
    fresh = [
        {k: b for k, b in bot.battles.items() if k not in old}
        for bot, old in zip((player, opponent), before, strict=True)
    ]
    if set(fresh[0]) != set(fresh[1]) or len(fresh[0]) != n_games:
        raise RuntimeError("New battle counts disagree with the requested matchup")
    if any(not b.finished for games in fresh for b in games.values()):
        raise RuntimeError("A requested battle did not finish")
    wins = sum(b.won is True for b in fresh[0].values())
    losses = sum(b.lost is True for b in fresh[0].values())
    ties = n_games - wins - losses
    if wins != sum(b.lost is True for b in fresh[1].values()) or losses != sum(
        b.won is True for b in fresh[1].values()
    ):
        raise RuntimeError("Player outcome records disagree")
    if ties != sum(b.won is None and b.lost is None for b in fresh[0].values()):
        raise RuntimeError("Invalid tie outcome records")
    rate, low, high = win_rate_ci(wins, n_games)
    result = {
        "schema_version": 1,
        "player_class": type(player).__name__,
        "opponent_class": type(opponent).__name__,
        "format": player.format,
        "games": int(n_games),
        "wins": wins,
        "losses": losses,
        "ties": int(ties),
        "win_rate": rate,
        "ci_lower": low,
        "ci_upper": high,
        "interval_method": "wilson",
        "z": 1.96,
        "seed": int(seed),
        "seed_scope": "Python/NumPy choices only; server teams are not seeded",
        "source": source,
        "server_source": server_revision,
        "started_utc": started.isoformat(),
        "elapsed_seconds": elapsed,
        "python_version": platform.python_version(),
        "poke_env_version": version("poke-env"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return {**result, "result_path": str(output_path)}
