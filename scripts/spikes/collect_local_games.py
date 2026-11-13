"""Play bot-vs-bot games on the local server and record THREE views of each battle.

Spike for foundation checks 2, 3 and 4 (see docs/08-foundation-checks.md).

For every battle we save:
1. spectator_log: the public battle log, as a spectator receives it when joining the room.
   This is the same view a recorded replay (our training data) has.
2. p1_stream / p2_stream: the messages each player's client actually received, which include
   that player's own private details (exact HP of their own Pokemon).
3. p1_live / p2_live: a snapshot of each player's live poke-env battle object every time it was
   asked to choose a move, keyed by turn number.

Usage (local server running):
    python scripts/spikes/collect_local_games.py --games 100 --out data/interim/local_games.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets
from poke_env import AccountConfiguration

sys.path.insert(0, str(Path(__file__).parent))
from snapshot import battle_summary  # noqa: E402

from pokeagent.config import BATTLE_FORMAT, LOCAL_SERVER_URL  # noqa: E402
from pokeagent.eval.baselines import BASELINES  # noqa: E402


def recording(base_cls):
    """Wrap a baseline bot class so it snapshots its battle before each choice."""

    class Recording(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.live_snapshots: dict[str, dict[int, dict]] = {}
            self.true_choices: dict[str, dict[int, dict]] = {}

        def choose_move(self, battle):
            order = super().choose_move(battle)
            # Forced replacements after a faint are not turn choices; skip them.
            if not battle.force_switch:
                snaps = self.live_snapshots.setdefault(battle.battle_tag, {})
                # A turn can ask twice (e.g. after an invalid choice); keep the first one.
                snaps.setdefault(battle.turn, battle_summary(battle))
                choice = order.order
                kind = "switch" if hasattr(choice, "species") else "move"
                name = choice.species if kind == "switch" else getattr(choice, "id", str(choice))
                self.true_choices.setdefault(battle.battle_tag, {}).setdefault(
                    battle.turn, {"kind": kind, "name": name}
                )
            return order

    Recording.__name__ = f"Recording{base_cls.__name__}"
    return Recording


async def fetch_spectator_logs(room_ids: list[str]) -> dict[str, list[str]]:
    """Join each finished battle room as a guest spectator and grab the full public log.

    When you join a battle room, Showdown sends everything that has happened so far in one
    message starting with `|init|battle`. That is exactly the spectator view.
    """
    logs: dict[str, list[str]] = {}
    async with websockets.connect(LOCAL_SERVER_URL, max_size=None) as ws:
        async for raw in ws:
            if "|challstr|" in raw:
                await ws.send("|/trn spectatorspike,0,")
            if "|updateuser|" in raw and "spectatorspike" in raw:
                break
        for room in room_ids:
            await ws.send(f"|/join {room}")
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                lines = raw.split("\n")
                if lines[0] == f">{room}" and len(lines) > 1 and lines[1] == "|init|battle":
                    logs[room] = lines[1:]
                    break
                if lines[0] == f">{room}" and "|noinit|" in raw:
                    break  # room expired
            await ws.send(f"|/leave {room}")
    return logs


def stream_lines(battle) -> list[str]:
    """The messages a player's client received and parsed, re-joined into log lines."""
    return ["|".join(m) for m in battle._replay_data]


async def main(p1_kind: str, p2_kind: str, games: int, out: Path, tag: str) -> None:
    p1 = recording(BASELINES[p1_kind])(
        account_configuration=AccountConfiguration(f"{tag}a", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
    )
    p2 = recording(BASELINES[p2_kind])(
        account_configuration=AccountConfiguration(f"{tag}b", None),
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=10,
    )
    await p1.battle_against(p2, n_battles=games)

    rooms = sorted(p1.battles)
    spectator = await fetch_spectator_logs(rooms)

    out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with out.open("a") as f:
        for room in rooms:
            if room not in spectator:
                continue
            b1, b2 = p1.battles[room], p2.battles[room]
            record = {
                "battle_id": room,
                "p1_kind": p1_kind,
                "p2_kind": p2_kind,
                "players": {b1.player_role: p1.username, b2.player_role: p2.username},
                "winner": p1.username if b1.won else p2.username if b2.won else None,
                "spectator_log": spectator[room],
                "streams": {b1.player_role: stream_lines(b1), b2.player_role: stream_lines(b2)},
                "live": {
                    b1.player_role: p1.live_snapshots.get(room, {}),
                    b2.player_role: p2.live_snapshots.get(room, {}),
                },
                "true_choices": {
                    b1.player_role: p1.true_choices.get(room, {}),
                    b2.player_role: p2.true_choices.get(room, {}),
                },
            }
            f.write(json.dumps(record) + "\n")
            n_written += 1
    print(f"{n_written}/{len(rooms)} battles written to {out} ({p1_kind} vs {p2_kind})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--p1", choices=BASELINES, default="heuristics")
    parser.add_argument("--p2", choices=BASELINES, default="heuristics")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("data/interim/local_games.jsonl"))
    parser.add_argument("--tag", default="spike", help="username prefix (must be unique per run)")
    args = parser.parse_args()
    asyncio.run(main(args.p1, args.p2, args.games, args.out, args.tag))
