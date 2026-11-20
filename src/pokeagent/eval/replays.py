"""Capture finished tournament games through Showdown's public spectator view."""

import asyncio
import json
import secrets
from pathlib import Path

import websockets
from poke_env.battle import AbstractBattle
from poke_env.player import Player
from websockets.asyncio.client import ClientConnection


async def _name_spectator(socket: ClientConnection, name: str) -> None:
    async for frame in socket:
        if not isinstance(frame, str):
            continue
        for line in frame.splitlines():
            if line.startswith("|challstr|"):
                await socket.send(f"|/trn {name},0,")
            if line.startswith("|updateuser|"):
                fields = line.split("|", 4)
                if len(fields) >= 4 and fields[2].strip() == name and fields[3] == "1":
                    return
    raise RuntimeError("Showdown disconnected before naming the spectator")


async def _room_log(socket: ClientConnection, room: str) -> list[str]:
    await socket.send(f"|/join {room}")
    async for frame in socket:
        if not isinstance(frame, str):
            continue
        lines = frame.splitlines()
        if not lines or lines[0] != f">{room}" or len(lines) < 2:
            continue
        if lines[1].startswith("|noinit|"):
            raise RuntimeError(f"Spectator could not join finished battle {room}")
        if lines[1] == "|init|battle":
            await socket.send(f"|/leave {room}")
            return lines[1:]
    raise RuntimeError(f"Showdown disconnected before sending battle {room}")


async def play_with_public_logs(
    player: Player, opponent: Player, games: int, previous_rooms: set[str]
) -> dict[str, list[str]]:
    """Play the matchup and fetch each new finished room before it can expire."""
    logs: dict[str, list[str]] = {}
    async with websockets.connect(player.ps_client.websocket_url, max_size=None) as socket:
        # Use a separate session and a name unrelated to either player so Showdown
        # serves the spectator channel rather than a player's exact HP and requests.
        name = f"evalspec{secrets.token_hex(5)}"
        await asyncio.wait_for(_name_spectator(socket, name), timeout=10)
        playing = asyncio.create_task(player.battle_against(opponent, n_battles=games))
        try:
            while True:
                done = playing.done()
                if done:
                    await playing  # Surface challenge/client errors before saving anything.
                # poke-env runs its client on a separate event loop and can add rooms
                # while this coroutine awaits network responses.
                for room, battle in player.battles.copy().items():
                    if battle.finished and room not in previous_rooms and room not in logs:
                        logs[room] = await asyncio.wait_for(_room_log(socket, room), timeout=10)
                if done:
                    return logs
                await asyncio.wait({playing}, timeout=0.25)
        finally:
            if not playing.done():
                playing.cancel()
            await asyncio.gather(playing, return_exceptions=True)


def write_public_logs(
    path: Path, battles: dict[str, AbstractBattle], logs: dict[str, list[str]]
) -> None:
    """Validate complete spectator logs and write one JSON line per finished battle."""
    if battles.keys() != logs.keys():
        raise RuntimeError("Spectator logs do not cover exactly the new battles")
    records = []
    for room, battle in sorted(battles.items()):
        lines = logs[room]
        outcome = "win" if battle.won else "loss" if battle.lost else "tie"
        terminal = [line for line in lines if line == "|tie" or line.startswith("|win|")]
        if len(terminal) != 1 or (terminal[0] == "|tie") != (outcome == "tie"):
            raise RuntimeError(f"Incomplete or inconsistent public outcome for {room}")
        if any(line.startswith(("|split|", "|request|")) for line in lines):
            raise RuntimeError(f"Private protocol messages in spectator log for {room}")
        records.append(
            {
                "battle_id": room,
                "player_side": battle.player_role,
                "outcome": outcome,
                "spectator_log": lines,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
