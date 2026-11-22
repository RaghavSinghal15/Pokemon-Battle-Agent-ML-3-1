"""The recorder counts fresh games correctly when real clients are reused."""

import asyncio
import json
from pathlib import Path

import pytest
from poke_env.player import MaxBasePowerPlayer, RandomPlayer

from pokeagent.config import BATTLE_FORMAT
from pokeagent.eval import tournament
from pokeagent.eval.tournament import run_matchup
from pokeagent.parser.replay import iter_public_states


@pytest.mark.showdown
def test_record_two_live_matchups_with_reused_players(tmp_path, monkeypatch):
    monkeypatch.setattr(tournament, "AGENT_GAMES_DIR", tmp_path / "agent_games")

    async def play():
        player = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=3)
        opponent = RandomPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=3)
        try:
            first = await run_matchup(
                player,
                opponent,
                3,
                output_path=tmp_path / "first.json",
                timeout=60,
                save_replays=True,
            )
            second = await run_matchup(
                player,
                opponent,
                2,
                output_path=tmp_path / "second.json",
                timeout=60,
                replay_path=tmp_path / "second.jsonl",
            )
            assert player.n_finished_battles == opponent.n_finished_battles == 5
            return first, second
        finally:
            await player.ps_client.stop_listening()
            await opponent.ps_client.stop_listening()

    first, second = asyncio.run(play())
    assert first["games"] == 3 and second["games"] == 2
    assert Path(first["replay_path"]).parent == tmp_path / "agent_games"
    assert Path(second["replay_path"]) == tmp_path / "second.jsonl"
    seen = set()
    for record in (first, second):
        assert record["wins"] + record["losses"] + record["ties"] == record["games"]
        summary = json.loads(Path(record["result_path"]).read_text())
        assert summary["games"] == record["games"]
        assert "replay_path" not in summary
        games = [json.loads(line) for line in Path(record["replay_path"]).read_text().splitlines()]
        assert len(games) == record["games"]
        assert sum(game["outcome"] == "win" for game in games) == record["wins"]
        for game in games:
            room, lines = game["battle_id"], game["spectator_log"]
            assert room not in seen
            assert set(game) == {"battle_id", "player_side", "outcome", "spectator_log"}
            assert game["player_side"] in ("p1", "p2")
            assert lines[0] == "|init|battle"
            assert not any(line.startswith(("|split|", "|request|")) for line in lines)
            for line in lines:
                if line.startswith(("|switch|", "|drag|", "|-damage|", "|-heal|")):
                    hp = line.split("|")[4 if line.startswith(("|switch|", "|drag|")) else 3]
                    if "/" in hp:
                        assert hp.split()[0].endswith("/100")
            assert list(iter_public_states(room, lines))
            seen.add(room)
