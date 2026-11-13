"""The recorder counts fresh games correctly when real clients are reused."""

import asyncio
import json
from pathlib import Path

import pytest
from poke_env.player import MaxBasePowerPlayer, RandomPlayer

from pokeagent.config import BATTLE_FORMAT
from pokeagent.eval.tournament import run_matchup


@pytest.mark.showdown
def test_record_two_live_matchups_with_reused_players(tmp_path):
    async def play():
        player = MaxBasePowerPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=3)
        opponent = RandomPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=3)
        try:
            first = await run_matchup(
                player, opponent, 3, output_path=tmp_path / "first.json", timeout=60
            )
            second = await run_matchup(
                player, opponent, 2, output_path=tmp_path / "second.json", timeout=60
            )
            assert player.n_finished_battles == opponent.n_finished_battles == 5
            return first, second
        finally:
            await player.ps_client.stop_listening()
            await opponent.ps_client.stop_listening()

    first, second = asyncio.run(play())
    assert first["games"] == 3 and second["games"] == 2
    for record in (first, second):
        assert record["wins"] + record["losses"] + record["ties"] == record["games"]
        assert json.loads(Path(record["result_path"]).read_text())["games"] == record["games"]
