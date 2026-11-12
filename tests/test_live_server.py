"""Tests that need a local Showdown server. Skipped by default; run them with:

pytest -m showdown
"""

import asyncio

import pytest
from poke_env.player import RandomPlayer

from pokeagent.agent.player import ExplainableAgent
from pokeagent.config import BATTLE_FORMAT


@pytest.mark.showdown
def test_agent_completes_games():
    async def play():
        agent = ExplainableAgent(battle_format=BATTLE_FORMAT, max_concurrent_battles=5)
        opponent = RandomPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=5)
        await agent.battle_against(opponent, n_battles=5)
        return agent

    agent = asyncio.run(play())
    assert agent.n_finished_battles == 5
