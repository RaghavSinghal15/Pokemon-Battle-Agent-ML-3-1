"""Real public-pipeline games with a declared test predictor, not a trained E5 model."""

import asyncio
import logging
from io import StringIO

import pytest
from poke_env.player import RandomPlayer
from test_agent_model import TestPredictor

from pokeagent.agent.player import ExplainableAgent
from pokeagent.config import BATTLE_FORMAT


@pytest.mark.showdown
@pytest.mark.parametrize("strategy", ["best_response", "lp"])
def test_model_pipeline_plays_and_explains_every_action(strategy, caplog):
    async def play():
        model, log = TestPredictor(), StringIO()
        agent = ExplainableAgent(
            battle_format=BATTLE_FORMAT,
            strategy=strategy,
            mode="model",
            opponent_model=model,
            explanation_stream=log,
            max_concurrent_battles=1,
        )
        opponent = RandomPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=1)
        try:
            await asyncio.wait_for(agent.battle_against(opponent, n_battles=3), 180)
            return agent, model, log.getvalue()
        finally:
            await agent.ps_client.stop_listening()
            await opponent.ps_client.stop_listening()

    agent, model, text = asyncio.run(play())
    assert agent.n_finished_battles == 3
    assert model.rows and agent.decision_counts[strategy] > 0
    assert "Opponent forecast:" in text
    assert text.count("Battle ") == sum(agent.decision_counts.values())
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
