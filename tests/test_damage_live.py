"""F3 works throughout real Gen 9 games; run with pytest -m showdown -s."""

import asyncio
import random
from collections import Counter
from copy import deepcopy

import pytest
from poke_env.player import Player, SimpleHeuristicsPlayer

from pokeagent.config import BATTLE_FORMAT, SEED
from pokeagent.decision.damage import estimate_move_damage


@pytest.mark.showdown
def test_damage_helper_in_complete_games():
    """Play ten games while checking predictions never change the tracked opponent state."""

    class DamageProbe(Player):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calculations = 0
            self.failed: Counter = Counter()
            self.changed_state = False

        def choose_move(self, battle):
            target = battle.opponent_active_pokemon
            before = None
            if target is not None:
                before = (deepcopy(target.stats), target.item, target.ability, dict(target.boosts))
            predictions = estimate_move_damage(battle)
            if target is not None:
                after = (target.stats, target.item, target.ability, target.boosts)
                self.changed_state |= before != after
            valid = []
            for move in battle.available_moves:
                estimate = predictions.get(move.id)
                if estimate is None:
                    continue
                self.calculations += 1
                if estimate.error:
                    self.failed[estimate.error] += 1
                else:
                    valid.append((estimate.expected_percent, move))
            if valid:
                _, move = max(valid, key=lambda entry: entry[0])
                return self.create_order(move)
            return self.choose_random_move(battle)

    async def play():
        random.seed(SEED)
        probe = DamageProbe(battle_format=BATTLE_FORMAT, max_concurrent_battles=1)
        opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=1)
        try:
            await asyncio.wait_for(probe.battle_against(opponent, n_battles=10), timeout=120)
            return probe
        finally:
            await probe.ps_client.stop_listening()
            await opponent.ps_client.stop_listening()

    probe = asyncio.run(play())
    print(
        f"F3: {probe.n_finished_battles} games, {probe.calculations} move estimates, "
        f"errors={dict(probe.failed)}, changed_state={probe.changed_state}"
    )
    assert probe.n_finished_battles == 10
    assert probe.calculations > 0
    assert not probe.failed
    assert not probe.changed_state
