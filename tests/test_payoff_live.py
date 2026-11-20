"""Observe provisional F4 tables during games without claiming an integrated F5 agent."""

import asyncio
import json
import math
import warnings
from collections import Counter
from copy import deepcopy

import pytest
from poke_env.player import SimpleHeuristicsPlayer

from pokeagent.config import BATTLE_FORMAT
from pokeagent.decision.payoff import (
    PayoffApproximationWarning,
    PayoffUnavailable,
    build_payoff_table,
)


@pytest.mark.showdown
def test_tables_during_live_games():
    """Build from revealed moves only; unsupported states must be explicit fallbacks."""

    class Observer(SimpleHeuristicsPlayer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.tables = 0
            self.unavailable = Counter()
            self.failures = []
            self.sample = None

        def choose_move(self, battle):
            if not battle.force_switch:
                before = (
                    deepcopy(battle.opponent_active_pokemon.stats)
                    if battle.opponent_active_pokemon
                    else None
                )
                try:
                    table = build_payoff_table(battle, {}, None)
                    assert len(table.values) == len(table.our_actions)
                    assert all(len(row) == len(table.opp_actions) for row in table.values)
                    assert all(math.isfinite(value) for row in table.values for value in row)
                    assert table.opp_probs is None
                    self.tables += 1
                    if self.sample is None and any(a.kind == "move" for a in table.opp_actions):
                        self.sample = {
                            "our_species": battle.active_pokemon.species,
                            "opponent_species": battle.opponent_active_pokemon.species,
                            "rows": [f"{a.kind}:{a.name}" for a in table.our_actions],
                            "columns": [f"{a.kind}:{a.name}" for a in table.opp_actions],
                            "values": [[round(x, 4) for x in row] for row in table.values],
                        }
                except PayoffUnavailable as error:
                    self.unavailable[str(error)] += 1
                except Exception as error:
                    self.failures.append(f"{type(error).__name__}: {error}")
                if battle.opponent_active_pokemon:
                    if before != battle.opponent_active_pokemon.stats:
                        self.failures.append("Live opponent stats changed")
            return super().choose_move(battle)

    async def play():
        observer = Observer(battle_format=BATTLE_FORMAT, max_concurrent_battles=1)
        opponent = SimpleHeuristicsPlayer(battle_format=BATTLE_FORMAT, max_concurrent_battles=1)
        try:
            await asyncio.wait_for(observer.battle_against(opponent, n_battles=5), timeout=180)
            return observer
        finally:
            await observer.ps_client.stop_listening()
            await opponent.ps_client.stop_listening()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PayoffApproximationWarning)
        observer = asyncio.run(play())
    print(
        json.dumps(
            {
                "games": observer.n_finished_battles,
                "tables": observer.tables,
                "unavailable": dict(observer.unavailable),
                "failures": observer.failures,
                "sample": observer.sample,
            }
        )
    )
    assert observer.n_finished_battles == 5
    assert observer.tables > 0
    assert observer.sample is not None
    assert not observer.failures
