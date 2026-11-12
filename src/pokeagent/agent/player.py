"""The agent as a poke-env Player.

This is a "walking skeleton": it already plays complete games (by choosing randomly), so the
whole system can be run end to end from day one. Each TODO replaces one piece with the real
component as it becomes ready, and the agent keeps working after every step.
"""

from typing import Literal

from poke_env.battle import AbstractBattle
from poke_env.player import BattleOrder, Player


class ExplainableAgent(Player):
    def __init__(
        self,
        *args,
        strategy: Literal["best_response", "lp"] = "best_response",
        explain: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strategy = strategy
        self.explain = explain

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        # TODO (role F), one step at a time, keeping the agent playable after each:
        # 1. state = parser.live.state_from_battle(battle)
        # 2. beliefs = beliefs.tracker.beliefs_for(state)
        # 3. probs = opponent_model.predict_proba(features.build.build_features(state, beliefs))
        # 4. table = decision.payoff.build_payoff_table(battle, beliefs, probs)
        # 5. decision = best_response(table) or minimax_lp(table), depending on self.strategy
        # 6. if self.explain: print(agent.explain.format_turn(...))
        # 7. return self.create_order(...) for decision.chosen
        return self.choose_random_move(battle)
