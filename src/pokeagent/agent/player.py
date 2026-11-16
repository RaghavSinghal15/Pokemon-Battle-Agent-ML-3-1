"""Play Gen 9 singles using one-turn payoffs and explicit legal fallbacks."""

import warnings
from collections import Counter
from typing import Literal

import numpy as np
from poke_env.battle import AbstractBattle, Battle
from poke_env.data import GenData, to_id_str
from poke_env.player import BattleOrder, Player

from pokeagent.beliefs.tracker import prior_belief
from pokeagent.config import RANDOM_BATTLE_SETS_PATH, SEED
from pokeagent.decision.damage import estimate_move_damage
from pokeagent.decision.payoff import (
    PayoffApproximationWarning,
    PayoffUnavailable,
    build_payoff_table,
)
from pokeagent.decision.strategies import best_response, minimax_lp
from pokeagent.interfaces import ACTION_CLASSES, Action, BeliefState, Decision


def flat_beliefs(battle: Battle) -> BeliefState:
    """Use public set priors, without a classifier or hidden-information observations."""
    opponent = battle.opponent_active_pokemon
    if opponent is None:
        return {}
    belief = prior_belief(opponent.species)
    if not belief.move_probs:
        # Temporary consumer-side workaround for cosmetic forms in D1 / PR #48.
        entry = GenData.from_gen(battle.gen).pokedex.get(opponent.species, {})
        base = to_id_str(entry.get("baseSpecies", ""))
        if base and base != opponent.species:
            belief = prior_belief(base)
    return {opponent.species: belief}


class ExplainableAgent(Player):
    """F5: flat action-class forecast, public set priors and auditable fallbacks.

    Full model inputs and public-state explanations are the separate F6 step.
    Our legal moves/team can inform decisions; no peer's battle is ever read.
    """

    def __init__(
        self,
        *args,
        strategy: Literal["best_response", "lp"] = "best_response",
        explain: bool = False,
        seed: int = SEED,
        **kwargs,
    ):
        if strategy not in ("best_response", "lp"):
            raise ValueError("strategy must be best_response or lp")
        if not RANDOM_BATTLE_SETS_PATH.is_file():
            raise FileNotFoundError(f"Public Random Battle sets missing: {RANDOM_BATTLE_SETS_PATH}")
        self.rng = np.random.default_rng(seed)
        self.strategy, self.explain = strategy, explain
        self.decision_counts = Counter()
        self.fallback_reasons = Counter()
        self.last_decision: Decision | None = None
        super().__init__(*args, **kwargs)

    def _order(self, battle: Battle, action: Action) -> BattleOrder:
        candidates = battle.available_moves if action.kind == "move" else battle.available_switches
        for candidate in candidates:
            name = candidate.id if action.kind == "move" else candidate.species
            if name == action.name and not action.tera:
                return self.create_order(candidate)
        raise ValueError(f"Decision is not currently legal: {action}")

    def _fallback(self, battle: Battle, reason: str) -> BattleOrder:
        """Prefer estimated immediate damage; forced switches prefer remaining HP."""
        self.fallback_reasons[reason] += 1
        if battle.available_moves and not battle.force_switch:
            estimates = estimate_move_damage(battle)
            scored = [
                (estimates[m.id].expected_percent, m)
                for m in battle.available_moves
                if m.id in estimates and estimates[m.id].expected_percent is not None
            ]
            move = max(scored, key=lambda pair: pair[0])[1] if scored else battle.available_moves[0]
            action = Action("move", move.id)
            detail = "greedy damage" if scored else "first legal move; no damage estimate"
        else:
            switches = [
                p for p in battle.available_switches if p.max_hp > 0 and p.current_hp_fraction > 0
            ]
            if not switches:
                self.last_decision = None
                self.decision_counts["default"] += 1
                if self.explain:
                    print(f"Turn {battle.turn}: server default; {reason}; no listed legal options")
                return self.choose_default_move()
            replacement = max(switches, key=lambda p: p.current_hp_fraction)
            action, detail = Action("switch", replacement.species), "healthiest legal replacement"
        decision = Decision(action, "baseline", explanation=f"Fallback: {reason}; {detail}.")
        return self._record(battle, decision)

    def _record(self, battle: Battle, decision: Decision) -> BattleOrder:
        order = self._order(battle, decision.chosen)
        self.last_decision = decision
        self.decision_counts[decision.strategy] += 1
        if self.explain:
            print(f"Turn {battle.turn}: {decision.explanation}")
        return order

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose from the table, or record why a legal fallback was needed."""
        if not isinstance(battle, Battle) or battle.gen != 9:
            raise ValueError("ExplainableAgent supports Gen 9 singles only")
        if battle.force_switch:
            return self._fallback(battle, "forced replacement, outside the simultaneous table")
        try:
            beliefs = flat_beliefs(battle)
            forecast = dict.fromkeys(ACTION_CLASSES, 1 / len(ACTION_CLASSES))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PayoffApproximationWarning)
                table = build_payoff_table(battle, beliefs, forecast)
            decision = (
                best_response(table)
                if self.strategy == "best_response"
                else minimax_lp(table, rng=self.rng)
            )
            return self._record(battle, decision)
        except (PayoffUnavailable, ValueError, RuntimeError) as error:
            return self._fallback(battle, f"{type(error).__name__}: {error}")
