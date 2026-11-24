"""Play Gen 9 singles using one-turn payoffs and explicit legal fallbacks."""

import json
import time
import warnings
from collections import Counter
from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np
from poke_env.battle import AbstractBattle, Battle
from poke_env.player import BattleOrder, Player

from pokeagent.agent.explain import format_turn, turn_record
from pokeagent.beliefs.tracker import beliefs_for, prior_belief
from pokeagent.config import (
    PAYOFF_SEARCH,
    PAYOFF_STALL_EPSILON,
    PAYOFF_STALL_TURNS,
    PAYOFF_STAND_INS_FOR,
    PAYOFF_UNSEEN_SAMPLES,
    RANDOM_BATTLE_SETS_PATH,
    SEED,
)
from pokeagent.decision._validation import action_label
from pokeagent.decision.damage import estimate_move_damage
from pokeagent.decision.payoff import (
    PayoffApproximationWarning,
    PayoffUnavailable,
    build_payoff_table,
    replacement_tables,
)
from pokeagent.decision.strategies import best_response, minimax_lp
from pokeagent.decision.value_utility import ValueScorer
from pokeagent.features.build import build_features
from pokeagent.interfaces import ACTION_CLASSES, Action, BeliefState, Decision, PayoffTable
from pokeagent.parser.live import state_from_battle

if TYPE_CHECKING:
    from pokeagent.models.opponent import OpponentModel


def flat_beliefs(battle: Battle) -> BeliefState:
    """Use public set priors, without a classifier or hidden-information observations."""
    opponent = battle.opponent_active_pokemon
    if opponent is None:
        return {}
    return {opponent.species: prior_belief(opponent.species)}


class ExplainableAgent(Player):
    """Flat development mode or an explicitly supplied learned opponent model.

    Model features come only from the shared public parser and belief tracker. Our legal
    moves/team can inform the payoff table; no peer's battle is ever read.
    """

    def __init__(
        self,
        *args,
        strategy: Literal["best_response", "lp"] = "best_response",
        explain: bool = False,
        seed: int = SEED,
        mode: Literal["flat", "model"] = "flat",
        opponent_model: "OpponentModel | None" = None,
        explanation_stream: TextIO | None = None,
        utility: Literal["hp", "value"] = "hp",
        value_model=None,
        belief_updates: bool = True,
        exact_move_model=None,
        search: bool = PAYOFF_SEARCH,
        trace_stream: TextIO | None = None,
        **kwargs,
    ):
        if strategy not in ("best_response", "lp"):
            raise ValueError("strategy must be best_response or lp")
        if mode not in ("flat", "model") or (mode == "model") != (opponent_model is not None):
            raise ValueError(
                "mode=model requires an opponent_model; flat mode must not receive one"
            )
        if utility not in ("hp", "value") or (utility == "value") != (value_model is not None):
            raise ValueError("utility=value requires a value_model; utility=hp must not get one")
        if not RANDOM_BATTLE_SETS_PATH.is_file():
            raise FileNotFoundError(f"Public Random Battle sets missing: {RANDOM_BATTLE_SETS_PATH}")
        self.rng = np.random.default_rng(seed)
        self.strategy, self.explain = strategy, explain
        self.mode, self.opponent_model = mode, opponent_model
        # V4: score cells by change in predicted P(win) (value) or HP traded plus a KO bonus (hp).
        self.utility, self.value_model = utility, value_model
        # Ablation (A6): False keeps every opponent belief at its Random Battle prior.
        self.belief_updates = belief_updates
        # E8: split each action class's probability among moves by the exact-move model (E6).
        self.exact_move_model = exact_move_model
        # F12: two-turn search in the payoff table; decision_seconds records the cost.
        self.search = search
        self.decision_seconds: list[float] = []
        self.explanation_stream = explanation_stream
        # F13 dashboard: one JSON line per explained turn (agent/explain.py: turn_record).
        self.trace_stream = trace_stream
        self.last_explanation = ""
        self.decision_counts = Counter()
        self.fallback_reasons = Counter()
        self.last_decision: Decision | None = None
        # #121: battle tag -> (HP snapshot last turn, turns in a row with no progress).
        self._stall: dict[str, tuple[tuple, int]] = {}
        super().__init__(*args, **kwargs)

    def _order(self, battle: Battle, action: Action) -> BattleOrder:
        candidates = battle.available_moves if action.kind == "move" else battle.available_switches
        for candidate in candidates:
            name = candidate.id if action.kind == "move" else candidate.species
            if name == action.name and (not action.tera or battle.can_tera):
                return self.create_order(candidate, terastallize=action.tera)
        raise ValueError(f"Decision is not currently legal: {action}")

    def _fallback(self, battle: Battle, reason: str) -> BattleOrder:
        """Prefer estimated immediate damage; unscorable forced switches prefer remaining HP."""
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
                self._emit(
                    f"Battle {battle.battle_tag} | Turn {battle.turn}: "
                    f"server default; {reason}; no listed legal options"
                )
                return self.choose_default_move()
            replacement = max(switches, key=lambda p: p.current_hp_fraction)
            action, detail = Action("switch", replacement.species), "healthiest legal replacement"
        decision = Decision(action, "baseline", explanation=f"Fallback: {reason}; {detail}.")
        return self._record(battle, decision)

    def _emit(self, text: str) -> None:
        self.last_explanation = text
        if self.explain:
            print(text)
        if self.explanation_stream is not None:
            self.explanation_stream.write(text + "\n\n")
            self.explanation_stream.flush()

    def _record(self, battle: Battle, decision: Decision, context=None) -> BattleOrder:
        order = self._order(battle, decision.chosen)
        self.last_decision = decision
        self.decision_counts[decision.strategy] += 1
        if context is not None:
            text = format_turn(*context)
            if self.trace_stream is not None:
                state, beliefs = context[0], context[1]
                record = turn_record(*context, features=build_features(state, beliefs))
                self.trace_stream.write(json.dumps(record) + "\n")
                self.trace_stream.flush()
        else:
            text = (
                f"Battle {battle.battle_tag} | Turn {battle.turn}: "
                f"{decision.chosen.kind}:{decision.chosen.name}; {decision.explanation}"
            )
        self._emit(text)
        return order

    def _table_score(self, table: PayoffTable) -> float:
        """How good a table is for us under our own strategy.

        Best response: the expected payoff of its best row. Minimax: the game value, our
        guaranteed expected payoff when the opponent answers our mix as badly as possible.
        """
        values = np.asarray(table.values, dtype=float)
        if self.strategy == "best_response":
            return float(np.max(values @ np.asarray(table.opp_probs, dtype=float)))
        decision = minimax_lp(table, rng=self.rng)
        mix = np.array(
            [decision.mixed_strategy.get(action_label(a), 0.0) for a in table.our_actions]
        )
        return float(np.min(mix @ values))

    def _choose_replacement(self, battle: Battle) -> BattleOrder:
        """Send in the reserve whose next-turn table scores best; HP decides only ties.

        The forecast is flat in both modes: the opponent model was trained on states with
        our real active Pokemon, not on hypothetical replacements.
        """
        beliefs = flat_beliefs(battle)
        flat = dict.fromkeys(ACTION_CLASSES, 1 / len(ACTION_CLASSES))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PayoffApproximationWarning)
            tables = replacement_tables(battle, beliefs, flat)
        hp = {mon.species: mon.current_hp_fraction for mon in battle.available_switches}
        scores = {species: self._table_score(table) for species, table in tables.items()}
        ranked = sorted(scores, key=lambda species: (scores[species], hp[species]), reverse=True)
        summary = ", ".join(f"{species} {scores[species]:+.3f}" for species in ranked)
        decision = Decision(
            Action("switch", ranked[0]),
            self.strategy,
            explanation=(
                f"Forced replacement by matchup: {ranked[0]} scores best next turn ({summary})."
            ),
        )
        return self._record(battle, decision)

    def _forecast(self, battle: Battle, state, beliefs) -> dict[str, float]:
        """The opponent model's class probabilities for this turn.

        A per-turn model needs only this turn's features. A subclass can override this to use
        more of the game (dl.sequence_agent does, for the GRU that reads the whole game).
        """
        return self.opponent_model.predict_proba(build_features(state, beliefs))

    def _stall_guard(self, battle: Battle, table: PayoffTable, decision: Decision) -> Decision:
        """#121: break a stall the one-turn table cannot see out of.

        A stall: several turns in a row where nobody's HP changed and every row of the table
        scores (about) zero, e.g. boosting forever with no attack left that can hurt them.
        The table looks one turn ahead, so it cannot see that switching pays off later. After
        PAYOFF_STALL_TURNS such turns, take the best-scoring switch instead.
        """
        snapshot = tuple(
            round(mon.current_hp_fraction, 3)
            for team in (battle.team, battle.opponent_team)
            for mon in team.values()
        )
        probs = np.asarray(table.opp_probs or [], dtype=float)
        values = np.asarray(table.values, dtype=float)
        expected = values @ probs if len(probs) == values.shape[1] else values.max(axis=1)
        flat = bool(np.all(np.abs(expected) < PAYOFF_STALL_EPSILON))
        previous, turns = self._stall.get(battle.battle_tag, (None, 0))
        turns = turns + 1 if flat and snapshot == previous else 0
        self._stall[battle.battle_tag] = (snapshot, turns)
        switches = [i for i, a in enumerate(table.our_actions) if a.kind == "switch"]
        if turns < PAYOFF_STALL_TURNS or decision.chosen.kind == "switch" or not switches:
            return decision
        best = max(switches, key=lambda i: expected[i])  # ties: first, so it is repeatable
        chosen = table.our_actions[best]
        self._stall[battle.battle_tag] = (snapshot, 0)
        return Decision(
            chosen,
            decision.strategy,
            mixed_strategy={action_label(a): float(a == chosen) for a in table.our_actions},
            explanation=(
                f"{decision.explanation} Stall guard: {turns} turns with no HP change and every"
                f" option scoring about zero, so switching to {action_label(chosen)} to change"
                " the matchup."
            ),
        )

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """Choose from the table, or record why a legal fallback was needed."""
        start = time.perf_counter()
        try:
            return self._choose_move(battle)
        finally:
            self.decision_seconds.append(time.perf_counter() - start)

    def _choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if not isinstance(battle, Battle) or battle.gen != 9:
            raise ValueError("ExplainableAgent supports Gen 9 singles only")
        if battle.force_switch:
            try:
                return self._choose_replacement(battle)
            except (PayoffUnavailable, ValueError, RuntimeError) as error:
                return self._fallback(
                    battle,
                    f"forced replacement could not be scored: {type(error).__name__}: {error}",
                )
        try:
            state = None
            if self.mode == "model":
                state = state_from_battle(battle)
                if self.belief_updates:
                    beliefs = beliefs_for(state)
                else:
                    beliefs = {p.species: prior_belief(p.species) for p in state.opp.pokemon}
                forecast = self._forecast(battle, state, beliefs)
            else:
                beliefs = flat_beliefs(battle)
                forecast = dict.fromkeys(ACTION_CLASSES, 1 / len(ACTION_CLASSES))
            scorer = None
            if self.utility == "value":
                if state is None:
                    state = state_from_battle(battle)
                scorer = ValueScorer(self.value_model, state)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", PayoffApproximationWarning)
                extra = {"scorer": scorer} if scorer is not None else {}
                if self.strategy in PAYOFF_STAND_INS_FOR:
                    extra["unseen_samples"] = PAYOFF_UNSEEN_SAMPLES
                if self.exact_move_model is not None and state is not None:
                    extra["move_weights"] = self.exact_move_model.move_weights(state)
                if self.search:
                    extra["search"] = True
                table = build_payoff_table(battle, beliefs, forecast, **extra)
            decision = (
                best_response(table)
                if self.strategy == "best_response"
                else minimax_lp(table, rng=self.rng)
            )
            decision = self._stall_guard(battle, table, decision)
            context = None
            if self.mode == "model":
                now = scorer.win_probability() if scorer is not None else None
                context = (state, beliefs, forecast, table, decision, now)
            return self._record(battle, decision, context)
        except (PayoffUnavailable, ValueError, RuntimeError) as error:
            return self._fallback(battle, f"{type(error).__name__}: {error}")
