"""DL3 (#78): the full agent with DL2's GRU as its opponent model, in live games.

The GRU (dl/sequence.py) reads the game so far for the opponent: at each of their earlier
decision turns, that turn's features and what they did (attack, non_damaging, switch, or
hidden). A per-turn model needs only this turn's features, so the normal agent never keeps
that history. This agent rebuilds it every turn from the battle log poke-env has received,
with the same parser walk and the same feature code that made the training sequences, so
the GRU sees live inputs built exactly like its training inputs.

No leakage: the history holds finished turns only, and their labels are what the log showed
(the parser's rule); the current turn is added unlabelled as the last step.
"""

from dataclasses import replace

import pandas as pd

from pokeagent.agent.player import ExplainableAgent
from pokeagent.beliefs.tracker import beliefs_for
from pokeagent.features.build import build_features
from pokeagent.interfaces import ACTION_CLASSES
from pokeagent.models.labels import action_class
from pokeagent.models.opponent import HIDDEN, LABEL
from pokeagent.parser.replay import choices_so_far

_PLACEHOLDER = "attack"  # the current turn's label is unknown; it is never used as an input


def _observer(state):
    """The same snapshot seen by the actor's opponent, as features/export.py builds rows."""
    return replace(
        state,
        perspective=state.opp.player,
        me=state.opp,
        opp=state.me,
        my_last_action=state.opp_last_action,
        opp_last_action=state.my_last_action,
    )


def history_rows(log: str, battle_id: str, opponent_role: str) -> list[dict]:
    """One feature row per finished decision turn of the opponent, in turn order."""
    observed, hidden = choices_so_far(battle_id, log)
    rows = []
    for state, action in observed:
        if state.perspective == opponent_role:
            try:
                label = action_class(action)
            except ValueError:
                continue  # a move missing from the move data: training dropped these rows too
            observer = _observer(state)
            rows.append({**build_features(observer, beliefs_for(observer)), LABEL: label})
    for state in hidden:
        if state.perspective == opponent_role:
            observer = _observer(state)
            rows.append({**build_features(observer, beliefs_for(observer)), LABEL: HIDDEN})
    return sorted(rows, key=lambda row: row["turn"])


def sequence_forecast(model, history: list[dict], current: dict) -> dict[str, float]:
    """Run the GRU over the history plus the current turn; return the current turn's forecast."""
    rows = [*history, {**current, LABEL: _PLACEHOLDER}]
    table = pd.DataFrame(rows)
    table["battle_id"], table["perspective"] = "live", "opponent"
    probs = model.predict_proba_table(table)[len(rows) - 1]
    return {name: float(p) for name, p in zip(ACTION_CLASSES, probs, strict=True)}


class SequenceAgent(ExplainableAgent):
    """ExplainableAgent whose forecast comes from the GRU over the opponent's game so far."""

    def _forecast(self, battle, state, beliefs) -> dict[str, float]:
        log = "\n".join("|".join(message) for message in battle._replay_data)
        opponent_role = "p2" if battle.player_role == "p1" else "p1"
        history = history_rows(log, battle.battle_tag, opponent_role)
        # Drop a history row for the current turn, if the log already shows one.
        history = [row for row in history if row["turn"] < state.turn]
        return sequence_forecast(self.opponent_model, history, build_features(state, beliefs))
