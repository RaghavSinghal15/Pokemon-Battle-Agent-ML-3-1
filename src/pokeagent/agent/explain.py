"""The per-turn explanation log: why the agent chose what it chose."""

import math

from pokeagent.decision._validation import action_label, payoff_matrix
from pokeagent.interfaces import ACTION_CLASSES, BeliefState, Decision, PayoffTable, PublicState


def format_turn(
    state: PublicState,
    beliefs: BeliefState,
    class_probs: dict[str, float],
    table: PayoffTable,
    decision: Decision,
    win_probability: float | None = None,
) -> str:
    """Explain public beliefs, the forecast and the strategy's actual scoring criterion.

    Best response shows row expectations under the feasible column distribution. Minimax
    shows the row minima and the mixture's worst case, not a forecast-based guarantee.
    """
    values = payoff_matrix(table)
    if (
        set(class_probs) != set(ACTION_CLASSES)
        or any(not math.isfinite(p) or p < 0 for p in class_probs.values())
        or not math.isclose(sum(class_probs.values()), 1, abs_tol=1e-6)
    ):
        raise ValueError("Explanation requires a valid three-class forecast")
    me, opp = state.me.active, state.opp.active
    matchup = " vs ".join(
        f"{mon.species} ({mon.hp_fraction:.0%})" if mon else "no active Pokemon"
        for mon in (me, opp)
    )
    lines = [f"Battle {state.battle_id} | Turn {state.turn}: {matchup}"]
    belief = beliefs.get(opp.species) if opp else None
    if belief and belief.role_probs:
        roles = sorted(belief.role_probs.items(), key=lambda pair: (-pair[1], pair[0]))
        lines.append("Beliefs: " + ", ".join(f"{name} {p:.0%}" for name, p in roles))
    else:
        lines.append(
            "Beliefs: no supported role distribution; using available public move evidence."
        )
    lines.append(
        "Opponent forecast: " + ", ".join(f"{c} {class_probs[c]:.0%}" for c in ACTION_CLASSES)
    )
    if decision.strategy == "best_response":
        if table.opp_probs is None:
            raise ValueError("Best-response explanation needs the feasible column probabilities")
        scores = values @ table.opp_probs
        label = "Expected row payoffs after impossible-class renormalisation"
    elif decision.strategy == "lp":
        scores = values.min(axis=1)
        label = "Row worst cases (the LP can improve these by mixing)"
        lines.append("Minimax ignores the opponent forecast and protects against supplied columns.")
        lines.append(
            "Mixture: "
            + ", ".join(
                f"{name} {probability:.1%}"
                for name, probability in decision.mixed_strategy.items()
                if probability > 0
            )
        )
    else:
        raise ValueError("A table explanation requires best_response or lp")
    lines.append(
        label
        + ": "
        + " | ".join(
            f"{action_label(action)} {score:+.3f}"
            for action, score in zip(table.our_actions, scores, strict=True)
        )
    )
    lines.append(f"Chose {action_label(decision.chosen)}: {decision.explanation}")
    if win_probability is None:
        lines.append(
            "Utility is an approximate one-turn HP score plus KO bonus, not win probability."
        )
    else:
        lines.append(
            f"Payoffs are changes in our predicted win probability this turn (now"
            f" {win_probability:.0%}): +0.050 means 5 points more likely to win."
        )
    return "\n".join(lines)
