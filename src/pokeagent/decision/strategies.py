"""Choosing an action from a payoff table."""

from pokeagent.interfaces import Decision, PayoffTable


def best_response(table: PayoffTable) -> Decision:
    """Pick the row with the highest expected payoff: sum over j of values[i][j] * opp_probs[j].

    This exploits the opponent model: it is strong if the predictions are right and can be
    punished if they are wrong.

    Requires a finite, non-negative probability for every opponent column. Their sum must
    be within 1e-6 of one; accepted rounding error is normalised away. Missing probabilities
    are an error rather than an implicit uniform forecast. Exact ties pick the first row,
    so results are repeatable. Invalid tables or probabilities raise ValueError.
    """
    import numpy as np

    from pokeagent.decision._validation import action_label, payoff_matrix

    values = payoff_matrix(table)
    if table.opp_probs is None:
        raise ValueError("Best response requires opponent probabilities")
    probabilities = np.asarray(table.opp_probs, dtype=float)
    if probabilities.shape != (len(table.opp_actions),):
        raise ValueError("Provide one probability per opponent action")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Opponent probabilities must be finite and non-negative")
    total = float(probabilities.sum())
    if not np.isclose(total, 1.0, rtol=0, atol=1e-6):
        raise ValueError("Opponent probabilities must sum to one")
    probabilities = probabilities / total

    # A weighted average per row: higher-probability opponent actions matter more.
    expected_payoffs = values @ probabilities
    chosen_index = int(np.argmax(expected_payoffs))
    chosen = table.our_actions[chosen_index]
    mixture = {}
    for index, action in enumerate(table.our_actions):
        mixture[action_label(action)] = float(index == chosen_index)

    return Decision(
        chosen=chosen,
        strategy="best_response",
        mixed_strategy=mixture,
        explanation=(
            f"Best response chooses {action_label(chosen)} with expected payoff "
            f"{expected_payoffs[chosen_index]:+.3f} using the supplied opponent probabilities."
        ),
    )


def minimax_lp(table: PayoffTable) -> Decision:
    """The mixed strategy that maximises our worst-case expected payoff.

    Linear program: choose probabilities x_i over our rows and a number v to
        maximise v
        subject to  sum_i x_i * values[i][j] >= v   for every opponent column j
                    sum_i x_i = 1,  x_i >= 0
    Solve with scipy.optimize.linprog (it minimises, so minimise -v), then sample the action
    from x. In rock-paper-scissors this gives 1/3 each.

    TODO (role F): add a test on rock-paper-scissors and on a game with an obvious best row.
    """
    raise NotImplementedError
