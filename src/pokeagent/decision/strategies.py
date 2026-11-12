"""Choosing an action from a payoff table."""

from pokeagent.interfaces import Decision, PayoffTable


def best_response(table: PayoffTable) -> Decision:
    """Pick the row with the highest expected payoff: sum over j of values[i][j] * opp_probs[j].

    This exploits the opponent model: it is strong if the predictions are right and can be
    punished if they are wrong.
    """
    raise NotImplementedError


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
