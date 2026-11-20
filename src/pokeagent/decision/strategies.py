"""Choosing an action from a payoff table."""

import numpy as np
from scipy.optimize import linprog

from pokeagent.config import SEED
from pokeagent.decision._validation import action_label, payoff_matrix
from pokeagent.interfaces import Decision, PayoffTable

# Seed once, not on every turn: repeated calls must draw different actions from the mix.
_RNG = np.random.default_rng(SEED)


def best_response(table: PayoffTable) -> Decision:
    """Pick the row with the highest expected payoff: sum over j of values[i][j] * opp_probs[j].

    This exploits the opponent model: it is strong if the predictions are right and can be
    punished if they are wrong.
    """
    raise NotImplementedError


def minimax_lp(table: PayoffTable, *, rng: np.random.Generator | None = None) -> Decision:
    """The mixed strategy that maximises our worst-case expected payoff.

    Linear program: choose probabilities x_i over our rows and a number v to
        maximise v
        subject to  sum_i x_i * values[i][j] >= v   for every opponent column j
                    sum_i x_i = 1,  x_i >= 0
    Solve with scipy.optimize.linprog (it minimises, so minimise -v), then sample the action
    from x. In rock-paper-scissors this gives 1/3 each.

    Opponent-model probabilities are deliberately ignored. The guarantee is for the mix's
    expected payoff within this table, not for the sampled action or the whole battle.
    Pass a generator seeded with config.SEED for a repeatable experiment; otherwise calls
    share a generator seeded once at module import.

    Raises ValueError for an empty, misaligned or non-finite table (or repeated action labels),
    and RuntimeError if the solver cannot find a strategy.
    """
    values = payoff_matrix(table)
    n_rows, n_cols = values.shape

    # Variables are [x_0, ..., x_(n_rows-1), v]. linprog minimises, so minimise -v.
    objective = np.zeros(n_rows + 1)
    objective[-1] = -1

    # Convert each payoff >= v constraint into linprog's <= form: -payoff + v <= 0.
    inequalities = np.column_stack((-values.T, np.ones(n_cols)))
    total_probability = np.ones((1, n_rows + 1))
    total_probability[0, -1] = 0
    result = linprog(
        objective,
        A_ub=inequalities,
        b_ub=np.zeros(n_cols),
        A_eq=total_probability,
        b_eq=[1.0],
        # v must be free: even the best guaranteed payoff can be negative.
        bounds=[(0, 1)] * n_rows + [(None, None)],
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Minimax linear program failed: {result.message}")

    # Remove tiny solver round-off errors before giving probabilities to the sampler.
    probabilities = np.clip(result.x[:-1], 0, 1)
    probabilities /= probabilities.sum()
    generator = _RNG if rng is None else rng
    chosen_index = int(generator.choice(n_rows, p=probabilities))
    mixture = {}
    for action, probability in zip(table.our_actions, probabilities, strict=True):
        mixture[action_label(action)] = float(probability)

    worst_case = float(np.min(probabilities @ values))
    chosen = table.our_actions[chosen_index]
    return Decision(
        chosen=chosen,
        strategy="lp",
        mixed_strategy=mixture,
        explanation=(
            f"Minimax mix has worst-case expected payoff {worst_case:+.3f} within this table. "
            f"Sampled {action_label(chosen)} with probability {probabilities[chosen_index]:.1%}."
        ),
    )
