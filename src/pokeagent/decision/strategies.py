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
