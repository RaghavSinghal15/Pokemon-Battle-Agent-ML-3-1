"""Input checks and action labels shared by the two decision strategies."""

import numpy as np

from pokeagent.interfaces import Action, PayoffTable


def action_label(action: Action) -> str:
    """Keep moves, switches and tera versions distinct in the explanation's probabilities."""
    label = f"{action.kind}:{action.name}"
    if action.tera:
        label += ":tera"
    return label


def payoff_matrix(table: PayoffTable) -> np.ndarray:
    """Require one finite payoff per action pair, and distinct labels for our actions."""
    if not table.our_actions or not table.opp_actions:
        raise ValueError("A payoff table needs at least one action for each player")

    values = np.asarray(table.values, dtype=float)
    expected_shape = (len(table.our_actions), len(table.opp_actions))
    if values.shape != expected_shape:
        raise ValueError(f"Payoffs must have shape {expected_shape}; got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Payoffs must all be finite numbers")

    labels = [action_label(action) for action in table.our_actions]
    if len(set(labels)) != len(labels):
        raise ValueError("Our actions must have distinct labels")
    return values
