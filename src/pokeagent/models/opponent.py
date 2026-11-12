"""Training and using the opponent model."""

import pandas as pd

from pokeagent.interfaces import FeatureRow


class OpponentModel:
    """Wraps a trained classifier so the rest of the system only sees probabilities."""

    def predict_proba(self, row: FeatureRow) -> dict[str, float]:
        """Probability of each action class, e.g. {"attack": 0.6, "non_damaging": 0.1,
        "switch": 0.3}. Must sum to 1."""
        raise NotImplementedError


def train_logistic_regression(train: pd.DataFrame, val: pd.DataFrame) -> OpponentModel:
    """Multinomial logistic regression (course topic). Standardise features first.

    TODO (role E): report accuracy, macro-F1 and log loss on validation. Compare with a naive
    baseline that always predicts the class frequencies from training; beating it is the
    minimum bar for M4.
    """
    raise NotImplementedError


def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame) -> OpponentModel:
    """Gradient-boosted trees (course topic: ensembles). Use validation for early stopping."""
    raise NotImplementedError
