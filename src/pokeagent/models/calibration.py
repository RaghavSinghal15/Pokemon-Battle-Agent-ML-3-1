"""Calibration: making predicted probabilities match how often things actually happen.

If the model says "70% switch" on 100 different turns, the opponent should switch on about
70 of them. The decision engine multiplies payoffs by these probabilities, so an overconfident
model makes the agent take bad bets.
"""

from pokeagent.models.opponent import OpponentModel


def calibrate(
    model: OpponentModel, val_features, val_labels, method: str = "isotonic"
) -> OpponentModel:
    """Fit a calibration map on the validation set ("sigmoid" = Platt scaling, or "isotonic").

    TODO (role E): compare reliability diagrams and log loss before and after calibration,
    on the TEST set (never on the data the calibrator was fitted on).
    """
    raise NotImplementedError
