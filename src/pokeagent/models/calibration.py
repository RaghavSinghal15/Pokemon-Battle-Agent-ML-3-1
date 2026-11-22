"""Calibration: do predicted probabilities match how often things actually happen?

If the model says "70% switch" on 100 different turns, the opponent should switch on about
70 of them. The decision engine multiplies payoffs by these probabilities, so an overconfident
model makes the agent take bad bets.

E4 finding (docs/decisions.md, 2026-11-22): the main miscalibration came from the label bias
(hidden choices are never switches), not from the model type. Training with a hidden class
fixes it; a separate Platt or isotonic step is not needed. These helpers measure calibration.
"""

import numpy as np


def reliability_table(probs: np.ndarray, happened: np.ndarray, bins: int = 10) -> list[dict]:
    """Group predictions into equal-sized bins; compare mean prediction with the observed rate.

    probs: predicted probability of one event per row; happened: 1 if it happened, else 0.
    """
    probs, happened = np.asarray(probs, dtype=float), np.asarray(happened, dtype=float)
    if probs.shape != happened.shape or probs.ndim != 1 or len(probs) < bins:
        raise ValueError("Need one prediction and one outcome per row, at least one per bin")
    order = np.argsort(probs, kind="stable")
    rows = []
    for part in np.array_split(order, bins):
        rows.append(
            {
                "mean_predicted": float(probs[part].mean()),
                "observed_rate": float(happened[part].mean()),
                "rows": len(part),
            }
        )
    return rows


def expected_calibration_error(table: list[dict]) -> float:
    """Row-weighted average gap between predicted and observed rates (0 = perfectly calibrated)."""
    total = sum(row["rows"] for row in table)
    return float(
        sum(row["rows"] * abs(row["mean_predicted"] - row["observed_rate"]) for row in table)
        / total
    )
