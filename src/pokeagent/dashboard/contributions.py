"""Why the forecast said what it said: per-feature contributions from LightGBM (F13).

LightGBM can split each prediction into one contribution per feature (pred_contrib=True).
These are SHAP values (Shapley additive explanations): for one turn, the contributions of all
features plus a constant add up to the model's raw score for a class, so a positive
contribution pushed that class up and a negative one pushed it down. LightGBM computes them
exactly for trees (TreeSHAP), so the separate shap library is not needed.

Raw scores are log-odds-like numbers before the softmax, not probabilities.
"""

import numpy as np
import pandas as pd

from pokeagent.features.build import FEATURE_NAMES


def forecast_contributions(model, features: dict[str, float]) -> dict[str, dict[str, float]]:
    """class name -> {feature: contribution} for one turn. {} if the model is not LightGBM.

    Classes are the ones the model was trained on (with "hidden" for the hidden-class model).
    """
    booster = getattr(getattr(model, "estimator", None), "booster", None)
    if booster is None or not hasattr(booster, "predict"):
        return {}
    row = pd.DataFrame([[float(features[name]) for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    raw = np.asarray(booster.predict(row, pred_contrib=True))[0]
    width = len(FEATURE_NAMES) + 1  # one per feature plus the constant (the expected value)
    result = {}
    for index, name in enumerate(model.classes):
        block = raw[index * width : (index + 1) * width]
        pairs = zip(FEATURE_NAMES, block[:-1], strict=True)
        result[name] = {feature: float(v) for feature, v in pairs}
        result[name]["(baseline)"] = float(block[-1])
    return result


def top_contributions(contributions: dict[str, float], count: int = 8) -> list[tuple[str, float]]:
    """The features with the largest effect either way, baseline left out."""
    items = [(k, v) for k, v in contributions.items() if k != "(baseline)"]
    return sorted(items, key=lambda pair: -abs(pair[1]))[:count]
