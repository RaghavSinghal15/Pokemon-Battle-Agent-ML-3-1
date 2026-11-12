"""Role E: opponent model. Pokémon knowledge needed: low.

Predicts the probability of each action class (interfaces.ACTION_CLASSES) the opponent will
choose next, from a FeatureRow. Version 1 compares multinomial logistic regression with
LightGBM (gradient-boosted decision trees), then calibrates the better one.

Produces:
    artifacts/opponent_model.pkl            the trained, calibrated model (git-ignored)
    reports/results/opponent_model.md       accuracy, macro-F1, log loss, calibration plots,
                                            feature importance, comparison of the two models
"""
