"""Role E: opponent model. Pokémon knowledge needed: low.

Predicts the probability of each action class (interfaces.ACTION_CLASSES) the opponent will
choose next, from a FeatureRow. Version 1 compares multinomial logistic regression with
LightGBM (gradient-boosted decision trees), trained with a "hidden" class to remove the
label bias from hidden choices (E4), which also makes the switch forecast well calibrated.

Produces:
    artifacts/opponent_model.pkl            the chosen model (git-ignored)
    reports/results/opponent_model.md       accuracy, macro-F1, log loss, calibration plots,
                                            feature importance, comparison of the two models
"""
