"""Role D: features. Pokémon knowledge needed: high (choosing what matters).

Turns a PublicState plus a BeliefState into a fixed-length row of numbers (a FeatureRow) that
the models can learn from.

Produces:
    data/processed/features.parquet   one row per decision point: battle_id, turn, perspective,
                                      every feature in FEATURE_NAMES, and the action label
"""
