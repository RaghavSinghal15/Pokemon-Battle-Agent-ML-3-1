"""Role C: parser. Pokémon knowledge needed: medium (what log events mean, not strategy).

Turns a battle log into a PublicState at the start of every turn, plus the action each player
took that turn (a DecisionPoint). Also converts a live poke-env battle into the same
PublicState, so training and live play see identical inputs.

Produces:
    data/processed/decision_points.parquet   one row per (battle_id, turn, perspective) with the
                                             state as JSON, the action taken, and who won

The single most important property of this module is NO LEAKAGE: the state for turn t may only
contain what was visible in the log before turn t started. See docs/04-architecture.md.

Reference: Showdown's battle protocol, sim/SIM-PROTOCOL.md in the pokemon-showdown repo.
"""
