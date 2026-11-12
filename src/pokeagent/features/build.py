"""PublicState + BeliefState -> FeatureRow."""

from pokeagent.interfaces import BeliefState, FeatureRow, PublicState

# The fixed, ordered list of feature names. Every FeatureRow has exactly these keys.
# TODO (role D): fill in. Starting ideas, each a number:
#   my_hp, opp_hp                       active Pokémon's remaining health (0 to 1)
#   type_eff_mine_on_opp, ..._opp_on_me best type-effectiveness multiplier each way (0, 0.25..4)
#   speed_advantage                     who is probably faster (+1, 0, -1)
#   my_remaining, opp_remaining         Pokémon left on each side
#   my_boost_sum, opp_boost_sum         total stat boosts
#   my_status, opp_status               one-hot status conditions
#   hazards_on_my_side, ...             entry hazards present
#   opp_p_has_setup_move, opp_p_has_recovery, opp_p_choice_item   from beliefs
#   opp_last_action_class               what they did last turn (one-hot)
#   turn
FEATURE_NAMES: list[str] = []


def build_features(state: PublicState, beliefs: BeliefState) -> FeatureRow:
    """Compute every feature in FEATURE_NAMES for one decision point.

    Rules:
    - Use only `state` and `beliefs`. Never the action taken or the battle result.
    - Every feature must be explainable in one sentence in docs/features.md (viva requirement).
    """
    raise NotImplementedError
