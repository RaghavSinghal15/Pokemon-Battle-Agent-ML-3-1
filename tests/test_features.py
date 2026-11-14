"""Test the feature schema and values on hand-built start-of-turn public states."""

import pytest
from test_interfaces import make_state

from pokeagent.features.build import FEATURE_NAMES, build_features
from pokeagent.interfaces import Action, SetBelief


def test_all_columns_with_public_state_and_move_beliefs():
    state = make_state()
    state.me.active.status = "brn"
    state.me.active.boosts = {"spa": 1, "def": -2, "accuracy": 3}
    state.me.side_conditions = {"Spikes": 2, "Reflect": 1}
    state.opp.active.revealed_item = "Choice Scarf"
    state.opp.active.revealed_moves = ["earthquake", "swordsdance"]
    state.opp.active.tera_type = "Water"
    state.opp.active.terastallized = True
    state.opp.tera_used = True
    state.field.trick_room = True
    beliefs = {
        "garchomp": SetBelief(
            "garchomp",
            move_probs={"swordsdance": 0.7, "dragondance": 0.4, "recover": 0.2, "earthquake": 1},
        )
    }
    row = build_features(state, beliefs)
    assert list(row) == FEATURE_NAMES
    assert all(isinstance(value, float) for value in row.values())
    assert row == {
        "turn": 7,
        "my_hp": 0.78,
        "opp_hp": 1,
        "my_remaining": 6,
        "opp_remaining": 5,
        "my_unrevealed": 5,
        "opp_unrevealed": 4,
        "my_boost_sum": -1,
        "opp_boost_sum": 2,
        "my_status": 1,
        "opp_status": 0,
        "my_paralyzed": 0,
        "opp_paralyzed": 0,
        "my_hazard_layers": 2,
        "opp_hazard_layers": 1,
        "my_tera_used": 0,
        "opp_tera_used": 1,
        "my_revealed_moves": 1,
        "opp_revealed_moves": 2,
        "my_type_matchup": 1,
        "opp_type_matchup": 2,
        "speed_advantage": 1,
        "trick_room": 1,
        "opp_choice_item_known": 1,
        "opp_best_setup_move_probability": 0.7,
        "opp_best_recovery_move_probability": 0.2,
        "opp_last_attack": 0,
        "opp_last_non_damaging": 1,
        "opp_last_switch": 0,
    }


@pytest.mark.parametrize(
    "action,flags",
    [
        (Action("move", "U-turn"), (1, 0, 0)),
        (Action("switch", "ditto"), (0, 0, 1)),
        (Action("move", "unmodelled"), (0, 0, 0)),
        (None, (0, 0, 0)),
    ],
)
def test_previous_action_flags(action, flags):
    state = make_state()
    state.opp_last_action = action
    row = build_features(state, {})
    assert (row["opp_last_attack"], row["opp_last_non_damaging"], row["opp_last_switch"]) == flags


def test_speed_status_and_removed_item():
    state = make_state()
    state.opp.active.status = "PAR"
    state.opp.active.revealed_item = "Choice Scarf"
    state.opp.active.item_removed = True
    row = build_features(state, {})
    assert row["opp_paralyzed"] == 1
    assert row["speed_advantage"] == 1
    assert row["opp_choice_item_known"] == 0


def test_missing_active_and_unknown_species_have_finite_defaults():
    state = make_state()
    state.me.active.species = "unknown"
    state.opp.active.active = False
    row = build_features(state, {})
    assert row["opp_hp"] == 0
    assert row["opp_type_matchup"] == row["my_type_matchup"] == 1
    assert row["speed_advantage"] == 0
    assert row["opp_best_recovery_move_probability"] == 0
