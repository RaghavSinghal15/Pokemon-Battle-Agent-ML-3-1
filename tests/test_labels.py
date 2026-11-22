"""The action-class label rule (E1, #14): one rule, shared by features, model and payoff table."""

import pytest
from poke_env.data import GenData

from pokeagent.config import RANDOM_BATTLE_GENERATION
from pokeagent.features.build import _action_class
from pokeagent.interfaces import ACTION_CLASSES, Action
from pokeagent.models.labels import action_class, move_class


def move(name, tera=False):
    return Action("move", name, tera)


def test_switch_is_switch():
    assert action_class(Action("switch", "garchomp")) == "switch"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("earthquake", "attack"),
        ("U-turn", "attack"),  # damages, then the user switches out: the choice was a move
        ("voltswitch", "attack"),
        ("seismictoss", "attack"),  # fixed damage, still a damaging move
        ("struggle", "attack"),
        ("swordsdance", "non_damaging"),  # raises Attack, hits nothing
        ("partingshot", "non_damaging"),  # lowers stats, then the user switches out
        ("leechseed", "non_damaging"),  # drains each turn, but deals no direct damage
        ("painsplit", "non_damaging"),
        ("sleeptalk", "non_damaging"),  # we label the choice, not the move it calls
        ("stealthrock", "non_damaging"),
        ("recover", "non_damaging"),
    ],
)
def test_moves_follow_their_category(name, expected):
    assert action_class(move(name)) == expected


def test_display_names_and_ids_agree():
    assert move_class("Swords Dance") == move_class("swordsdance") == "non_damaging"


def test_terastallizing_does_not_change_the_label():
    assert action_class(move("earthquake", tera=True)) == action_class(move("earthquake"))


def test_unknown_move_is_an_error_not_a_guess():
    with pytest.raises(ValueError):
        move_class("notarealmove")


def test_labels_are_the_interface_classes():
    for action in (move("earthquake"), move("swordsdance"), Action("switch", "garchomp")):
        assert action_class(action) in ACTION_CLASSES


def test_feature_code_uses_the_same_rule():
    """D's feature table and E's model must never disagree about a label."""
    dex = GenData.from_gen(RANDOM_BATTLE_GENERATION)
    for action in (move("uturn"), move("partingshot"), Action("switch", "garchomp")):
        assert _action_class(dex, action) == action_class(action)
    assert _action_class(dex, None) is None
    assert _action_class(dex, move("notarealmove")) is None
