"""The shared data structures behave as every module expects."""

import json

from pokeagent.interfaces import (
    ACTION_CLASSES,
    Action,
    FieldView,
    PokemonView,
    PublicState,
    SideView,
)


def make_state() -> PublicState:
    """A small hand-made state. Copy this pattern to build fake inputs for your own tests."""
    me = SideView(
        player="p1",
        pokemon=[
            PokemonView(
                species="gholdengo",
                level=76,
                hp_fraction=0.78,
                active=True,
                revealed_moves=["makeitrain"],
            ),
        ],
    )
    opp = SideView(
        player="p2",
        pokemon=[
            PokemonView(species="garchomp", level=74, active=True, boosts={"atk": 2}),
            PokemonView(species="corviknight", level=80, fainted=True, hp_fraction=0.0),
        ],
        side_conditions={"stealthrock": 1},
    )
    return PublicState(
        battle_id="test-1",
        turn=7,
        perspective="p1",
        me=me,
        opp=opp,
        field=FieldView(weather="Sandstorm"),
        opp_last_action=Action(kind="move", name="swordsdance"),
    )


def test_round_trip_through_json():
    state = make_state()
    restored = PublicState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored == state


def test_side_helpers():
    opp = make_state().opp
    assert opp.active.species == "garchomp"
    assert opp.n_unrevealed == 4
    assert opp.n_remaining == 5


def test_action_classes_fixed():
    assert ACTION_CLASSES == ("attack", "non_damaging", "switch")
