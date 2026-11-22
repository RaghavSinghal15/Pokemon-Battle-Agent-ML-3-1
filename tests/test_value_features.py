"""Value-model table (V1, #71): position features and the export."""

import json
from dataclasses import replace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from poke_env.data import GenData
from test_interfaces import make_state

from pokeagent.features.value import (
    LABEL,
    VALUE_FEATURE_NAMES,
    export_value_features,
    position_features,
)
from pokeagent.interfaces import FieldView, PokemonView


@pytest.fixture(scope="module")
def dex():
    return GenData.from_gen(9)


def state_with_reserves():
    """Gholdengo (5 unrevealed teammates) against Garchomp, a poisoned Blissey and a fainted
    Corviknight (3 unrevealed)."""
    state = make_state()
    state.opp.pokemon.append(PokemonView("blissey", 88, hp_fraction=0.5, status="tox"))
    return state


def test_team_hp_and_status_counts(dex):
    values = position_features(state_with_reserves(), dex)
    assert values["my_team_hp"] == pytest.approx(0.78 + 5)  # unrevealed count as full HP
    assert values["opp_team_hp"] == pytest.approx(1 + 0.5 + 3)  # fainted Corviknight adds 0
    assert (values["my_statused"], values["opp_statused"]) == (0, 1)


def test_speed_counts_and_trick_room(dex):
    state = state_with_reserves()
    values = position_features(state, dex)
    # Gholdengo (172 speed at level 76) outspeeds Blissey (147) but not Garchomp (194).
    assert (values["my_faster_than"], values["opp_faster_than"]) == (1, 1)
    trick_room = position_features(replace(state, field=FieldView(trick_room=True)), dex)
    assert (trick_room["my_faster_than"], trick_room["opp_faster_than"]) == (1, 0)


def test_super_effective_counts(dex):
    values = position_features(state_with_reserves(), dex)
    # Steel/Ghost hits neither Garchomp nor Blissey super effectively; Ground hits Gholdengo.
    assert (values["my_super_effective"], values["opp_super_effective"]) == (0, 1)


def test_export_labels_each_side_with_its_own_win_and_keeps_splits(tmp_path):
    state = make_state()
    mirrored = replace(state, perspective="p2", me=state.opp, opp=state.me)
    rows = [
        {
            "battle_id": "test-1",
            "turn": 7,
            "perspective": "p1",
            "state": json.dumps(state.to_dict()),
            "won": True,
        },
        {
            "battle_id": "test-1",
            "turn": 7,
            "perspective": "p2",
            "state": json.dumps(mirrored.to_dict()),
            "won": False,
        },
    ]
    points, splits, out = (
        tmp_path / "points.parquet",
        tmp_path / "splits.json",
        tmp_path / "v.parquet",
    )
    pq.write_table(pa.Table.from_pylist(rows), points)
    splits.write_text(json.dumps({"train": [], "val": ["test-1"], "test": []}))
    assert export_value_features(points, splits, out) == 2
    table = pd.read_parquet(out)
    assert list(table["split"]) == ["val", "val"] and list(table[LABEL]) == [True, False]
    assert LABEL not in VALUE_FEATURE_NAMES
    # Each row is from its own side: p1's my_team_hp is p2's opp_team_hp.
    assert table.loc[0, "my_team_hp"] == table.loc[1, "opp_team_hp"]
    assert table.loc[0, "my_hp"] == pytest.approx(0.78)
