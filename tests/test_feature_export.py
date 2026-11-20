"""Exercise the replay-to-feature handoff using small Parquet fixtures."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_interfaces import make_state

from pokeagent.beliefs import tracker
from pokeagent.features.build import FEATURE_NAMES
from pokeagent.features.export import export_features
from pokeagent.interfaces import Action


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    path = tmp_path / "sets.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(tracker, "RANDOM_BATTLE_SETS_PATH", path)
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps({"train": ["battle-1"], "val": ["battle-2"], "test": ["battle-3"]}),
        encoding="utf-8",
    )
    return splits, tmp_path / "points.parquet", tmp_path / "features.parquet"


def _write_points(path, actions, battle_ids, *, as_json=False):
    records = []
    for battle_id, action in zip(battle_ids, actions, strict=True):
        state = make_state()
        state.battle_id = battle_id
        state.my_last_action = Action(kind="move", name="makeitrain")
        records.append(
            {
                "battle_id": battle_id,
                "turn": state.turn,
                "perspective": state.perspective,
                "state": json.dumps(state.to_dict()),
                "action": json.dumps(action) if as_json else action,
                "won": False,
            }
        )
    pq.write_table(pa.Table.from_pylist(records), path)


@pytest.mark.parametrize("as_json", [False, True])
def test_export_copies_splits_and_uses_observer_perspective(inputs, as_json):
    splits, points, output = inputs
    actions = [
        {"kind": "move", "name": "makeitrain"},
        {"kind": "move", "name": "nastyplot"},
        {"kind": "switch", "name": "corviknight"},
    ]
    _write_points(points, actions, ["battle-1", "battle-2", "battle-3"], as_json=as_json)
    assert export_features(points, splits, output) == 3
    table = pq.read_table(output)
    rows = table.to_pylist()
    assert table.column_names == [
        "battle_id",
        "perspective",
        "split",
        *FEATURE_NAMES,
        "action_class",
    ]
    assert [row["split"] for row in rows] == ["train", "val", "test"]
    assert [row["action_class"] for row in rows] == ["attack", "non_damaging", "switch"]
    assert rows[0]["perspective"] == "p1"
    assert rows[0]["opp_hp"] == 0.78
    assert rows[0]["my_hp"] == 1
    assert rows[0]["opp_last_attack"] == 1
    assert rows[0]["opp_last_non_damaging"] == 0
    # Changing the current choice changes the label only.
    assert [rows[0][name] for name in FEATURE_NAMES] == [rows[2][name] for name in FEATURE_NAMES]


def test_missing_split_keeps_existing_output(inputs):
    splits, points, output = inputs
    output.write_bytes(b"existing output")
    _write_points(points, [{"kind": "move", "name": "makeitrain"}], ["unassigned"])
    with pytest.raises(ValueError, match="no split assignment"):
        export_features(points, splits, output)
    assert output.read_bytes() == b"existing output"
    assert set(output.parent.glob("*.parquet")) == {points, output}


def test_duplicate_split_is_rejected(inputs):
    splits, points, output = inputs
    splits.write_text(
        json.dumps({"train": ["battle-1"], "val": [], "test": ["battle-1"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate split assignment"):
        export_features(points, splits, output)
    assert not output.exists()
