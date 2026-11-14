"""Export the model's feature table from role B splits and role C decision points."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq
from poke_env.data import GenData

from pokeagent.beliefs.tracker import beliefs_for
from pokeagent.config import (
    DECISION_POINTS_PATH,
    FEATURES_PATH,
    RANDOM_BATTLE_GENERATION,
    SPLITS_PATH,
)
from pokeagent.features.build import FEATURE_NAMES, _action_class, build_features
from pokeagent.interfaces import Action, PublicState

_INPUT_COLUMNS = ["battle_id", "turn", "perspective", "state", "action"]
_OUTPUT_SCHEMA = pa.schema(
    [
        ("battle_id", pa.string()),
        ("perspective", pa.string()),
        ("split", pa.string()),
        *((name, pa.float64()) for name in FEATURE_NAMES),
        ("action_class", pa.string()),
    ]
)


def _split_by_battle(path: Path) -> dict[str, str]:
    splits = json.loads(path.read_text(encoding="utf-8"))
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test in the split file")
    result = {}
    for split, battle_ids in splits.items():
        if not isinstance(battle_ids, list):
            raise ValueError(f"Expected a list of battle IDs for {split}")
        for battle_id in battle_ids:
            if battle_id in result:
                raise ValueError(f"Duplicate split assignment for {battle_id}")
            result[battle_id] = split
    return result


def _feature_record(record: dict, splits: dict[str, str], dex: GenData) -> dict:
    state = PublicState.from_dict(json.loads(record["state"]))
    battle_id = record["battle_id"]
    if (state.battle_id, state.turn, state.perspective) != (
        battle_id,
        record["turn"],
        record["perspective"],
    ):
        raise ValueError(f"Decision point metadata disagrees with its public state in {battle_id}")
    if battle_id not in splits:
        raise ValueError(f"Battle {battle_id} has no split assignment")

    # A decision point is recorded from the actor's side. The model predicts the other side,
    # so use the same snapshot from that actor's opponent's point of view.
    observer = replace(
        state,
        perspective=state.opp.player,
        me=state.opp,
        opp=state.me,
        my_last_action=state.opp_last_action,
        opp_last_action=state.my_last_action,
    )
    action = record["action"]
    if isinstance(action, str):
        action = json.loads(action)
    label = _action_class(dex, Action(**action)) if action else None
    if label is None:
        raise ValueError(f"Unlabelled or unknown action in {battle_id} on turn {state.turn}")
    return {
        "battle_id": battle_id,
        "perspective": state.perspective,
        "split": splits[battle_id],
        **build_features(observer, beliefs_for(observer)),
        "action_class": label,
    }


def export_features(decision_points: Path, split_file: Path, output: Path) -> int:
    """Write one feature row per decision point in batches and return the row count."""
    if output.resolve() in {decision_points.resolve(), split_file.resolve()}:
        raise ValueError("The feature table must use a different path from its inputs")
    splits = _split_by_battle(split_file)
    dex = GenData.from_gen(RANDOM_BATTLE_GENERATION)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as temp:
        temporary_path = Path(temp.name)

    count = 0
    try:
        with (
            pq.ParquetFile(decision_points) as source,
            pq.ParquetWriter(temporary_path, _OUTPUT_SCHEMA) as writer,
        ):
            missing = set(_INPUT_COLUMNS) - set(source.schema_arrow.names)
            if missing:
                raise ValueError(f"Missing decision point columns: {', '.join(sorted(missing))}")
            for batch in source.iter_batches(columns=_INPUT_COLUMNS):
                rows = [_feature_record(record, splits, dex) for record in batch.to_pylist()]
                writer.write_table(pa.Table.from_pylist(rows, schema=_OUTPUT_SCHEMA))
                count += len(rows)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return count


def main() -> None:
    """Read the shared decision points and splits and produce a feature Parquet file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-points", type=Path, default=DECISION_POINTS_PATH)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--out", type=Path, default=FEATURES_PATH)
    args = parser.parse_args()
    count = export_features(args.decision_points, args.splits, args.out)
    print(f"Wrote {count} feature rows to {args.out}")


if __name__ == "__main__":
    main()
