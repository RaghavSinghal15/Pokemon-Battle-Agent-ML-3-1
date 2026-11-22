"""Features for the value model (V1, #71): who is winning from this position?

The value model predicts P(win) for the player whose turn it is, from the start-of-turn public
state. It reuses the opponent-model features and adds whole-team position features, because the
opponent-model features look mostly at the two active Pokemon.

Orientation differs from the opponent-model table: there `opp_` is the side being predicted;
here `my_` is the side whose win we predict (the decision point's own perspective), and `opp_`
is its opponent. The label is that side's `won`, which is never a feature.

    python -m pokeagent.features.value      # writes data/processed/value_features.parquet
"""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq
from poke_env.data import GenData

from pokeagent.beliefs.tracker import beliefs_for
from pokeagent.config import (
    DECISION_POINTS_PATH,
    RANDOM_BATTLE_GENERATION,
    SPLITS_PATH,
    VALUE_FEATURES_PATH,
)
from pokeagent.features.build import FEATURE_NAMES, _speed, _type_matchup, build_features
from pokeagent.features.export import _split_by_battle
from pokeagent.interfaces import PokemonView, PublicState, SideView

POSITION_FEATURE_NAMES = [
    "my_team_hp",
    "opp_team_hp",
    "my_statused",
    "opp_statused",
    "my_faster_than",
    "opp_faster_than",
    "my_super_effective",
    "opp_super_effective",
]
VALUE_FEATURE_NAMES = [*FEATURE_NAMES, *POSITION_FEATURE_NAMES]
LABEL = "won"

_INPUT_COLUMNS = ["battle_id", "turn", "perspective", "state", "won"]
_OUTPUT_SCHEMA = pa.schema(
    [
        ("battle_id", pa.string()),
        ("perspective", pa.string()),
        ("split", pa.string()),
        *((name, pa.float64()) for name in VALUE_FEATURE_NAMES),
        (LABEL, pa.bool_()),
    ]
)


def _alive(side: SideView) -> list[PokemonView]:
    """Revealed team members that have not fainted."""
    return [p for p in side.pokemon if not p.fainted and p.hp_fraction > 0]


def _team_hp(side: SideView) -> float:
    """Total HP left, in Pokemon: unrevealed members have not been hit, so they count as full."""
    return sum(p.hp_fraction for p in _alive(side)) + side.n_unrevealed


def _faster_than(dex: GenData, active: PokemonView | None, others: list, trick_room: bool) -> int:
    """How many of the other side's revealed Pokemon our active one moves before."""
    mine = _speed(dex, active)
    count = 0
    for other in others:
        theirs = _speed(dex, other)
        if mine is None or theirs is None:
            continue
        if (mine < theirs) if trick_room else (mine > theirs):
            count += 1
    return count


def _super_effective(dex: GenData, active: PokemonView | None, others: list) -> int:
    """How many of the other side's revealed Pokemon our active one's own types hit 2x or more."""
    return sum(_type_matchup(dex, active, other) >= 2 for other in others)


def position_features(state: PublicState, dex: GenData) -> dict[str, float]:
    me, opp = state.me, state.opp
    trick_room = state.field.trick_room
    # Counts cover the other side's whole revealed team, active Pokemon included.
    values = (
        _team_hp(me),
        _team_hp(opp),
        sum(bool(p.status) for p in _alive(me)),
        sum(bool(p.status) for p in _alive(opp)),
        _faster_than(dex, me.active, _alive(opp), trick_room),
        _faster_than(dex, opp.active, _alive(me), trick_room),
        _super_effective(dex, me.active, _alive(opp)),
        _super_effective(dex, opp.active, _alive(me)),
    )
    return {name: float(v) for name, v in zip(POSITION_FEATURE_NAMES, values, strict=True)}


def build_value_features(state: PublicState, dex: GenData) -> dict[str, float]:
    """All value-model columns for the player `state.perspective`, before this turn's choices."""
    return {**build_features(state, beliefs_for(state)), **position_features(state, dex)}


def export_value_features(decision_points: Path, split_file: Path, output: Path) -> int:
    """Write one row per decision point, labelled with whether that player won."""
    if output.resolve() in {decision_points.resolve(), split_file.resolve()}:
        raise ValueError("The value table must use a different path from its inputs")
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
            for batch in source.iter_batches(columns=_INPUT_COLUMNS):
                rows = []
                for record in batch.to_pylist():
                    state = PublicState.from_dict(json.loads(record["state"]))
                    battle_id = record["battle_id"]
                    if battle_id not in splits:
                        raise ValueError(f"Battle {battle_id} has no split assignment")
                    if state.perspective != record["perspective"]:
                        raise ValueError(f"Perspective mismatch in {battle_id}")
                    rows.append(
                        {
                            "battle_id": battle_id,
                            "perspective": record["perspective"],
                            "split": splits[battle_id],
                            **build_value_features(state, dex),
                            LABEL: bool(record["won"]),
                        }
                    )
                writer.write_table(pa.Table.from_pylist(rows, schema=_OUTPUT_SCHEMA))
                count += len(rows)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decision-points", type=Path, default=DECISION_POINTS_PATH)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--out", type=Path, default=VALUE_FEATURES_PATH)
    args = parser.parse_args()
    count = export_value_features(args.decision_points, args.splits, args.out)
    print(f"Wrote {count:,} value rows to {args.out}")


if __name__ == "__main__":
    main()
