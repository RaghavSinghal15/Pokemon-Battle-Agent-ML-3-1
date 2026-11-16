"""Write labelled decision points from the shared battle Parquet file."""

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq

from pokeagent.config import INTERIM_DIR, PROCESSED_DIR, SPLITS_PATH
from pokeagent.parser.replay import ReplayParseError, parse_battle

_SCHEMA = pa.schema(
    [
        ("battle_id", pa.string()),
        ("turn", pa.int32()),
        ("perspective", pa.string()),
        ("state", pa.string()),
        ("action", pa.struct([("kind", pa.string()), ("name", pa.string()), ("tera", pa.bool_())])),
        ("won", pa.bool_()),
    ]
)


@dataclass
class ParseSummary:
    """Aggregate parser results without exposing usernames or raw logs."""

    battles: int = 0
    rows: int = 0
    errors: Counter[str] = field(default_factory=Counter)


def export_decision_points(
    battles_path: Path,
    output: Path,
    *,
    battle_ids: set[str] | None = None,
) -> ParseSummary:
    """Convert selected battles, skipping whole games with invalid or unsupported events."""
    if output.resolve() == battles_path.resolve():
        raise ValueError("Input and output must use different files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".parquet", delete=False) as temporary:
        temporary_path = Path(temporary.name)

    summary = ParseSummary()
    seen: set[str] = set()
    try:
        with (
            pq.ParquetFile(battles_path) as source,
            pq.ParquetWriter(temporary_path, _SCHEMA) as writer,
        ):
            columns = set(source.schema_arrow.names)
            if not {"battle_id", "log"} <= columns:
                raise ValueError("Battle Parquet must contain battle_id and log")
            selected = sorted(columns & {"battle_id", "log", "winner", "ratings"})
            for batch in source.iter_batches(batch_size=128, columns=selected):
                rows = []
                for battle in batch.to_pylist():
                    battle_id = battle["battle_id"]
                    if battle_ids is not None and battle_id not in battle_ids:
                        continue
                    if battle_id in seen:
                        summary.errors["duplicate battle"] += 1
                        continue
                    seen.add(battle_id)
                    summary.battles += 1
                    log = battle["log"]
                    if not isinstance(log, str) or not log:
                        summary.errors["missing log"] += 1
                        continue
                    ratings = battle.get("ratings") or {}
                    if isinstance(ratings, str):
                        ratings = json.loads(ratings)
                    try:
                        points = parse_battle(battle_id, log, battle.get("winner"), dict(ratings))
                    except ReplayParseError as exc:
                        summary.errors[exc.event] += 1
                        continue
                    if not points:
                        summary.errors["no observable choices"] += 1
                    for point in points:
                        rows.append(
                            {
                                "battle_id": battle_id,
                                "turn": point.state.turn,
                                "perspective": point.state.perspective,
                                "state": json.dumps(point.state.to_dict()),
                                "action": asdict(point.action),
                                "won": point.won,
                            }
                        )
                if rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=_SCHEMA))
                    summary.rows += len(rows)
            if battle_ids is not None and battle_ids - seen:
                raise ValueError(f"{len(battle_ids - seen)} selected battles are absent from input")
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    return summary


def main() -> None:
    """Run the parser over battle Parquet, optionally selecting a shared split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INTERIM_DIR / "battles.parquet")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR / "decision_points.parquet")
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--split", choices=("train", "val", "test"))
    args = parser.parse_args()
    battle_ids = None
    if args.split:
        battle_ids = set(json.loads(args.splits.read_text(encoding="utf-8"))[args.split])
    summary = export_decision_points(args.input, args.out, battle_ids=battle_ids)
    print(f"Wrote {summary.rows:,} rows from {summary.battles:,} battles to {args.out}")
    for reason, count in summary.errors.most_common():
        print(f"  {reason}: {count:,}")


if __name__ == "__main__":
    main()
