# data/ (never committed)

Everything in this folder except this README is git-ignored. Recreate it with the scripts
instead of sharing files:

| Folder | Made by | Contents |
|---|---|---|
| `raw/gen9randombattle.parquet` | role B, `pokeagent.data.ingest` | every downloaded Gen 9 Random Battle, unfiltered |
| `interim/battles.parquet` | role B | filtered battles, one row per battle |
| `splits.json` | role B | train / val / test battle ids (made once, never regenerated) |
| `processed/decision_points.parquet` | role C | one row per (battle, turn, perspective) |
| `processed/features.parquet` | role D | feature rows plus labels |

The replay dataset has no stated licence. Use it for this course only, and never upload or
redistribute it (no Drive links, no committing it, no pushing it anywhere).

## The battle table (`raw/gen9randombattle.parquet` and `interim/battles.parquet`)

One row per battle. Both files have the same columns:

| Column | Type | Meaning |
|---|---|---|
| `battle_id` | text | Replay id, e.g. `gen9randombattle-2123456789` |
| `upload_time` | whole number | Upload time in Unix seconds (seconds since 1 Jan 1970, UTC); used for the time-based split |
| `rating_p1`, `rating_p2` | whole number or missing | Each player's ladder rating at the start of the game; missing if unrated |
| `ratings` | JSON text | The same two ratings, `{"p1": 1534, "p2": null}`; read by role C's exporter |
| `winner` | `"p1"`, `"p2"` or missing | Missing for ties and unfinished games |
| `tie` | true/false | The game ended in a tie |
| `turns` | whole number | Turns played (the highest `|turn|N` line) |
| `log` | text | The full public battle log |

Source: the Gen 9 Random Battle files of `HolidayOugi/pokemon-showdown-replays` at the
revision pinned in `config.HF_REPLAY_REVISION`, keeping only rows whose `formatid` is exactly
`gen9randombattle`. Recreate everything (download, filter, split) with one command:

```bash
python -m pokeagent.data.build --max-battles 5000     # quick sample for development
```

It also writes `reports/results/data_summary.json` (counts and distributions only, no names).
`data/splits.json` is never overwritten unless you pass `--overwrite-splits`.
