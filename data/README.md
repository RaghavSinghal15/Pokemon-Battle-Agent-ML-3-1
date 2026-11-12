# data/ (never committed)

Everything in this folder except this README is git-ignored. Recreate it with the scripts
instead of sharing files:

| Folder | Made by | Contents |
|---|---|---|
| `raw/` | role B, `pokeagent.data.ingest` | untouched downloads from Hugging Face |
| `interim/battles.parquet` | role B | filtered battles, one row per battle |
| `splits.json` | role B | train / val / test battle ids (made once, never regenerated) |
| `processed/decision_points.parquet` | role C | one row per (battle, turn, perspective) |
| `processed/features.parquet` | role D | feature rows plus labels |

The replay dataset has no stated licence. Use it for this course only, and never upload or
redistribute it (no Drive links, no committing it, no pushing it anywhere).
