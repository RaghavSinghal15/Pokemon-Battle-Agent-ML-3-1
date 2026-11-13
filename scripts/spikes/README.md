# Spikes

Quick experiments that answered the foundation questions in
[docs/08-foundation-checks.md](../../docs/08-foundation-checks.md). They are not part of the
`pokeagent` package and have no tests. When a module owner builds the real version, they may
copy pieces from here, but they write the module properly, with tests.

| Script | Answers | Needs |
|---|---|---|
| `collect_local_games.py` | Records bot games three ways (spectator log, each player's message stream, live snapshots) plus each bot's true choices | Local server |
| `replay_parse.py` | Helper: feed a public log into poke-env's `Battle` | — |
| `snapshot.py` | Helper: flatten a `Battle` into a dict for comparisons | — |
| `compare_views.py` | Checks 2 and 3: replay view vs live view | Output of `collect_local_games.py` |
| `label_visibility.py` | Check 4: how often the opponent's choice is visible, checked against true choices | Output of `collect_local_games.py` |
| `damage_calc_check.py` | Check 5: damage calculator with hidden information | Local server |
| `check_dataset.py` | Check 1: what the Hugging Face replay dataset actually contains | Internet (or `--local` for an offline test) |

Run them from the repo root, e.g. `python scripts/spikes/compare_views.py data/interim/local_games.jsonl`.
