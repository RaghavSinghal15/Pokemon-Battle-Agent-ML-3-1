# Recorded local matchups (A2)

With the local Showdown server running:

    python scripts/run_tournament.py --p1 heuristics --p2 random --games 100 --server-dir ../pokemon-showdown

The command prints counts and saves an exclusive, timestamped JSON file under
`reports/results/matchups/`. `--out` selects an explicit path; an existing file is never
overwritten. `--timeout` bounds the whole matchup, and `--concurrent` controls parallel games.
`--seed` defaults to config.SEED. The CLI closes its clients even if a run fails.

The public `run_matchup` function accepts any two idle poke-env players in the same format.
It excludes earlier completed games, verifies both clients agree on the new outcomes, and
rejects partial completion. It does not close caller-owned players, so they can be reused.
Do not concurrently reuse those players or run multiple global-seed experiments in one process.
Timeout bounds the local wait, not the simulator's lifetime: a caller handling a timeout
must also clean up its clients and any server-side games. No successful result is saved.

Win rate is wins / all games. Ties are reported separately and count as non-wins in the
Wilson interval; this is not a half-credit score. Each record contains UTC time, elapsed
time, Python/poke-env versions, the code commit, whether tracked files were modified, and
the Python/NumPy seed. Commit source changes before formal experiments. Untracked/ignored
outputs do not mark source dirty; this does not prove untracked source files were absent.

The server's random teams are not seeded by this function. A recorded Python seed makes
client-side random choices traceable; it does not reproduce the entire battle stream.
`--server-dir` records the actual server checkout revision and tracked-change flag. Without
it, server provenance is null, not guessed. Summary records contain counts and provenance only.

## Save public battle logs (A5)

    python scripts/run_tournament.py --p1 agent --p2 heuristics --games 20 --save-replays

`agent` runs the flat-forecast best-response agent. The same option works with baseline bots.
`--save-replays` saves a timestamped JSONL file under `data/interim/agent_games/`.
`--replay-out PATH` enables capture and chooses the file explicitly. Each JSON line has a
`battle_id`, the first player's `player_side` and `outcome` (win, loss or tie), and a list
of `spectator_log` lines. This makes it possible to find losses and replay them through the
public-state parser. The command prints both output paths.

Python callers can also pass `save_replays=True` or `replay_path=Path(...)` to `run_matchup`
with their own agent and opponent. A separate spectator session collects only new, finished
rooms while the matchup is running. If a room is unavailable or its log is incomplete,
the capture fails before either output is written. Capture time is included in `--timeout`.
The logs may contain public player names and belong in the ignored data directory.
