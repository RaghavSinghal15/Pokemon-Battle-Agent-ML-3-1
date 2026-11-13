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
it, server provenance is null, not guessed. No player names, raw battle logs or replay data
are stored in these summary records. Game logs remain a separate A5 task.
