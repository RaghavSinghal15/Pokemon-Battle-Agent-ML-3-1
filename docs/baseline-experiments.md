# Baseline batch (A3)

With a local simulator running, commit the source changes, then run:

    python scripts/run_baselines.py --server-dir ../pokemon-showdown

The fixed batch compares all three unordered pairs among random, maximum-power and
heuristics bots: 1,000 games per pair, 500 in each challenge direction. It writes six
matchup records under `reports/results/baselines/` and a table at `reports/results/baselines.md`.
The report is written only after all six matchups pass validation. Partial runs leave their
records for inspection, not a completed report. Existing outputs are never overwritten;
choose fresh `--out-dir` and `--report` paths to run another batch.

`--games-per-pair` may be changed for development but must be positive and even. The default
remains the task's 1,000 games. `--seed` starts six consecutive Python/NumPy seeds. Each
orientation creates and closes its own clients. `--timeout` bounds each orientation's wait;
the same timeout caveat as [the runner](evaluation-runner.md) applies. Server-generated teams
are not controlled by this Python seed, so repeated batches are new random samples.

The aggregator rejects inconsistent counts, missing/duplicate directions, unequal direction
sizes, mixed formats, mixed source revisions and tracked source/server changes. Both bots'
win intervals include ties as non-wins; rates are not necessarily complements when ties exist.
The report contains all measured outcomes, without claiming significance or choosing a
winning agent variant. This is baseline-only evaluation and never touches human replay data,
training splits or model selection. F5/F6 and their success criteria remain separate work.
