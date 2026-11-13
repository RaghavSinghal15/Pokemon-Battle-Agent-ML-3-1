# Baseline bot comparisons (A3)

Format: `gen9randombattle`. Evaluator source: `c04028799cc722de07bbf9f312b41765d3e7d29a`. Showdown: `5c28e341ef9f9730a35874390cc3bc2aead7a16e`.

Each pair uses equal numbers in both challenge directions. This is baseline-only
evaluation; no agent, model or human-replay test set is evaluated here.

| First bot | Second bot | Games | First wins | Second wins | Ties | First win rate (95% Wilson) | Second win rate (95% Wilson) |
|---|---|---:|---:|---:|---:|---|---|
| heuristics | max_power | 1000 | 870 | 130 | 0 | 87.0% [84.8%, 88.9%] | 13.0% [11.1%, 15.2%] |
| heuristics | random | 1000 | 986 | 14 | 0 | 98.6% [97.7%, 99.2%] | 1.4% [0.8%, 2.3%] |
| max_power | random | 1000 | 917 | 83 | 0 | 91.7% [89.8%, 93.3%] | 8.3% [6.7%, 10.2%] |

Win rate means wins / all games. Ties are non-wins for both bots, not half-credit.
Wilson intervals use z=1.96 and describe sampling uncertainty, not a guarantee.
No claim of statistically significant differences or integrated-agent strength is made.

Python/NumPy seeds, UTC timestamps, package versions and durations are in the records.
The server's random teams are not seeded by the Python seed, so replaying this command
will produce another random sample. Source flags describe tracked changes only.

## Recorded matchups

- [RandomPlayer vs MaxBasePowerPlayer, 500 games, seed 464](baselines/random-vs-max_power.json)
- [MaxBasePowerPlayer vs RandomPlayer, 500 games, seed 465](baselines/max_power-vs-random.json)
- [RandomPlayer vs SimpleHeuristicsPlayer, 500 games, seed 466](baselines/random-vs-heuristics.json)
- [SimpleHeuristicsPlayer vs RandomPlayer, 500 games, seed 467](baselines/heuristics-vs-random.json)
- [MaxBasePowerPlayer vs SimpleHeuristicsPlayer, 500 games, seed 468](baselines/max_power-vs-heuristics.json)
- [SimpleHeuristicsPlayer vs MaxBasePowerPlayer, 500 games, seed 469](baselines/heuristics-vs-max_power.json)
