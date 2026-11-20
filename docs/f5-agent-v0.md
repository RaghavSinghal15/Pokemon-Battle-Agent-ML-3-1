# F5: first payoff-driven agent

The default `ExplainableAgent` builds F4's one-turn table and chooses the highest expected
payoff with F2. The initial forecast is one third each for attack, non-damaging move and
switching. F4 removes classes proved impossible and rescales the remainder. Move inclusion
weights come from D1's public set priors; these are not calibrated choice predictions.
F5 does not need a trained opponent model or the replay parser. It can also run F1's minimax
strategy with `strategy="lp"`; the opponent forecast does not influence that strategy.

Only legal supplied moves and switches become orders. New tera choices remain omitted.
Forced replacements choose the legal reserve with the greatest remaining HP fraction.
If a table cannot be built or solved, the fallback chooses the available move with greatest
estimated midpoint damage. If every estimate failed, it chooses the first legal move. With
no listed legal option it requests Showdown's default order. `last_decision`,
`decision_counts` and `fallback_reasons` expose which path was used. `explain=True` prints
the choice and reason. This short trace is distinct from F6's full public-state explanation.

The agent temporarily maps cosmetic forms to their public base-species set prior if D1
returns no moves (review feedback on PR #48). This changes only the agent's belief input,
not the live species or D's module. Revealed moves are still added by F4. Full Bayes updates,
public feature construction and learned forecasts belong to F6.

## Reproduce the development check

Start the local simulator and commit the source, then run:

```sh
python scripts/evaluate_agent_v0.py --server-dir ../pokemon-showdown
```

This plays 200 games each against random and max power, with the agent issuing the challenge.
It saves two A2 records and a summary with decision/fallback counts under
`reports/results/agent-v0/`. It refuses to replace existing outputs. The source and simulator
must have no tracked changes and the simulator checkout must also supply the public sets.
Seeds cover Python/NumPy action choices; server teams remain stochastic. These are development
games for F5 acceptance, not the frozen final 1,000-game evaluation or a trained-model result.
Passing means both Wilson lower bounds exceed 50%; no held-out human data is used or tuned on.

The PR is stacked on a temporary foundation combining reviewed F1–F4, D1–D2 (#48) and A1–A2.
Those source modules are inherited unchanged except for the F2/F4 review fixes. The foundation
exists to exercise dependencies without merging teammates' open PRs into `main`. Retarget the
F5 PR to `main` after those dependencies are accepted and merged.
