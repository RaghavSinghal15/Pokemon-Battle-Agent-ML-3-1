# Evaluation plan (draft: team to confirm, then freeze)

This plan is written **before** we have any results, and then frozen. If we choose the
opponents, the number of games or the success bar after seeing the numbers, it looks (and is)
like moving the goalposts. Examiners ask about this. Any later change goes in
[decisions.md](decisions.md) with the reason and the date.

## What we measure

**The agent (owner: A).** Win rate of our agent on the local server, format `gen9randombattle`,
against three fixed opponents from poke-env:

| Opponent | What it does | Games | Role in the report |
|---|---|---|---|
| `RandomPlayer` | Any legal action at random | 1,000 | Sanity check |
| `MaxBasePowerPlayer` | Always the move with the highest listed power | 1,000 | Weak baseline |
| `SimpleHeuristicsPlayer` | Hand-written rules: type matchups, switching, setup moves | 1,000 | **Headline opponent** |

Each win rate comes with a 95% **Wilson confidence interval**: the range the true win rate
almost certainly lies in, given the number of games. Wilson is the standard interval for a
proportion; unlike the simple "p ± 1.96 × standard error" formula, it behaves well near 0% and
100%.

**Why 1,000 games.** With 1,000 games, a win rate near 50% has an interval of about ±3 points.
Comparing two versions of the agent (for example, with and without beliefs), a difference
of about 4.5 points or more is distinguishable from luck. Fewer games would hide differences
we care about. Games are cheap: bots play several per second.

**The opponent model (owner: E).** On the **test split of human replays**, which is touched
once, at the end:

| Metric | What it tells us |
|---|---|
| Log loss (main metric) | How good the probabilities are, not just the top guess. Lower is better |
| Accuracy, macro-F1 | How often the top guess is right; macro-F1 weighs the three classes equally, so the rare class counts |
| Confusion matrix | Which classes get mixed up |
| Calibration: reliability diagram and expected calibration error | Whether "70%" really happens 70% of the time |
| Naive baseline | Always predict the training-set class frequencies. **Both models must beat it on log loss** |

## Ablations (owner: A, with F)

An **ablation** switches one component off and reruns, to measure what that component is
worth. 1,000 games each vs `SimpleHeuristicsPlayer`:

| Variant | What is switched off |
|---|---|
| Full agent, best response | Nothing (reference) |
| Full agent, minimax LP | Strategy: LP instead of best response |
| No opponent model | Opponent probabilities replaced by the naive class frequencies |
| No beliefs | Opponent's hidden set replaced by its most common Random Battle role |
| Damage only | No opponent model, no beliefs: just the highest calculated damage |

## Success criteria (fixed now)

| Level | Criterion |
|---|---|
| **Minimum (project is safe)** | Agent beats `RandomPlayer` and `MaxBasePowerPlayer`: lower end of the 95% interval above 50%. Both opponent models beat the naive baseline on test log loss |
| **Target** | Agent beats `SimpleHeuristicsPlayer`: lower end of the interval above 50% |
| **Evidence the ML matters** | The full agent beats "damage only" by a statistically clear margin (non-overlapping intervals, or a two-proportion test at 5%) |

Reference point from the foundation checks: a bot that plays the highest calculated damage
won 15 of 40 games against `SimpleHeuristicsPlayer` (interval 24% to 53%). So the target is a
real bar, not a formality.

## Rules

- The test split comes from `data/splits.json` only and is evaluated **once**, after all
  tuning on validation is finished.
- Every reported number is saved with the commit hash, the seed and the number of games
  (`eval/tournament.py` does this).
- Losses are kept as replays, and we analyse at least 10 of them in the report: where does
  the agent go wrong?
