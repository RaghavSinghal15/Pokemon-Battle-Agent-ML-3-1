# Decision log

Every design choice that someone might ask "why?" about, newest first. One entry per decision:
what we chose, what else we considered, and why. Keep entries short. These become the
"Design choices" section of the report and the answers to viva questions.

Template:

```
## YYYY-MM-DD: Short title (role, author)
**Decision:** what we chose.
**Alternatives:** what else we considered.
**Why:** the reason, with evidence (a number, a test, a plot) where possible.
```

---

## 2026-11-13: Minimax probabilities and repeatable sampling (F1, Devam)
**Decision:** maximise the worst-case expected payoff with `scipy.optimize.linprog`, then
sample an action from the resulting probabilities. The value variable can be negative.
The default random generator uses `config.SEED` once; experiments can pass their own seeded
generator. Probability labels include action kind, name and tera to distinguish legal options.
**Alternatives:** always take the most probable action; restart the seed every turn; constrain
the guaranteed value to be non-negative; use the opponent model's probabilities in the LP.
**Why:** predictable choices lose the protection of a mixed strategy. Negative-value games
still need a legal solution. Minimax protects against every supplied opponent column, so it
does not use a forecast. Tests cover uniform and non-uniform games, negative payoffs and
repeatable sequences of varied actions. This guarantee applies only to the table's expected
payoff, not to one sampled action or to the full battle.

## 2026-11-14 · Use the same 29 public features offline and live (D)

**Decision.** Use the ordered schema in [features.md](features.md). Include rough type and
speed comparisons, public status and team counts, the previous opponent action, and two
move probabilities from the belief tracker.

**Alternatives.** Encode species and moves as categories or run the damage calculator as a
feature.

**Why.** These numbers have the same meaning across species and can be computed before
either player acts. Type and speed summarize the matchup cheaply, with their assumptions
documented for review. Keeping the current action out of the feature builder avoids
exposing the model's training label.

## 2026-11-14 · Approximate hidden sets from Showdown's set file (D)

**Decision.** Give each listed set template equal weight. Assume equally likely move combinations
within each template and uniform ability and Tera type choices. Keep hidden items unknown.
Recompute beliefs from cumulative public observations on each turn.

**Alternatives.** Use empirical Random Battle usage rates or reproduce the generator's full
move, team and item rules.

**Why.** The local set file gives a consistent starting point for replay and live features. The
assumptions let us calculate the chance of seeing a combination of moves and show exactly how
it changes the estimate. Showdown's generator can prefer some combinations. Older replays and
effects like Transform can also disagree with the set file. If every template is ruled out,
keep the role prior and the observed details. See [Showdown's set file and generator][gen9].

[gen9]: https://github.com/smogon/pokemon-showdown/tree/master/data/random-battles/gen9

## 2026-11-14 · Snapshot at the turn boundary and attach labels afterward (C)

**Decision.** Rebuild a public battle with poke-env and snapshot both sides at each turn
marker. Read observable choices from the following events. Rebuild the live view with
the same parser, ignoring private requests and normalizing HP to the spectator percentage.

**Alternatives.** Use the live player's internal team state or attach labels while building
the state for the same turn.

**Why.** The public log supplies the same inputs in training and live play. Taking the
snapshot before reading the choice also prevents the move, revealed items and damage
from entering their own training row. Drop turns with hidden choices and keep the
observed disguise when the log shows a voluntary switch under Illusion.
## 2026-11-18: Time-ordered train / validation / test split (B, Rishi)
**Decision:** sort battles by upload time; the oldest 70% are training, the next 15%
validation, the newest 15% test. Whole battles only, never single turns. Ties in time are
broken by battle id so the split is the same on every computer. The file is written once and
`save_splits` refuses to overwrite it.
**Alternatives:** random split (lets the model learn from games played after some test games
and hides changes in play over time); random validation drawn from the older 85% (the earlier
plan in `splits.py`: tuning would then be judged on the same period as training, so it would
look better than it will on the newer test set); splitting by turn (turns of one game are
nearly identical, so the test set would contain near-copies of training rows).
**Why:** the agent plays future games, so the test set must be "the future". Putting
validation between train and test in time means tuning and calibration are judged the way the
test set will judge them.

## 2026-11-18: Which battles we keep (B, Rishi)
**Decision:** drop duplicate ids, games without a winner (ties, unfinished), games shorter than
2 turns, and games where either player is unrated or below a rating threshold. The threshold
is chosen from the data: the rating that keeps the top 35% of usable games, ranked by the
lower-rated player (`RATING_KEEP_FRACTION` in config). On the 50,000 newest battles this is
**1,893**, keeping 16,082 battles (see `reports/data_report.md`).
**Alternatives:** a fixed round number such as 1500 (a guess that may keep too few or too many
games); filtering on the average of the two ratings (lets a strong player carry a weak one);
keeping everything (the model would learn beginners' habits, but our agent should predict
competent play).
**Why:** both players must be reasonably skilled for their choices to be worth imitating, and
a fraction tied to the data stays sensible whatever the rating scale looks like. The cost is
that the model sees only good players, which we report as a limitation.

## 2026-11-18: How the replay data is downloaded and stored (B, Rishi)
**Decision:** read only the dataset's `[Gen 9] RANDOMBATTLE_part*.parquet` files at a pinned
dataset revision, keep rows whose `formatid` is exactly `gen9randombattle`, and write one local
Parquet table with the log plus summary columns (ratings, winner, tie, turns, upload time) read
from the log once. Ratings are read per player from the `|player|` lines. When we take a
sample (`max_battles`), we take the newest battles by upload time.
**Alternatives:** `datasets.load_dataset` over the whole 34-million-row dataset (scans every
format); keeping close variants such as Blitz (different time pressure, so different play);
using the dataset's single `rating` column (one number per game, often empty, and we need both
players' ratings to filter); CSV (larger and slower, loses column types).
**Why:** reading only our files is far less data to stream. A pinned revision makes the
download reproducible. Storing the summary columns means filtering and splitting never parse
logs again. The `ratings` column matches what role C's exporter already reads. The first
2,000 rows of part 1 all came from 18 to 20 July 2025 (check 1 in docs/08), so "the first N
rows" would be a few days of old games; the newest games are closest to what our agent will
face, because Showdown updates the Random Battle sets over time.

## 2026-11-13: Draft evaluation plan, written before any results (A, Raghav)
**Decision:** 1,000 games per opponent (random, max power, heuristics) with Wilson intervals;
heuristics is the headline opponent; log loss is the opponent model's main metric; success
criteria fixed in [09-evaluation-plan.md](09-evaluation-plan.md). Team to confirm, then frozen.
**Alternatives:** 500 games (interval about ±4.4 points, too wide to compare ablations);
accuracy as the main metric (ignores probability quality, which the decision engine uses).
**Why:** 1,000 games distinguishes differences of about 4.5 points; fixing criteria before
results prevents goalpost-moving.

## 2026-11-17: F4 domain-review decisions and failure cases (F, Devam)
**Decision:** Raghav accepted concrete move columns, k = 0.25, no new tera rows, zero
outgoing credit against unseen switch targets, and inclusion weights within classes in
PR #44. Remove impossible class mass and rescale only when the full move-pool belief or
team count proves impossibility. Retain errors for missing information or zero remaining
forecast mass. Skip 0/0-HP Illusion entries when counting revealed members and reserves.
**Alternatives:** abort the whole table for one unsupported candidate; silently invent
missing moves; treat unsupported moves as ordinary damage without disclosing it.
**Why:** these were the main failures in the reviewer's integration games. Charge moves
now score as a zero-damage first turn, Protect variants as plain Protect, and self-KO
attacks as damage plus the user's remaining HP loss on an attempt. Sleep/freeze use fixed
1/3 and 0.2 action chances, separate from accuracy. These are explicit approximations,
not exact game mechanics; details and exceptions are in [f4-implementation.md](f4-implementation.md).
Hand-built regressions cover the four requested groups, including misses, inability to act,
earlier KOs, incomplete move evidence and phantom reserves. Status/setup utility beyond
healing and protection remains a non-blocking follow-up, not part of these repairs.

## 2026-11-14: Provisional F4 payoff policy for domain review (F, Devam)
**Decision:** implement the original stub's individual-move columns plus an aggregate switch
column. Preserve class probability mass using explicitly heuristic inclusion weights. Score
HP fractions and a 0.25 KO bonus across endpoint/miss/order branches on copies, cancelling
actions after KOs. Missing candidates and calculation failures remain unavailable, not zero.
**Alternatives:** exactly three class-average columns; midpoint damage before KO checks;
silent forecast renormalisation; inventing hidden switch species or unreviewed status bonuses.
**Why:** retain differences between candidate attacks for minimax, keep the probability and
utility units clear, and expose missing information to callers. The user requested proceeding
with implementation while earlier reviews are pending; this is not domain approval.
**Limits:** supported mechanics, omitted effects, warning/fallback semantics and all provisional
choices are in [f4-implementation.md](f4-implementation.md). This remains subject to review;
no live-agent integration, model training, win-rate claim or shared-interface change is made.

## 2026-11-13: Damage estimates without changing live state (F3, Devam)
**Decision:** `decision/damage.py` fills missing opponent stats with the default Gen 9
Random Battle spread (85 EVs, 31 IVs, neutral nature), using the experiment in check 5 of
[08-foundation-checks.md](08-foundation-checks.md). Copies of the active Pokemon and team
mappings isolate these guesses from the live battle. Known stats, revealed items, abilities,
boosts, types and field conditions are retained; no peer/oracle battle is used.
**Alternatives:** modify the tracked opponent directly; guess its item/ability; score a
calculator failure as zero; rank moves by maximum damage alone.
**Why:** guessed stats must not become observations. Each result records minimum, maximum
and midpoint damage as percent of estimated maximum HP. Failures have `None` values and an
explicit error, distinct from a genuine zero (status move or immunity). This lets F4 choose
a documented fallback. Forced switches and absent/fainted active Pokemon produce no scores.
Recharge is a forced rest turn and scores zero without entering the calculator, because
poke-env's recharge placeholder lacks ordinary move metadata. A regression test covers it.
**Limits:** the midpoint is an approximation conditional on a non-critical hit, not a full
expected payoff. Accuracy, KO capping, secondary effects, switching and status utility are
left to F4. Default spreads miss reduced Attack/Speed and HP adjustments; transformed or
disguised opponents can be misestimated. Unknown items and abilities remain unknown; belief
averaging is future work. poke-env 0.16.1 itself approximates multi-hit moves and has other
documented calculator gaps. Unit tests exercise actual mechanics, and a server-marked test
checks the helper throughout ten complete games without mutating tracked opponent details.

## 2026-11-13: Fill in opponent stats from Random Battle spreads before damage calc (F, Raghav)
**Decision:** before calling poke-env's damage calculator, set the opponent's stats from
species and level with 85 EVs, 31 IVs and a neutral nature in every stat (the generator's
default spread).
**Alternatives:** skip the calculator and use base power only (ignores stats and types).
**Why:** without stats the calculator crashes every time. With them, real damage fell inside
the predicted range 89% of the time with the item hidden and 94% with it revealed, against
94% using the true hidden information. See check 5 in
[08-foundation-checks.md](08-foundation-checks.md).

## 2026-11-13: Drop hidden-choice turns from opponent-model training, and report the bias (E, Raghav)
**Decision:** turns where the opponent's choice is hidden (fainted before acting, couldn't
move, locked into a move) get no label and are dropped.
**Alternatives:** impute a label; reweight classes; use "fainted first" as evidence against
switching.
**Why:** 88% of turns are still usable (bot games), and those labels matched true choices
99.98% of the time. The rows dropped are mostly attacks (switches are never hidden), so
switching is slightly over-represented in the labels (13.0% vs a true 11.4%). Role E decides
whether to correct this; see check 4 in [08-foundation-checks.md](08-foundation-checks.md).

## 2026-11-13: One parser for replays and live games (C, Raghav)
**Decision:** use poke-env's `Battle.parse_message` as the log parser. Build live states by
re-parsing the messages our bot received (`battle._replay_data`) with the same code as
replays, converting our own HP to the public percentage. poke-env pinned to 0.16.1.
**Alternatives:** write our own parser; read the live poke-env battle object directly.
**Why:** poke-env parsed all 400 test games without errors. Re-parsing the live stream matched
the replay view exactly (0 differences in 1.4 million field checks). Reading the live object
directly differs in 0.04% of checks, because it includes private information (exact HP,
Regenerator heals, seeing through Illusion, Ditto copying our moves). See checks 2 and 3 in
[08-foundation-checks.md](08-foundation-checks.md).

## 2026-11-14: Balanced baseline batch with six auditable records (A3, Devam)
**Decision:** play each of the three baseline pairs in both challenge directions, 500 games
each, and aggregate to 1,000 per pair. Keep every JSON summary and compute separate Wilson
win intervals for both bots. Refuse partial/unbalanced or mixed-provenance reports.
**Alternatives:** one challenge direction; combining cumulative counters; reporting only
the most favourable sample; committing a table without the input records or code revision.
**Why:** balancing direction reduces a possible setup confound, while the fixed batch and
linked records make all numbers inspectable. Python seeds do not fix server-generated teams;
these are measured stochastic baseline results, not a claim about our agent or a trained model.
See [baseline-experiments.md](baseline-experiments.md) for reproduction and failure handling.

## 2026-11-14: Fresh-game accounting and saved provenance (A2, Devam)
**Decision:** snapshot each player's battle ids, count only the new finished games, and
cross-check outcomes from both clients. Save an exclusive JSON summary with commit, tracked
change flag, UTC date, package versions, format and seed. The CLI owns and closes its clients.
**Alternatives:** cumulative client counters; overwriting one results file; reporting a partial
run as a completed matchup; treating a Python seed as a seeded server simulation.
**Why:** repeated matchups must not inherit earlier wins. Code and server provenance make
results auditable, but server-generated teams remain stochastic. Source should be committed
before formal runs; raw logs and player names are not needed for this summary. See
[evaluation-runner.md](evaluation-runner.md) for the contracts and limitations.

## 2026-11-14: Wilson intervals measure wins, with ties kept separate (A1, Devam)
**Decision:** return a Wilson score interval for integer wins out of a positive game count;
default z=1.96, nominal 95%. Reject missing/invalid samples. A tie counts as a non-win in
the win fraction, while tournament output reports ties separately.
**Alternatives:** normal/Wald intervals; half a win per tie; a fabricated result for zero games.
**Why:** Wilson behaves sensibly at zero and all wins. Half-credit measures a different
quantity; no games provide no estimated win probability. Bounds describe sampling
uncertainty, not model uncertainty or a guaranteed range. Tests check known values and edges.
Formula reference: [NIST Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

## 2026-11-17: Play with a flat forecast before a learned model (F5, Devam)
**Decision:** best response to a flat three-class forecast with public move-pool priors;
explicit greedy-damage, forced-replacement and server-default fallback paths with counters.
Keep both strategies available, using a per-agent seeded generator for minimax sampling.
**Alternatives:** random play while E5 is unfinished; claim a flat prior is a trained model;
swallow failed tables without reporting coverage.
**Why:** this tests the decision pipeline and F5's weak-baseline milestone independently
of model training. A temporary agent-side cosmetic-form mapping consumes D1 without changing
the peer-owned belief code. See [f5-agent-v0.md](f5-agent-v0.md) for the policy and measurement.

## 2026-11-17: Explicit learned-model mode and strategy-specific explanations (F6, Devam)
**Decision:** require an injected model for model mode; build its inputs only through C3,
D2 and D4. Explain the original class forecast and the actual table criterion, distinguishing
best-response expectations from minimax's worst-case mixture. Log every fallback too.
**Alternatives:** silently use thirds when a model is absent; feed private live stats into
features; explain minimax as though it optimises the model forecast.
**Why:** these alternatives hide missing dependencies or confuse the mathematical guarantee.
E5 is still a stub, so fixed-forecast integration games are labelled smoke-only and cannot
complete F6. See [f6-integration.md](f6-integration.md) for the remaining real-model check.

## 2026-11-13: Initial architecture (all)
**Decision:** public-information states for both sides; split by battle with the newest
battles as test; three action classes for the opponent model; one-turn payoffs; forced
switches excluded; a walking-skeleton agent that always plays.
**Alternatives:** full player view for our own side; random turn-level split; exact-move
prediction; multi-turn search.
**Why:** see [04-architecture.md](04-architecture.md), section "Key design decisions".

## 2026-11-13: Best response requires an explicit opponent forecast (F2, Devam)
**Decision:** take each row's probability-weighted average payoff and pick the highest.
Exact ties choose the first row. Missing probabilities, incorrect lengths, negative or
non-finite entries and sums outside 1 +/- 1e-6 raise `ValueError`. Accepted rounding error
is normalised away. The returned mix assigns probability one to the chosen action.
**Alternatives:** assume uniform probabilities if missing; accept arbitrary weights; choose
randomly between tied rows.
**Why:** an implicit fallback can conceal a missing model connection. Requiring a distribution
makes the expected payoff meaningful. A fixed tie rule makes experiments repeatable without
adding randomness to this strategy. Tests use hand-calculated weighted averages, changed
forecasts, negative payoffs, zero-probability columns and malformed inputs. Callers that want
a uniform forecast must supply it; minimax is the separate strategy that needs no forecast.
