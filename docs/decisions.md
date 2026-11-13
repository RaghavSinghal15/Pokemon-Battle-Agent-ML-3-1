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

## 2026-11-13: Draft evaluation plan, written before any results (A, Raghav)
**Decision:** 1,000 games per opponent (random, max power, heuristics) with Wilson intervals;
heuristics is the headline opponent; log loss is the opponent model's main metric; success
criteria fixed in [09-evaluation-plan.md](09-evaluation-plan.md). Team to confirm, then frozen.
**Alternatives:** 500 games (interval about ±4.4 points, too wide to compare ablations);
accuracy as the main metric (ignores probability quality, which the decision engine uses).
**Why:** 1,000 games distinguishes differences of about 4.5 points; fixing criteria before
results prevents goalpost-moving.

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

## 2026-11-13: Initial architecture (all)
**Decision:** public-information states for both sides; split by battle with the newest
battles as test; three action classes for the opponent model; one-turn payoffs; forced
switches excluded; a walking-skeleton agent that always plays.
**Alternatives:** full player view for our own side; random turn-level split; exact-move
prediction; multi-turn search.
**Why:** see [04-architecture.md](04-architecture.md), section "Key design decisions".
