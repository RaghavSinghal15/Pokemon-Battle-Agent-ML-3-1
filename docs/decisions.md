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
