# Foundation checks

Before anyone builds their module on top of the design, we checked the assumptions that would
sink the project if they were wrong. This page records what we tested, what we found and what
it means for each role. The scripts live in [`scripts/spikes/`](../scripts/spikes/) and can be
rerun by anyone (see the bottom of this page).

A **spike** is a quick, throwaway experiment that answers one question. Spike code is not
production code: roles C, D, E and F reuse the *findings* and may copy pieces, but their own
modules are written properly with tests.

## Summary

| # | Question | Status | Answer in one line |
|---|---|---|---|
| 1 | Is the replay dataset what we think it is? | **Waiting on a laptop run** | Script ready: `scripts/spikes/check_dataset.py`. The workspace that ran the other checks can't reach Hugging Face |
| 2 | Can we turn a replay into turn-by-turn states? | **Yes** | poke-env's own battle tracker parses public logs. 400 games, 0 crashes |
| 3 | Do replays and live games give the same state? | **Yes, if built the same way** | Re-parsing the live player's message stream gives 0 differences in 1.4 million field checks |
| 4 | Can we see the opponent's choice often enough? | **Yes (bots; humans to confirm)** | 88% of turns give a label. The labels match true choices 99.98% of the time. One bias to handle |
| 5 | Does the damage calculator cope with hidden info? | **Yes, with one fix** | It refuses to run until we fill in the opponent's stats, then is in range 89–94% of the time |
| 6 | Freeze `interfaces.py` | **Ready to freeze, after 3 small changes** | See "Proposed interface changes" below |
| 7 | Fix the split once | Blocked on 1 | Rule agreed: split by battle, newest games are test |
| 8 | Write the evaluation plan before results | **Draft written** | [09-evaluation-plan.md](09-evaluation-plan.md), team to confirm |
| 9 | Everyone's setup works | Each person | Run `python scripts/check_setup.py`, post the output |
| 10 | Know what the instructor wants | Raghav / team | Deliverables, viva format, required algorithms |

All local results come from **400 bot-vs-bot games** on the local server (heuristics vs
heuristics, random vs heuristics, max power vs heuristics, max power vs random, 100 each):
**17,460 decision points**, where a decision point is one player at the start of one turn.

---

## 2. Turning a replay into states

**Question.** A replay is a text log of everything a spectator saw. We need to turn it into a
snapshot of the battle at the start of every turn. Writing that parser from scratch would be
the biggest task in the project.

**What we did.** poke-env already reads these logs: it's how a live bot keeps track of a game.
We created an empty poke-env `Battle`, told it which player's side to take (the
"perspective"), and fed it the spectator log line by line, taking a snapshot at every
`|turn|N` line. Code: [`scripts/spikes/replay_parse.py`](../scripts/spikes/replay_parse.py), about
40 lines.

To get genuine spectator logs, a third connection joined each finished battle room as a guest.
Showdown sends a joining spectator the whole public log, which is the same view a recorded
replay has.

**Result.** All 400 games parsed with no errors, in about 2 milliseconds per game (both
perspectives). At that speed, 100,000 games take about 3 minutes on one core.

**What it means.** Role C does not need to write a parser. Their job becomes converting
poke-env's `Battle` into our `PublicState`, plus the rules below. Human replays contain lines
bots never produce (chat, timer notices, lines from older server versions). The dataset
script counts any line poke-env can't handle, so we'll know how much that matters.

## 3. Replay view vs live view

**Question.** The opponent model learns from states built from replays, but at play time it
reads states built from a live game. If the two disagree for the same situation, the model
silently gets inputs it never saw in training. This kind of disagreement between training and
use is called **training/serving skew**.

**What we did.** For every battle we recorded three views at the start of every turn, from
both players' perspectives:

- **R (replay):** the spectator log, parsed as in check 2.
- **L (live):** the live bot's own battle object at the moment it chose a move.
- **S (stream):** the messages the live bot actually received, parsed with the same code as R.

Then we compared every field: HP, status, boosts, revealed moves, item, ability, tera, side
conditions, weather. Code: [`scripts/spikes/compare_views.py`](../scripts/spikes/compare_views.py).

**Results.**

| Comparison | Field checks | Disagreements |
|---|---|---|
| R vs S (replay vs re-parsed live stream) | 1,408,114 | **0** (after the HP rule below) |
| R vs L (replay vs live battle object) | 1,226,113 | 475 (0.04%), all explained below |

(R vs L checks fewer fields: on our own side, the live object knows our private moves, item
and ability, so only public fields are compared there.)

Every R vs L disagreement is a case where the live bot legitimately knows something a
spectator doesn't:

| Cause | What happens | Count |
|---|---|---|
| Exact HP | A player sees its own HP exactly (221/315); everyone else sees a rounded percentage (71/100) | Fixed by the HP rule |
| Form changes | Mimikyu, Eiscue, Terapagos and Palafin change form mid-battle; the two views name them differently (Ogerpon also did in an earlier run) | 270 |
| Regenerator | This ability silently heals a Pokémon when it switches out. The owner sees the new HP; spectators don't until it comes back | 157 |
| Illusion | Zoroark disguises itself as a teammate. Spectators are fooled, the owner isn't | 39 (26 HP, 13 other) |
| Ditto | When Ditto copies one of our Pokémon, the live view copies our *private* moves and ability into the opponent's Ditto | 9 |

**The HP rule.** Showdown shows other people `ceil(100 × hp / max)` percent, except that a
Pokémon that is not at full HP never shows 100 (it shows 99). Source:
`getHealth` in `sim/pokemon.ts` of pokemon-showdown.

**What it means (design decision).** The live converter should **not** read the live battle
object. It should take the messages the bot received and run them through *the same parser*
the replays go through, with own-side HP converted by the HP rule. Then training and play
share one code path, and parity holds by construction rather than by careful coding. The
Ditto case also shows that reading the live object directly can leak our private information
into the opponent's description.

The received messages live in poke-env's `battle._replay_data`. That's a private attribute,
so poke-env could change it in an update. We pin poke-env to the exact version we tested
(`0.16.1`). As a fallback, the bot can record messages itself.

## 4. Can we see what the opponent chose?

**Question.** The opponent model's label is "what the opponent did this turn". A replay shows
what *happened*, not what was *chosen*, so some choices are hidden.

**What we did.** For every turn and player, we read the public log and put the turn in a
bucket. The local bots also recorded what they *really* chose, so we could check the labels
against the truth. Code: [`scripts/spikes/label_visibility.py`](../scripts/spikes/label_visibility.py).

**Results (all 400 games).**

| Bucket | Share | Usable label? |
|---|---|---|
| attack (Physical or Special move) | 69.0% | Yes |
| switch | 11.4% | Yes |
| non-damaging (Status move) | 7.5% | Yes |
| fainted before acting | 9.8% | No: choice hidden |
| can't move (asleep, paralysed, flinched, recharging, hit itself in confusion) | 1.7% | No: choice hidden |
| locked into a move (Outrage etc.): not a free choice | 0.6% | No |
| battle ended first | 0.03% | No |

**88% of turns give a usable label, and those labels match the bots' true choices in
15,338 of 15,341 cases (99.98%).** All 3 mismatches came from the recording script, not from
the labelling rules: in each, the log clearly shows the move, and the bot's forced replacement
afterwards had been filed under the same turn.

The rules that make this work:

- A switch that comes before any move in the turn is a real choice. Switches always happen
  before moves. Switches after a move are follow-ups (U-turn) or forced (Eject Button).
- A move with a `[from]` tag isn't the player's choice: it was called by another effect (Magic
  Bounce, Dancer) or it's a lock-in (`[from]lockedmove`).
- Anything after `|upkeep|` is a forced replacement after a faint, not a turn choice.

**The bias.** Hidden choices are not random. Because switches always happen first, *a switch
is never hidden*. Of the 2,119 hidden turns, 91% were really attacks and only 2 were
switches. So dropping hidden turns makes
switching look more common than it is:

| | attack | non-damaging | switch |
|---|---|---|---|
| All true choices | 80.0% | 8.6% | 11.4% |
| Usable labels only | 78.5% | 8.5% | 13.0% |

Statistics calls this **missing not at random**. Options for role E, to decide and record in
[decisions.md](decisions.md):

1. Accept it and report it (simplest).
2. Reweight classes during training so class frequencies match an unbiased estimate.
3. Use the fact that hidden turns are almost never switches: treat "fainted before acting" as
   evidence of "not a switch".

Bots are not humans: human players switch far more and play status moves differently. These
shares will change on real replays. The dataset script measures the same buckets on human
games.

## 5. The damage calculator with hidden information

**Question.** Role F scores each of our moves with poke-env's damage calculator. The calculator
needs the defender's stats, item and ability, which we usually don't know for the opponent.

**What we did.** A probe bot played 40 games against the heuristics bot. Every turn, for each
of its damaging moves, it computed the damage range two ways:

- **Public:** opponent stats filled in from species and level, item and ability as revealed so
  far. This is what we could do in a real game.
- **Oracle:** the opponent's *true* stats, item and ability, peeked from the other bot. This is
  only possible in an experiment. It separates errors caused by hidden information from errors
  in the calculator itself.

It then compared both predictions with the damage the move actually did. Code:
[`scripts/spikes/damage_calc_check.py`](../scripts/spikes/damage_calc_check.py).

**Results.**

- **Without filling in opponent stats, the calculator crashes every time**
  (`AssertionError: defender stats not defined`).
- Stats can be filled in almost exactly. The Random Battle generator gives every Pokémon 85
  EVs and 31 IVs in every stat with a neutral nature. The only exceptions are 0 Attack for sets
  with no physical moves, 0 Speed for Trick Room or Gyro Ball sets, and a small HP tweak.
  Source: `data/random-battles/gen9/teams.ts`.
- Real damage inside the predicted range (455 measured turns):

| Prediction | In range |
|---|---|
| Public, opponent item still hidden | 89.4% (313/350) |
| Public, opponent item revealed | 94.3% (99/105) |
| Oracle (true hidden information) | 94.0% (425/452) |

So hidden information costs about 5 points of accuracy, and the calculator's own approximations
another 6 (its comments say it ignores several edge cases; Knock Off was the most common miss).
For *ranking* our moves, that's good enough for version 1.

**What it means.** Role F needs a helper that fills in the opponent's stats from species and
level before calling the calculator. Role D's beliefs come in for item and ability. Option A:
compute damage under the most likely item. Option B: compute it under each likely item and
average with the belief probabilities. Option B is a nice "beliefs change decisions" example
for the report.

A reference point: the probe bot simply plays the highest calculated damage, and it won 15 of
40 games against the heuristics bot (95% confidence interval 24% to 53%). Pure damage greed
loses, so beating the heuristics bot is a meaningful bar.

## 1. The dataset (run on a laptop)

The workspace that ran the checks above can't reach Hugging Face, so this check has to run on
a laptop:

```bash
pip install -e ".[dev]"
python scripts/spikes/check_dataset.py            # samples 2,000 games, takes a few minutes
```

It prints only counts and summaries (no usernames, no log text). It reports:

- the dataset's files and licence;
- the number of Gen 9 Random Battle games;
- the columns;
- whether logs are complete and public-view;
- the rating distribution and date range;
- poke-env parse success on human games;
- label visibility on human games;
- parse speed.

Paste the whole output into the group. If the data turns out unusable, the fallbacks are in
[05-plan.md](05-plan.md).

## Proposed interface changes (check 6)

These follow from the findings. Each one needs the affected owners' agreement before
`interfaces.py` is frozen:

1. **`PokemonView.hp_fraction` is always the public percentage** (a multiple of 0.01, by the
   HP rule), for both sides, including our own. Affects C, D. *Docstring change only.*
2. **`PokemonView.species` is the species when it was first revealed**, ignoring mid-battle
   form changes (Mimikyu-Busted, Eiscue-Noice, Terapagos-Terastal, Palafin-Hero). That is the
   name `sets.json` uses, which role D needs. Affects C, D. *Docstring change only.*
3. **Item semantics, spelled out.** poke-env uses `"unknown_item"` for "not revealed yet" and
   `""` or `None` for "known to have no item". Map these to `revealed_item=None` (unknown) vs
   `item_removed=True` (known: none). Affects C, D, F. *Docstring change only.*

None of these changes a field's type, so nobody's code breaks. Accepted risk, stated in the
docstrings: Illusion means the public state is sometimes fooled. That's correct for a
public-information design (a human spectator is fooled too) and rare (39 field checks out of
1.2 million).

## Rerunning everything

With the local server running (see [07-setup.md](07-setup.md)):

```bash
python scripts/spikes/collect_local_games.py --p1 heuristics --p2 heuristics --games 100 --tag hh
python scripts/spikes/collect_local_games.py --p1 random --p2 heuristics --games 100 --tag rh
python scripts/spikes/compare_views.py data/interim/local_games.jsonl          # checks 2 and 3
python scripts/spikes/label_visibility.py data/interim/local_games.jsonl       # check 4
python scripts/spikes/damage_calc_check.py --games 40                          # check 5
python scripts/spikes/check_dataset.py --local data/interim/local_games.jsonl  # offline test of check 1
```

`--tag` must be different on every run: it sets the bots' usernames, and the server won't
reuse a name that's still logged in. Generated games go to `data/interim/`, which is
git-ignored.
