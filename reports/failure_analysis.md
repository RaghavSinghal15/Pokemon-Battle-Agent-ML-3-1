# Failure analysis (M7a)

What goes wrong when the agent loses, read from its own explanation logs. This is the
qualitative side of the evaluation: the win rates say *how often* we lose, this says *why*.

## Method

- **Games:** 100 games of the final agent (best response, value utility, LightGBM forecast,
  beliefs, lasting effects, tera; `main` at 9960abf) against `SimpleHeuristicsPlayer`, played
  with `python scripts/demo_explanations.py --games 100 --quiet --out data/interim/analysis`.
  The agent won 83. The explanation log and the 100 replays stay in `data/` (not committed).
- **Reading:** for each of the 17 losses we lined up, turn by turn, what the agent chose and
  why (its predicted win probability, forecast and best rows) against what then happened in the
  replay. We looked hardest at the turns where the predicted win probability fell most, and gave
  each game one main cause. A game where no choice was clearly wrong (the position was lost to
  a better team or to luck) is marked so rather than forced into a category.
- **Counts over all 100 games:** to check that a cause is not just noise, we also counted how
  often each pattern happens in won games (script logic described at the end).

## Causes

| Cause | Losses | Games |
|---|---|---|
| A. Healing or setting up into a knockout | 5 | 23532, 23550, 23554, 23561, 23577 |
| B. A hidden item or ability made the opponent faster or stronger than estimated | 4 | 23533, 23552, 23609, 23614 |
| C. Tera spent too early or on a Pokemon about to faint | 2 | 23527, 23601 |
| D. No plan to break a stall (one-turn horizon) | 1 | 23526 |
| E. Forecast fit for humans, wrong for the bot | 1 | 23593 |
| No clear error (worse team, or behind after misses and critical hits) | 4 | 23538, 23541, 23566, 23602 |

Luck mattered as a contributing factor in at least four losses (a missed Hurricane in 23532,
a flinch and a missed Fire Blast in 23550, critical hits in 23527 and 23609), but no loss was
decided by luck alone after a sound plan.

### A. Healing or setting up into a knockout (5 losses)

The agent heals (Recover, Roost, Synthesis) or sets up (Shell Smash, Substitute) when the
opponent's strongest attack would knock it out anyway, so the heal never happens.

- 23561, turn 22: Empoleon at 44% uses Roost. Oinkologne's High Horsepower had just done 63%;
  it is faster and knocks Empoleon out. Two turns earlier Gastrodon (38%) did the same with
  Recover.
- 23550, turn 11: Tropius at 40% uses Synthesis against a faster Chien-Pao whose Icicle Crash
  had just done 64%.
- 23554, turn 18: Cloyster at 70% uses Shell Smash in front of Tyranitar; a super-effective
  Stone Edge leaves it at 1% and the sandstorm finishes it.

**Why the table liked these moves.** Within "attack", the table shares the forecast's
probability among the opponent's moves in proportion to *how likely the opponent has each
move*, not *how likely it is to choose it*. Empoleon's Roost loses badly only in the High
Horsepower column, but that column gets only part of the attack probability; against the
other columns (Double-Edge into a Steel type, for example) healing looks good. A real
opponent, human or bot, picks the move that knocks us out. Across all 100 games, 29 of 233
healing moves (12%) were answered by a knockout that same turn.

**Fix (#118):** inside the attack class, put more weight on the opponent's most damaging
moves (for example, weights that grow with the damage each move does to us). This is the
"competent opponent" assumption the bot and most humans satisfy. The exact-move model (E6)
does this from data for humans; it made no difference against the bot, which simply picks
the highest-damage move.

### B. Hidden items and abilities (4 losses)

Damage and speed are estimated with the opponent's revealed item and ability only. An
unrevealed boost makes a hit bigger or a Pokemon faster than the table thinks.

- 23609: Mabosstiff held a Choice Scarf (revealed later by Knock Off), so it moved first and
  knocked out Falinks at 3% before its Close Combat. The table scored Close Combat +0.20,
  which only makes sense if Falinks (+1 Speed from No Retreat) moves first.
- 23552: Talonflame's Gale Wings gives Brave Bird priority at full HP; it knocked out Gallade
  (90%) before it could move.
- 23614: Porygon2's Download raised its Special Attack; Ice Beam knocked out a full-health
  Breloom.
- 23533: a Life Orb Pawmot knocked out the healthy Zacian we switched in, instead of the 26%
  Grimmsnarl we could have left in.

**Fix (#119):** use the belief over Random Battle sets for abilities and items too: count
priority from Gale Wings at full HP, and allow for a Choice Scarf when the opponent's set can
have one, at least in the speed order.

### C. Tera timing (2 losses)

Tera was used on turn 1 (Gholdengo, 23527) and on turn 2 by a Golduck at 52% that was knocked
out the same turn (23601). Across the 100 games, losses used tera on turn 1 or 2 almost twice
as often as wins (47% of losses against 24% of wins), though with 17 losses this is a
pattern to test, not proof.

**Fix (#120):** tune `PAYOFF_TERA_COST` (now a first guess of 0.25 of a Pokemon's HP) in
games, for example 0.25, 0.5 and 1.0.

### D. Stalls (1 loss)

23526 lasted 103 turns. Spectrier used Dark Pulse on a resting Snorlax until it ran out of PP;
then its only moves were Shadow Ball (Snorlax is immune), a maxed-out Nasty Plot and
Substitute. Every row scored exactly 0.000, so for 50 turns it used Nasty Plot while Snorlax
used Curse, until Snorlax won. A one-turn table cannot see that switching to a Pokemon that
can hurt Snorlax pays off later. A cheap guard (#121) would be to prefer a switch when every
move row scores zero for several turns.

### E. Forecast trained on humans (1 loss)

23593, turn 2: the opponent model gave Exeggutor an 81% chance to switch out of a bad
matchup, as a human would. The bot stayed in, and Ariados, switched in to meet the expected
switch, was knocked out by Psychic. This matches the ablation result that the learned forecast
is worth nothing against the bot (746 of 1,000 with plain class frequencies against 745): the
model describes humans, so its value has to be measured on the ladder.

## What this means

- The biggest fixable cause (A) is in the decision engine, not in the learned models: a
  modelling assumption about *which* attack the opponent picks. It is also the easiest to test.
- Causes B and C are missing mechanics and an untuned constant.
- Only one loss (E) came from the forecast, and against a bot that is expected.

## How the counts were made

The explanation log has one block per decision (turn, chosen action, predicted win
probability, forecast, row scores); the replay has what happened. For each decision we checked
whether our active Pokemon fainted in that same turn, and grouped the chosen move as an attack,
a healing move or a setup move. The scripts are small and specific to this run, so they are not
part of the package.
