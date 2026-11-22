# An annotated game

One game of the full agent (value-utility payoff table, LightGBM opponent forecast, beliefs,
lasting effects, best response) against `SimpleHeuristicsPlayer`, played on the local server
with `python scripts/demo_explanations.py`. The agent won. The complete explanation log is
[example_game.log](example_game.log) and the replay is [example_game.html](example_game.html)
(open it in a browser).

How to read a turn of the log:

- **Beliefs**: the probability of each Random Battle role (set) for the opponent's active
  Pokemon, after Bayes' rule on everything it has shown so far.
- **Opponent forecast**: the LightGBM model's probability that the opponent attacks, uses a
  non-damaging move, or switches this turn.
- **Expected row payoffs**: for each of our options, the forecast-weighted average of the
  payoff table row. With the value utility, a payoff is the change in our predicted chance of
  winning over this turn: +0.050 means 5 percentage points more likely to win.
- **Chose**: best response takes the highest row.

## Turn 1: Knock Off into an unknown set

> poliwrath (100%) vs munkidori (100%); Beliefs: AV Pivot 50%, Fast Attacker 50%;
> forecast attack 90%; knockoff +0.161, liquidation +0.072, closecombat -0.038

Munkidori could be either of its two Random Battle roles, so the beliefs split 50/50. Close
Combat looks strong on paper but Munkidori is part Psychic and resists it. Knock Off is
super effective on the Poison/Psychic Munkidori and removes its item, so it scores highest.

## Turn 2: getting a 4% Pokemon out of the way

> poliwrath (4%) vs munkidori (15%); every attack -0.046; switch:mandibuzz +0.065

Poliwrath is at 4% and Munkidori is faster: any attack is lost because Poliwrath faints first
(all attack rows have the same value). Switching keeps Poliwrath alive for later (a sacrifice
or a free switch) and brings in Mandibuzz, which is Dark type and immune to Munkidori's Psychic
moves. The beliefs collapsed to "Fast Attacker 100%" after what Munkidori showed on turn 1.

## Turn 3: the forecast expects a switch

> mandibuzz (100%) vs munkidori (15%); forecast switch 61%; knockoff -0.006

At 15% HP and facing a bad matchup, the opponent model gives 61% to Munkidori switching out.
The agent still attacks: if Munkidori stays, Knock Off knocks it out. Every option is slightly
negative because of the likely switch: a healthy Pokemon then takes the place of a 15% one, which
the value model scores as a slightly worse position for us, and an attack into a Pokemon we have
not seen yet gets no damage credit (a known approximation of the payoff table).

## Turn 9: switching to a resistance

> toxtricity (20%) vs klawf (100%); every attack negative; switch:rhydon -0.003

Klawf is a Rock type with strong physical attacks and Toxtricity is low. Rhydon resists Rock,
so it is the only option that does not lose win probability this turn.

## Turn 10: a deliberate sacrifice

> Forced replacement: poliwrath +0.304, vikavolt +0.282, mandibuzz +0.250 ...

Rhydon fainted. The agent sends in Poliwrath at 4% HP, saved on turn 2: it will be knocked out
next turn, but that gives the next Pokemon a free switch in, without taking a hit.
(Replacement choices are still scored with the HP utility; see docs/decisions.md.)

## Turn 13: a known gap

> Fallback: forced replacement could not be scored ... Both active Pokemon must be alive and revealed

Volt Switch knocked out Klawf, so both sides choose a replacement at the same moment and the
opponent's next Pokemon is not known yet. The table cannot be built without an opponent, so the
agent falls back to its healthiest Pokemon (Delphox). This is the rare documented fallback.

## Turns 14-17: setup and the finish

> Turn 14: delphox (100%) vs honchkrow (100%); nastyplot +0.028 (lasting effect)
> Turn 17: toxtricity (20%) vs komala (12%); overdrive +0.211; P(win) now 79%

Nasty Plot (+2 Special Attack) is chosen because the lookahead values the boosted damage on
later turns (F8). The boosted Fire Blasts bring Komala down to 12% before Delphox faints, and
Toxtricity finishes the game. The predicted win probability climbs from 48% on turn 1 to 79%
before the last attack.
