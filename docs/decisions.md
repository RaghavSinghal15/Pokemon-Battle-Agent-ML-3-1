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

## 2026-11-23: Expect the opponent's attacks that hurt us most (F15 #118, Raghav with Claude)
**Decision:** within the attack class, a candidate move's weight is its belief inclusion
probability times exp(expected damage to us / `PAYOFF_ATTACK_TEMPERATURE`), with damage capped
at our current HP (every knockout counts the same) and temperature 0.1. Non-damaging moves keep
their belief weights; the class probabilities from the opponent model are unchanged.
**Alternatives:** belief weights only (as before); always assume their single strongest attack;
the exact-move model (E6), which learns human move choice.
**Why:** the failure analysis found that healing or setting up into a knockout lost 5 of 17
games: a knockout move got only its share of the attack probability, so Roost or Recover could
look best. Both the bot and humans pick the move that knocks us out. Against
SimpleHeuristicsPlayer, best response wins 1,653 of 2,000 (82.7%) with this against 794 of
1,000 (79.4%) without (p ~ 0.03). Puzzles: 11/12 (HP utility) and 12/12 (value utility), as
before. A softmax keeps some weight on weaker attacks, unlike "always the strongest", because
the opponent does not always know which move hurts most. Minimax is unaffected: it ignores the
forecast probabilities.

## 2026-11-23: Our own tera as an option in the payoff table (F14, Raghav with Claude)
**Decision:** while tera is still available, every move also appears as a "move + tera" row. In
that row our Pokemon takes its tera type before anyone moves, so both our same-type attack
bonus and our weaknesses change. A tera row also gets (a) the change in next turn's table with
the new type (as for other lasting effects) and (b) minus a fixed cost
`PAYOFF_TERA_COST = 0.25` of one Pokemon's HP, converted to P(win) like hazards, because using
tera now gives up using it later, possibly on another Pokemon, which a one-turn look cannot
see. Our tera type comes from the server's request for our own team: our private knowledge,
used only for our own options, never as a model input. The opponent's tera is not modelled.
**Alternatives:** never tera (as before); tera at the first chance; poke-env's heuristic
(tera when the type matchup score is above 1); learn when humans tera from the replays.
**Why:** tera is the main Gen 9 mechanic and humans use it almost every game. Against
SimpleHeuristicsPlayer, best response wins 794 of 1,000 with tera against 745 without
(+4.9 points, p ~ 0.01), with the same code otherwise. Minimax with stand-ins (F13) and tera
wins 700 of 1,000, against 310 of 500 with stand-ins alone and 319 of 1,000 with neither. The cost 0.25 is a first guess, not yet
tuned; learning tera timing from replays is a possible follow-up.

## 2026-11-23: Score attacks into an unseen switch-in against stand-in species, for minimax only (F13, Raghav with Claude)
**Decision:** when the opponent may switch to a Pokemon we have not seen, the switch column
can score our attack against 8 stand-in species drawn from the Random Battle pool (at most one
per base species, none already on their team, the same 8 for the whole battle, seeded by
`SEED`). Before, such an attack did no damage at all. The agent uses stand-ins for minimax
only (`PAYOFF_STAND_INS_FOR = ("lp",)`). To keep it fast, a cell now deep-copies only the two
active Pokemon instead of both whole teams; tables are otherwise identical (checked on all 12
puzzles, both utilities, maximum difference 0.0).
**Alternatives:** keep zero credit; weight reserves by how likely a human is to bring each in
(would need a switch-target model); one representative "average" Pokemon.
**Why:** minimax plans for the worst case. With zero credit, "they switch and my attack does
nothing" was always the worst case, so it played passively and won only 31.9% of 1,000
games. With stand-ins it wins 310 of 500 (62.0%). Best response got worse with stand-ins
(700 of 1,000 against 745, p ~ 0.03): crediting attacks into a 17% switch forecast makes it
stay in and attack where switching out was better against this bot. So the change is
applied where it helps. The puzzles are unchanged for the default (no stand-ins).

## 2026-11-23: Exact-move weights in the payoff table: available, off by default (E8, Raghav with Claude)
**Decision:** the agent can split each action class's probability among the opponent's moves
by the exact-move model (E6) instead of by belief inclusion probabilities
(`ExplainableAgent(exact_move_model=...)`), but the final agent does not use it.
**Alternatives:** always on; leave E6 out of the agent.
**Why:** E6 predicts human move choice far better than the belief prior (51% against 31% top-1),
but against SimpleHeuristicsPlayer it made no difference: 755 of 1,000 with it against 751
without (same code, value utility, hazard weight 1.0). The bot does not choose moves the way
people do, so this is the wrong opponent to measure it on; the ladder (A8) is. Until it shows a
gain there, the simpler agent is the final one.

## 2026-11-23: Hazard weight 1.0, measured from replays instead of a guessed 0.5 (F10 follow-up, Raghav with Claude)
**Decision:** `PAYOFF_HAZARD_WEIGHT = 1.0`: entry hazards are worth the damage of one more
switch-in for every remaining Pokemon.
**Alternatives:** keep the guessed 0.5; use the raw replay figure (about 1.6); tune it on games.
**Why:** in our training replays, a Pokemon that still has to come in does so 1.6 times after
turn 1 (1.5 after turn 5, 1.3 after turn 20). Heavy-Duty Boots (immune to hazards) and hazard
removal (Rapid Spin, Defog) cancel part of that, so we round down to 1.0. Games, value utility,
best response, 1,000 each against SimpleHeuristicsPlayer: 751 at 1.0 and 751 at 0.5, so the
heuristic bot does not reward hazards either way. With 1.0 the value-utility agent solves all
12 puzzles, including Stealth Rock on turn 1. We chose the weight from the replays, not to fit
the puzzle, so the puzzle stays an independent check.

## 2026-11-23: Score the payoff table by change in predicted win probability (V4 #74, Raghav with Claude)
**Decision:** each cell is P(win at the end of the turn) - P(win now), from the logistic value
model (V2), instead of HP traded plus a 0.25 knockout bonus. Only the HP features change within
a turn (active HP, team HP, Pokemon remaining); the rest of the position comes from the
start-of-turn public state, for the two Pokemon that will face each other. A knocked-out
active Pokemon is replaced by the average HP of the rest of its team; losing the last Pokemon
decides the game. Every row is measured against the same P(win now). Hazard damage still to
come is converted at the current rate of P(win) per Pokemon of team HP. The HP utility stays
available (`utility="hp"`) for the ablations.
**Alternatives:** the LightGBM value model (slightly better log loss, 0.581 vs 0.590, but one
model call per outcome branch: thousands per decision); a hand-tuned KO bonus.
**Why:** HP is not worth the same everywhere. Knocking out their last Pokemon wins the game;
chipping a healthy Pokemon when we are 3 Pokemon behind changes little. The value model learned
these exchange rates from 765,596 human positions, and with logistic regression a change in HP
moves the logit linearly, so each branch is one multiplication. Results, same code, best
response, 1,000 games each against SimpleHeuristicsPlayer:

| Utility | Wins | 95% interval |
|---|---|---|
| HP traded + KO bonus | 674 | 64.4% to 70.2% |
| change in P(win) | 751 | 72.3% to 77.7% |

+7.7 points, z = 3.8, p = 0.0001: the largest single gain so far. Puzzles: 11/12 with either
utility, but the value utility's margins are clearer (switching out of a 4x weakness +0.07
against -0.03 for staying in). Puzzle positions now give us a full team (6), so the value
model is not asked about an unrealistic 1-against-6 position.

## 2026-11-22: Exact-move prediction from (position, candidate move) pairs (E6 #79, Raghav with Claude)
**Decision:** for a decision where the opponent used a move, the candidates are the moves the
beliefs allow plus those already revealed. A LightGBM classifier scores each (position,
candidate) pair by whether that move was the one used; the scores are normalised within the
decision. Pair features: belief probability, revealed, used last turn, attack or not, base
power, accuracy, priority, type effectiveness against our active Pokemon, STAB, setup /
recovery / hazard / Protect / pivot / status flags, the user's and target's HP, the user's
boosts, the turn and the number of candidates.
**Alternatives:** one multiclass model over all ~700 moves (most never apply to the Pokemon on
the field); the belief prior alone.
**Why:** scoring pairs lets one model learn "players pick super-effective, strong, same-type
moves" across every species. Validation (all decisions where a move was used; the used move
was a candidate 99.6% of the time, and misses count with probability 1e-4):

| Model | Top-1 | Top-3 | Log loss |
|---|---|---|---|
| uniform over candidates | 22.4% | 61.7% | 1.713 |
| belief prior | 30.7% | 72.4% | 1.622 |
| LightGBM on pairs | 50.8% | 86.9% | 1.281 |

The belief probability carries 25% of the gain, type effectiveness 15%, "used last turn" 10%.
It is not wired into the payoff table yet: the table weights the opponent's moves by belief
probabilities, and this model could replace those weights in a later change.

## 2026-11-22: Deep-learning opponent models: neither beats LightGBM; game history helps a little (DL1 #76, DL2 #77, Raghav with Claude)
**Decision:** keep the per-turn LightGBM as the agent's opponent model. Report the two neural
models as the comparison: DL1, a per-turn MLP (two ReLU layers with dropout, on the same 36
features), and DL2, a GRU that reads the player's whole game so far (each decision's features
plus their previous action, including "hidden").
**Alternatives:** use the GRU in the agent; a small transformer instead of the GRU.
**Why:** validation, all-turns log loss (all trained with the hidden class, same splits):

| Model | Log loss | Accuracy | Macro-F1 |
|---|---|---|---|
| naive class frequencies | 0.891 | 57.8% | 0.244 |
| logistic regression | 0.759 | 62.7% | 0.511 |
| MLP (256, 128), DL1 | 0.681 | 66.3% | 0.569 |
| GRU, no memory (ablation) | 0.687 | 66.1% | 0.570 |
| GRU over the game, DL2 | 0.677 | 66.8% | 0.583 |
| LightGBM (chosen) | 0.670 | 67.2% | 0.584 |

The same GRU with its memory removed loses 0.0095, so what this player did earlier in the game
carries real signal beyond the current position (and beyond the "last action" features). But
LightGBM on the per-turn features is still best: on tabular features like ours, boosted trees
usually beat neural networks, and most of the history's value is already in the position. The
GRU also costs far more to train (84 minutes against a few for LightGBM) and needs the game
history at play time. A transformer would answer the same question with more parameters; the
GRU's small gain suggests it would not change the conclusion. A test cuts each game after turn
N and checks that predictions up to N do not change, so the GRU cannot see the future.

## 2026-11-22: Play styles: four soft groups, and no style feature in the opponent model (E7 #80, Raghav with Claude)
**Decision:** describe each player by six shares of their decisions in training games (switch,
attack, setup, hazard, other status, Protect), cluster with k-means and a Gaussian mixture on
standardised shares, and plot with PCA. Report the clusters; do not add a style feature to the
opponent model.
**Alternatives:** cluster on all six shares; use a player's style (from their other training
games, leave-one-out) as features for LightGBM.
**Why:** 1,058 players have 5+ training games. The shares add up to 1, so we cluster on five
of them: with all six, the Gaussian mixture's covariances were near-singular and its BIC fell
from 2,841 to 93 for no real reason. On five shares, k-means (best silhouette) and the mixture
(best BIC) agree on four groups: switchers (23% switches), setup sweepers (11% setup),
status/Protect users (15% other status) and hazard setters (4% hazards). The silhouette is
only 0.18, so styles are a continuum with soft groups, not distinct types (2-D PCA keeps 53% of
the variance). Adding the style to LightGBM made validation log loss worse (0.678 vs 0.670,
also for the 83% of rows whose player has a known style), and early stopping came at 424 trees
instead of 999: the near-unique style vector acts as a player fingerprint and the trees
overfit to it. Usernames link a player's games locally and never leave data/.

## 2026-11-22: Drain, recoil, Substitute and end-of-turn HP in the payoff table (F11 #69, Raghav with Claude)
**Decision:** drain moves heal their share of the damage dealt and recoil moves cost theirs
(both converted to the user's HP); Substitute costs 1/4 of max HP, then absorbs hits and blocks
effects aimed at its owner (sound moves and hazards go through; an existing Substitute is
assumed full, because the log does not show its HP); at the end of the turn a Pokemon still
standing gains or loses HP from Leftovers, Black Sludge, burn, poison, toxic (counting up) and
sandstorm, with Magic Guard and Poison Heal respected when known.
**Alternatives:** leave them out (the old warning listed them as missing); a full end-of-turn
order as in the game.
**Why:** these are common in Random Battles and the table was blind to them: Giga Drain looked
like a plain weak attack and a burned opponent's chip damage did not count. Games (best
response, 1,000 each against SimpleHeuristicsPlayer): 650 (61.9% to 67.9%) with F11 against 670
(64.0% to 69.8%) without, a difference well within noise (p about 0.34). Puzzles stay at 11/12.
We keep it because it models the game more faithfully at no measurable cost; the heuristic bot
rarely uses Substitute or drain, so these games cannot show much either way.

## 2026-11-22: Allow PyTorch for the deep-learning comparison, inside `dl/` only (X4 #75, Raghav)
**Decision:** `torch` may be imported only in `src/pokeagent/dl/`, installed with the optional
`[dl]` extra. Nothing else imports from `dl/`, so `pip install -e ".[dev]"` and CI stay light.
**Alternatives:** scikit-learn's MLPClassifier only (no sequence models); no deep learning.
**Why:** the question we want to answer is whether a model that reads the whole game so far
(a GRU or small transformer) predicts the opponent better than per-turn models. scikit-learn
has no sequence models, so this needs a deep-learning library. Keeping it in one package keeps
the core agent's install and the viva story simple.

## 2026-11-22: Puzzle positions with their own forecast, and the healing moves they exposed (V3 #73, Raghav with Claude)
**Decision:** twelve hand-built positions in `eval/puzzles.py`, each with the accepted answer(s)
and the forecast a strong player would expect (Blissey uses Seismic Toss 70% of the time, not a
third). `scripts/run_puzzles.py` reports best response with the puzzle's forecast and with a
flat 1/3 forecast, and how much probability minimax puts on accepted answers. The puzzles also
exposed a bug: Moonlight, Morning Sun, Synthesis, Shore Up, Rest and Strength Sap healed 0,
because the move data gives no fixed amount (it depends on the weather or the target). They now
heal 1/2 (2/3 in sun, 1/4 in other weather), Shore Up 2/3 in sand, Rest fully plus sleep, and
Strength Sap the target's Attack.
**Alternatives:** puzzles with a flat forecast only; puzzles in the opponent model's hands; a
larger set generated from replays (no agreed answers).
**Why:** with a flat forecast the agent "failed" puzzles because the forecast was absurd, not
because it chose badly. Fixing the forecast per puzzle tests the payoff table alone. Result:
11/12 (flat forecast 9/12); minimax puts 82% of its probability on accepted answers. The miss is
Stealth Rock on turn 1 against Blissey: Earthquake's chip (+0.18) beats the hazard (+0.10)
because Blissey's healing is not counted against our damage and the hazard weight is 0.5. This
is a known open question for the value model (#74). Minimax spreads its probability when the
opponent might switch (24% on the obvious knockout), because an attack into an unseen switch-in
gets no credit: a known approximation that makes minimax cautious.

## 2026-11-22: Value lasting effects by their change to next turn's table (F8 #66, F9 #67, F10 #68, Raghav with Claude)
**Decision:** a move that leaves something behind is worth what it changes later. Stat changes
(Swords Dance, Draco Meteor's drop, Icy Wind) and statuses (Thunder Wave, Will-O-Wisp, Toxic,
Spore, with type and known-ability immunities) are valued as the change in next turn's table
value, computed one turn ahead from the current HP, times `PAYOFF_EFFECT_TURNS` (1.0). Entry
hazards and their removal are valued as the change in entry damage summed over every Pokemon
still to switch in (unseen opponents count as neutral), times `PAYOFF_HAZARD_WEIGHT` (0.5). An
effect counts only if the move lands, and a boost only if the user survives the turn (a status
only if the target survives). With both weights at 0 the lookahead is skipped.
**Alternatives:** hand-set bonuses per move ("Swords Dance = +0.2"); a deeper search; leaving it
to the value model (#74).
**Why:** the one-turn table gave setup, status and hazard moves no value, so the agent never
used them (the bottleneck in the entry below). Measuring through next turn's table uses the
same damage and speed model as everything else, so +2 Attack is worth more when it turns 2HKOs
into OHKOs, and nothing when we are about to be knocked out. Hazards need their own sum, because
one turn ahead sees only the active Pokemon (the lookahead gave Stealth Rock only +0.01). Best
response, LightGBM forecast (36 features), 1,000 games each against SimpleHeuristicsPlayer: same
code with effects off 606 (57.5% to 63.6%); effects on 670 (64.0% to 69.8%), +6.4 points,
z = 2.98, p = 0.003. For reference, before this change the agent won 641 (61.1% to 67.0%). In 60
logged games the agent now uses Swords Dance, Nasty Plot and Shell Smash. Thunder Wave is now
scored correctly (+0.11 against a fast attacker) but still below a strong attack. Cost: each
decision is about 2.4 times slower (about 0.15 s), fine for live play.
## 2026-11-22: Value model: LightGBM on the position table (V2 #72, Raghav with Claude)
**Decision:** predict P(win) for the player to move with LightGBM (15 leaves, 446 trees by early
stopping), chosen on validation log loss over logistic regression and a one-feature baseline
(team HP lead only). Saved to `artifacts/value_model.pkl`.
**Alternatives:** logistic regression on all 44 features; bigger trees (63 or 255 leaves); the
team HP lead alone.
**Why:** validation (117,018 rows, test untouched):

| Model | Log loss | Brier | Accuracy | AUC | Calibration error |
|---|---|---|---|---|---|
| coin flip | 0.693 | 0.250 | | | |
| team HP lead only | 0.626 | 0.218 | 63.9% | 0.701 | 0.008 |
| logistic regression | 0.590 | 0.204 | 66.6% | 0.743 | 0.012 |
| LightGBM, 15 leaves | 0.581 | 0.201 | 67.1% | 0.750 | 0.010 |

Bigger trees overfit slightly (0.582, 0.584), and the C setting barely matters for logistic
regression (0.58976 to 0.58978). The first five turns stay near a coin flip (0.686), as expected
with random teams nobody has seen yet. By turn 31+ log loss is 0.479. Team HP and Pokemon
remaining carry 71% of the gain. The two players' predictions for the same turn add up to 1
within 0.03 on average, so the model treats the game as symmetric without being forced to.
The baseline's coefficient says +1 Pokemon of team HP lead moves P(win) from 52% to 72% at an
even position, a useful scale for the payoff's KO bonus.

## 2026-11-22: The value model's training table (V1 #71, Raghav with Claude)
**Decision:** one row per decision point, with features from that player's own side and the
label "did this player win". The features are the 36 opponent-model features plus eight
whole-team ones: total HP, status count, how many of their Pokemon our active one outspeeds,
and how many it hits super effectively (and the same for them). Both players' rows of a turn are
kept. Ladder ratings are excluded.
**Alternatives:** one row per turn from p1's side only; damage-calculator features such as "how
many of theirs can we knock out in one hit"; ratings as features.
**Why:** both sides' rows double the data and teach the model that the game is symmetric.
Splits are by battle, so the two mirrored rows never cross splits. Type and speed counts are
cheap (2 minutes for 765,596 rows) and need no guess about unknown sets. Damage-calculator
counts can come later if the model needs them. Ratings describe the players rather than the
position, and the agent does not reliably see them in a live game. A quick check on train shows
the signal: win rate rises from 11% when 2+ Pokemon behind on team HP to 92% when 2+ ahead.
Rows are 51.6% wins rather than 50%, because players who end up winning have slightly more
non-forced decision points.
## 2026-11-22: How we test on the public ladder (A8 #82, Raghav with Claude)
**Decision:** `scripts/play_ladder.py` plays one game at a time on the official
gen9randombattle ladder from a dedicated bot account whose name and password are environment
variables on Raghav's laptop. We record each game's result and Elo before and after (read from
the server's rating line, because poke-env keeps only the "before" number), and save replays
and explanation logs in `data/` (git-ignored, since they name other players). Only a summary with
no usernames (wins, Wilson interval, rating path, mean opponent rating) goes to `reports/`.
`--local` rehearses the same loop on the local server against the heuristics bot.
**Alternatives:** only bot-vs-bot games on the local server; several games at once; committing
per-game logs.
**Why:** people are the target the opponent model was trained on, so the ladder is the real
test; the heuristics bot stays the fast, repeatable yardstick during development. One game at a
time keeps us a good citizen on the public server. A 2-game local rehearsal recorded 1000 ->
1045 -> 1034 correctly.

## 2026-11-22: Seven belief-based features for non-damaging threats (D7 #90, Raghav with Claude)
**Decision:** add seven features computed from the beliefs about the acting player's moves: the
expected number of Status moves in its set; the most likely status-inflicting move that could
land on our active Pokemon now (no status yet, no type immunity); the most likely hazard move
whose hazard isn't maxed on our side (only if we have more than one Pokemon left); the most
likely hazard-removal move when hazards are on its own side; and the probabilities of Protect,
Substitute and a pivot move (U-turn, Parting Shot ...).
**Alternatives:** more raw counts (revealed status moves only); a separate model per move type;
leaving the model to learn this from the existing 29 features.
**Why:** the opponent model caught only 32% of non-damaging turns, mostly calling them attacks,
because the 29 features never said what the player could usefully do with a status move right
now. On validation (bias-corrected LightGBM, all turns) log loss falls from 0.721 to 0.670,
accuracy rises from 64.8% to 67.2% and non-damaging recall from 32% to 48%; the switch forecast
stays calibrated (error 0.004). "Expected status moves" is the most-used feature by gain (13%).
All seven use only public information through the beliefs, so they add no leakage.

## 2026-11-22: A better forecast barely moves the win rate; work on the payoff table next (E, F, Raghav with Claude)
**Decision:** after plugging the opponent model into the agent, put the next effort into the
payoff table (#66 to #69) and the value model (#71 to #74), not into a better forecast.
**Alternatives:** keep improving the opponent model first; run more games to resolve the small
differences.
**Why:** best response against SimpleHeuristicsPlayer, 1,000 games each (Wilson 95%): no beliefs
and a flat 1/3 forecast 617 (58.6% to 64.7%); beliefs with a flat forecast 606 (57.5% to 63.6%);
beliefs with the bias-corrected LightGBM forecast (29 features) 632 (60.2% to 66.1%). The model
adds about 2.6 points over beliefs alone, which is not significant (z about 1.2). The forecast
mostly helps when we can punish a predicted switch or setup turn, and the one-turn payoff table
gives setup, status and hazard moves almost no value, so it cannot act on that knowledge. A
1,000-game run can only detect differences of about 4 points, so more games would not settle
this; the puzzle positions (#73) will show which decisions the forecast changes.

## 2026-11-22: Correct the hidden-choice label bias with a "hidden" class; no separate calibration (E4 #32, C6 #91, Raghav with Claude)
**Decision:** the parser also exports turns whose choice the log hides (the Pokemon fainted
before acting, couldn't move or was dragged out; lock-ins and recharging are not choices and
stay out). The opponent model trains on all turns with "hidden" as a fourth class, then folds
it back: P(switch) is the switch column unchanged, and the hidden share is split between attack
and non-damaging in the ratio the model gives them for that state. The model is chosen and
reported by log loss on all turns, where a hidden turn scores -log(1 - P(switch)). No Platt or
isotonic step is added.
**Alternatives:** accept and report the bias; reweight classes to an estimated true frequency
(fixes the average but not the per-state error); a separate model of P(choice visible | state)
(equivalent, but two models); Platt or isotonic calibration on validation.
**Why:** 10.4% of human choices are hidden (88,624 turns), and a switch is never one of them,
because switches happen first. Trained on visible turns only, LightGBM predicts a 19.9% switch
rate on validation against a true 17.1%, and is most overconfident where it matters: when it
says 66% switch, players switch 55% of the time. With the hidden class it predicts 17.3%, and
the switch calibration error falls from 0.028 to 0.004 (reports/figures/
opponent_switch_reliability.png). All-turns log loss improves from 0.729 to 0.721. The folding
assumption is supported by bot games with known choices: 91% of hidden choices were attacks,
against 90% of visible non-switch choices (docs/08). Attack-vs-non-damaging was already
calibrated (error 0.007), so the remaining miscalibration came from the bias, not the model, and
a calibration step would have to be fitted on the validation set that already chose the model.
Visible-only log loss gets worse (0.774 to 0.782), as it should: that metric rewards the bias.

## 2026-11-22: Opponent model v1 candidates: logistic regression and LightGBM, chosen by validation log loss (E2 #15, E3 #31, Raghav with Claude)
**Decision:** both models train on the train split; every choice (logistic regression's C,
LightGBM's leaves per tree and number of trees by early stopping) is made on validation log
loss. `models.opponent.load_split` refuses the test split unless asked explicitly, so test is
read once, in E5. Log loss is the headline metric, because the payoff table multiplies by the
probabilities; accuracy and macro-F1 are reported alongside.
**Alternatives:** choose by accuracy or macro-F1; cross-validation by battle instead of one
validation split; a wider hyperparameter search.
**Why:** validation results (117,018 decision points): uniform 1.099, naive class frequencies
0.972, logistic regression 0.854, LightGBM 0.774 (accuracy 0.578, 0.614, 0.654; macro-F1 0.244,
0.469, 0.556). C made no difference (0.8539 for 0.01 to 10), so the linear model is limited by
its form, not by overfitting; LightGBM's gain shows it uses interactions the linear model
cannot. LightGBM's three tree sizes were within 0.002 of each other (15 leaves 0.7755, 63
0.7738, 255 0.7735), so a wider search is unlikely to matter. One validation split of 2,412
battles is large enough that differences of this size are not noise. Validation numbers are
slightly optimistic because validation chose the settings; E5 reports test.

## 2026-11-20: Save tournament replays from the spectator channel (A5, Devam)
**Decision.** Capture finished rooms through a separate spectator session while games run.
Write a separate local JSONL archive with one outcome and public log per fresh battle.
**Alternatives.** Save a player's replay stream or fetch all rooms after the tournament.
**Why.** Showdown's spectator channel already hides the private player view. Reading rooms
as they finish keeps early replays available in long tournaments. The summary is saved only
after checking that each completed game has a public log with an ending.
## 2026-11-20: Newest-sample ties filled by battle id; rebuild refuses first (B, #64, Raghav with Claude)
**Decision:** when `--max-battles` cuts through battles uploaded in the same second, every
battle strictly newer than that second is kept, and the remaining places go to the tied
battles in battle-id order. The build checks for an existing `splits.json` before it downloads
or filters anything.
**Alternatives:** keep ties in file order (what B2 did); keep all ties and exceed the limit;
check the split only when saving it (what B4 did).
**Why:** Devam's reviews of #56 and #58 reproduced both bugs. File order could drop a newer
battle for an older tie, so the sample depended on how Hugging Face happened to store the
rows. Battle-id order is deterministic and independent of storage. Exceeding the limit would
make `--max-battles` mean something different from what it says. Checking the split last let a
refused run replace the files the frozen split refers to.
## 2026-11-22: Action-class labels follow the move's category, in one shared rule (E1 #14, Raghav with Claude)
**Decision:** a switch is "switch"; a move is "attack" if its official category is Physical or
Special and "non_damaging" if it is Status (Showdown's Gen 9 move data, via poke-env). The rule is
written once, in `models/labels.py`, and used by the label column of the feature table, the
"opponent's last action" feature and the payoff table's grouping of the opponent's moves.
**Alternatives:** hand-written exceptions, e.g. counting Status moves that switch the user out
(Parting Shot, Shed Tail, Teleport, Chilly Reception) as switches, or labelling Sleep Talk by the
move it calls; keeping separate copies of the rule in each module.
**Why:** the label answers "what type of choice did the player make", not "how good was it";
valuing moves is the payoff table's job (#66 to #69, value model #71 to #74). Exceptions are
rare: on the 765,596 decision points, the four status pivots are 1,806 turns (0.24%) and Sleep
Talk 276. U-turn (11,435), Volt Switch (4,959) and Flip Turn (4,064) are attacks either way:
the player chose a damaging move. Every recorded move has a category, so no turn is dropped.
One shared rule means the model can never predict one grouping while the payoff table reads
another; a test checks the feature code against it, and recomputing all 765,596 labels with it
reproduces the existing feature table exactly.

## 2026-11-20: Forced replacements chosen by next-turn matchup (F7 #61, Raghav with Claude)
**Decision:** after a faint or a forced switch, build next turn's payoff table as if each
reserve were in battle (its own moves against the opponent's likely moves, with hazard damage
on entry) and send in the reserve whose table scores best: the best expected row for best
response, the game value for minimax. The forecast is flat 1/3 in both modes. HP decides only
ties, and the healthiest reserve is still the fallback when no table can be built (both active
Pokemon fainted at once, so the opponent's next Pokemon is unknown).
**Alternatives:** keep "most HP"; a hand-written type-chart score; query the opponent model
on hypothetical states.
**Why:** about 19% of decisions are forced replacements, and "most HP" sends in Pokemon that
the opponent hits super effectively. Reusing the payoff table brings in type matchups,
abilities, items, Speed and Stealth Rock's type scaling with no new scoring rules to defend.
The opponent model was trained on states with our real active Pokemon, so asking it about
hypothetical replacements would be outside its training data. In 200 development games per
strategy against `SimpleHeuristicsPlayer`, best response went from 93 to 122 wins and
minimax from 60 to 72.

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

## A8: ladder runs follow Showdown's bot policy (2026-11-24)

**Decision.** The ladder accounts are registered as bots by their owner before any ladder game.
Each bot says it is a bot at the start of every game, plays one game at a time, and
`--exclude-opponent` drops games our two accounts play against each other.

**Why.** Since October 2026 Showdown requires bots to be registered (running an unregistered
one is a bannable offence). A registered bot is matched only with players who opted in to
playing bots, and with other bots, so ladder opponents are no longer a random sample of all
players: some of them are other bots. We report this as a limitation rather than claiming
"against humans" outright.

**Alternatives.** (1) Ladder without registering: against the rules. (2) Only challenge
volunteers (friends, the course): fewer games and a much narrower skill range.
(3) Comparing win rates across the two accounts: the ladder pairs players of similar rating, so
win rates drift towards 50%. We compare final ratings instead.

## DL3: every opponent model plays in the agent, the GRU through a live history (2026-11-24)

**Decision.** For the agent games of the deep-learning comparison, the full agent's LightGBM
forecast is swapped for another model and nothing else changes: logistic regression and DL1's
MLP drop straight in (they read one turn's features). DL2's GRU needs the opponent's earlier
turns, so `dl/sequence_agent.py` rebuilds them every turn from the battle log poke-env has
received, with the parser walk (`parser.replay.choices_so_far`) and feature code that built the
training sequences. The agent got one hook for this, `ExplainableAgent._forecast`.

**Why.** Rebuilding the history with the training code, rather than tracking it separately in
the agent, guarantees the GRU sees live inputs made the same way as its training inputs
(labels follow models/labels.py; hidden and locked turns are treated as in training). Games
are short (about 30 turns), so re-parsing the log each turn is cheap.

**Alternatives.** (1) Track the history incrementally in the agent: faster, but a second
implementation of the label rule that could drift from training. (2) Test-split metrics only,
no games: the plan asked for games, and showing that games against the bot barely separate
forecasts is itself a result.

## F16 (#119): unrevealed items and abilities count by their belief (2026-11-24)

**Decision.** When the opponent's item or ability is not revealed yet, the payoff table weighs
what it might be: P(Choice Scarf) mixes the speed order, P(Gale Wings) gives Flying moves +1
priority at full HP, and Life Orb, Choice Band and Choice Specs scale their damage by the
expected multiplier. Item probabilities come from 50,000 teams sampled from the local
simulator (`beliefs/items.py`, `beliefs/sample_items.py`), conditioned on the revealed moves and
ability, so the item belief is Bayes' rule by counting.

**Why.** 4 of 17 analysed losses came from a boost we only learned about when it hit us.
Items are not in sets.json (teams.ts picks them from the moveset), so sampling the real
generator is the honest prior, and it needs no hand-written copy of teams.ts.

**Alternatives.** (1) Re-implement teams.ts item rules: a lot of code that drifts when
Showdown updates. (2) Usage statistics from pkmn.github.io/randbats: not reachable from every
machine, and not conditioned on the moves we have seen. (3) Assume the worst item: too
pessimistic, it would make the agent switch out of winning matchups. Opponent-model features do
not read item beliefs, so the trained models are unaffected.

## F18 (#121): a stall guard (2026-11-24)

**Decision.** If PAYOFF_STALL_TURNS (3) turns in a row change nobody's HP and every row of the
table scores within ±0.0005 of zero, the agent takes the best-scoring switch instead.

**Why.** In a 103-turn loss our Spectrier, out of attacking PP, used Nasty Plot for 50 turns
while Snorlax set up. A one-turn table sees no gain anywhere, so it never switches. Requiring no
HP change keeps the guard out of won positions where the value model is simply saturated.

**Alternatives.** Count turns with a flat table only (would fire in already-won games), or a
deeper search (#70), which is much more expensive per decision.

## F12 (#70): two-turn search as a correction on the one-turn table (2026-11-24)

**Decision.** With search on, the agent expands our 3 best move rows against the opponent's 2
most likely moves. For each of those cells it takes the HP both Pokemon are expected to have at
the end of the turn (over the branches where both survive), builds next turn's table from that
position, and adds (its value - the value of next turn's table from the HP as now) to the cell.
Cells where a knockout is more likely than not, switch rows and the switch column are not
expanded. Off by default; the `two_turn` ablation turns it on, and it becomes the default only
if it wins.

**Why.** The one-turn table cannot see that a trade leaves us slower or in KO range next turn.
Adding a difference keeps expanded and unexpanded cells on the same scale (the same form as
lasting effects, F8). Expanding only the likeliest cells bounds the cost: on 4 local games the
mean decision time went from 0.19 s to 0.31 s.

**Alternatives.** (1) Full two-ply expectimax over every cell: tens of next-turn tables per
decision, several times slower. (2) Replace the cell value by the two-turn value instead of
adding a difference: unexpanded cells would then be on a different scale. (3) Monte Carlo
rollouts with the simulator: closest to the real game, but far slower, and no longer a table
the explanation can show.
