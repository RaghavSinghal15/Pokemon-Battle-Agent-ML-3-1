# F4 payoff table: proposal for review

Status: design preparation for [issue #33](https://github.com/RaghavSinghal15/Pokemon-Battle-Agent-ML-3-1/issues/33).
This document proposes a scoring policy and worked test scenarios. It does not implement
`build_payoff_table`, finalise game assumptions, or complete F4. F1/F2/F3 are still awaiting
review in PRs #40, #41 and #42. Revisit this proposal if those reviews change their contracts.

## What a cell means

A row is one action we can choose. A column is a possible opponent action. The number in
their intersection estimates how good that pair is for us over one turn; higher is better.
F1 finds a protective mix of rows. F2 picks the best row against predicted column probabilities.
Neither strategy can compensate for misleading numbers in the table.

Keep the existing `PayoffTable` and `build_payoff_table(battle, beliefs, class_probs)` signature.
Build scenarios on copies. Our private team details may be used by this decision policy, as
allowed by [the architecture](04-architecture.md); learned features remain public-information
only. Do not inspect a peer bot, future replay events, an unrevealed opponent team or test results.

## Rows, columns and probabilities

**Rows.** Use the current legal moves and switches, with stable ordering and distinct labels.
Forced replacements are outside this simultaneous-choice table. For voluntary type changes
(terastallization), reviewers must choose whether the first implementation includes separate
legal tera rows or explicitly defers them. Existing type changes must still affect damage.
Never give a tera row the damage of its ordinary-move row without recalculating it.

**Columns: recommended interpretation.** Keep individual plausible opponent moves plus one
aggregate switch column, as the existing `decision/payoff.py` stub describes. The issue's
"three action classes" wording instead suggests exactly three columns. Raghav should confirm
the intended interpretation before implementation. Averaging all attacks into a single column
would also weaken what minimax protects against: it would protect against class averages.

For the recommended interpretation, assign each move to attack or non-damaging using the
shared class definitions. A belief's `move_probs` is the probability a move is present, not
the probability the opponent chooses it. Proposed initial within-class weights are proportional
to these inclusion probabilities, clearly labelled as a heuristic. Do not claim they are
calibrated action-selection predictions. Revealed moves can supplement candidates, but a
missing belief is not evidence that no other moves exist.

For class probability p(c), candidate weight w(j) and class members J(c):

    q(j) = p(c) * w(j) / sum(w(k) for k in J(c))
    q(switch) = p(switch)

Reject invalid class distributions. If positive class mass has no candidates or positive
weights, return a clear unavailable-table error rather than silently deleting and renormalising
that mass. The live caller then needs an explicit fallback. In particular, F5 cannot assume
uniform class predictions solve missing candidate information; its no-beliefs fallback needs
agreement. With `class_probs=None`, return `opp_probs=None` for F1; F2 requires an explicit
forecast. F5 may supply a uniform distribution over classes, not over individual columns.

## Proposed scoring rule

Use fractions of each affected Pokemon's maximum HP (health), not raw HP or percentages.
For a complete one-turn scenario, let L be HP lost and R be HP restored on each side:

    utility = L_opponent - L_us + R_us - R_opponent
              + k * (opponent_knocked_out - our_pokemon_knocked_out)

Start with k = 0.25 as a **proposed heuristic**, not a value learned from data. Show sensitivity
at k = 0 and k = 0.5 on development scenarios before choosing it. Do not use final test scores
to choose it. Count a knock-out once per affected Pokemon and ignore speculative future value.
In the implementation, put agreed shared policy constants in `config.py` with owner agreement.

F3 returns percent of estimated maximum HP. Divide by 100 before using it here. Cap each
damage event by the target's remaining HP, and each heal by its missing HP **at that point**.
State changes happen in action order. A combatant knocked out before its action cannot deal
its planned damage or heal. Use move priority before effective Speed; account for supported
Speed modifiers and reversed Speed order under Trick Room. Unsupported ordering effects
must be visible in diagnostics, not silently treated as ordinary Speed. Speed ties are
proposed as equal-probability order branches. Verify against the simulator before shipping.

**Chance.** A suggested first approximation is two equally weighted on-hit damage outcomes,
F3's minimum and maximum, plus a miss branch. This preserves F3's midpoint but gives only a
coarse knock-out estimate; it is not the true damage-roll distribution. A hit probability is
an input to the scenario calculation, not automatically the move's listed accuracy. Accuracy
boosts, weather, abilities and special accuracy rules need either supported handling or a
documented approximation. Do not multiply average damage by hit probability and then use
that reduced damage to decide whether a hit knocks out its target. Branch first, then score.
Critical hits and secondary effects remain proposed omissions, disclosed in explanations.

## Switches, status moves and missing information

| Situation | Proposed treatment or review decision |
|---|---|
| We switch, they attack | Replace our active Pokemon on the copy, apply supported entry effects, then calculate their attack against the incoming teammate. Do not charge a second arbitrary switch penalty. |
| They switch, we attack | Average over plausible revealed reserves under explicit within-switch weights. The unrevealed part of the team needs a separately disclosed policy; do not read it from another client. |
| Unknown switch target | Proposed conservative placeholder: zero credited outgoing damage, not a claimed immunity. Whether this bound should be mixed with revealed targets, and with what weights, needs review. |
| Switch representation | Agree the aggregate column's `Action.name` and explain that it is a modelled column, not an action to send to the server. Do not silently redefine the shared interface. |
| Hazards on entry | Subtract actual modelled entry damage. Begin with only verified hazard rules; boots, immunity and hazard knock-outs need simulator fixtures. Otherwise mark the scenario unsupported. |
| Both switch | No attack damage; supported entry effects still apply to each incoming Pokemon. |
| Healing | Credit capped HP recovery in action order. An earlier knock-out prevents the heal. |
| Protect or other blocking moves | Model supported prevention explicitly. Zero direct damage does not mean zero tactical effect. |
| Setup, poison, hazards and other longer-term effects | Do not invent bonuses initially. If scored as zero beyond immediate effects, label this as a known blind spot; Raghav must accept that scope before integration. |
| Recharge | No attack damage; the other side's action can still matter. |
| F3 error / missing incoming damage | Unavailable estimate, never zero. Do not silently drop a dangerous opponent column or change row order to hide it. |

The current F3 helper scores only our available moves against the current opponent. F4 will
also need incoming damage and alternative switch targets. Propose a shared calculation helper
inside Role F that accepts explicit combatants, moves and a copied scenario. Keep F3's public
wrapper compatible and wait for its review before that refactor. The existing damage calculator
does not by itself simulate a whole turn, hazards, switching, healing or all ordering effects.

## Worked scenarios for later executable tests

These are synthetic scoring examples, not claims about a particular real battle. Unless
specified otherwise, attacks hit, damage is fixed, and k = 0.25. All HP values below are
fractions of the affected Pokemon's maximum HP. The expected outputs can be calculated by hand.

| ID | Input | Expected result |
|---|---|---|
| T1: units | F3 returns 40%; opponent has 0.80 HP; no reply | L_opponent = 0.40; utility = +0.40. |
| T2: exchange | We deal 0.40 and take 0.25; neither faints | Utility = +0.15, whichever acts first. |
| T3: faster KO | Opponent has 0.30 HP; we deal 0.50 first; their planned attack would deal 0.60 | Their attack is cancelled; utility = 0.30 + 0.25 = +0.55. |
| T4: slower KO | We have 0.20 HP; opponent deals 0.30 first | Our attack is cancelled; utility = -0.20 - 0.25 = -0.45. |
| T5: Speed tie | We have 0.20 HP, opponent has 0.30; attacks deal 0.50 and 0.60; either side acts first with probability 0.50 | Expected utility = (0.55 - 0.45) / 2 = +0.05. |
| T6: miss then reply | T3 attack hits with probability 0.80; on a miss we take 0.10 and survive | Expected utility = 0.80 * 0.55 + 0.20 * (-0.10) = +0.42. |
| T7: uncertain KO | Opponent has 0.30 HP; on-hit outcomes are 0.20 or 0.40 equally; their reply is zero | Two-outcome approximation gives (0.20 + 0.30 + 0.25) / 2 = +0.375, not an asserted exact game probability. |
| T8: capped heal | We have 0.70 HP, heal 0.50 before taking 0.20 | Recovery is 0.30; utility = +0.10; final HP = 0.80. |
| T9: switch cost | Incoming teammate starts at 1.00 HP, loses 0.125 to entry hazards and 0.25 to an attack | Utility = -0.375, not damage against the outgoing teammate and not an extra switch penalty. |
| T10: class weights | Class masses attack/non-damaging/switch = 0.60/0.10/0.30; attack weights 3:1; one non-damaging move | Column probabilities = 0.45, 0.15, 0.10, 0.30. Sum = 1. |
| T11: immunity vs failure | A calculated immunity has zero damage; a different cell has an F3 error | Immunity may score zero; the failed table must remain visibly unavailable. |
| T12: missing candidates | Attack mass is 0.60 but no attack candidates are available | Explicit unavailable-table error; never redistribute the 0.60 to switching or status. |

Additional contract tests: all legal rows appear once; no illegal/duplicate tera options;
finite rectangular values; input battle, beliefs and forecasts unchanged; no private opponent
data accessed; no forecast needed for F1; F2 rejects an absent forecast. Test priority beating
Speed, Trick Room, current defensive types, status prevention, recharge and switch-entry
knock-outs against known simulator traces before claiming mechanics support.

## Review needed before implementation

1. Raghav: confirm concrete-move columns versus exactly three class columns, and the scope of
   tera, status utility, hazards and turn order for the first implementation.
2. Raghav and Role D/E: agree candidate priors, within-class weights and the missing-beliefs /
   unknown-switch policy. Inclusion probabilities must not be advertised as chosen-action probabilities.
3. Raghav: accept or revise the KO bonus and two-outcome chance approximation; define which
   unsupported cases use a labelled fallback rather than a fabricated payoff.
4. Rishi: review the worked examples, error paths, copy isolation and proposed test coverage.
5. Incorporate F1-F3 review feedback, then record accepted choices in `docs/decisions.md`.
   F4 remains open until implemented and a live table looks sensible to the domain reviewer.
