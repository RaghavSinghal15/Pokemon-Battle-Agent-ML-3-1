# F4: provisional one-turn payoff policy

This is the implementation companion to [the design proposal](https://github.com/RaghavSinghal15/Pokemon-Battle-Agent-ML-3-1/pull/43).
It implements issue #33 for review; the choices below have **not** been approved by the domain
reviewer. F1/F2/F3 and the design proposal remain separate pending reviews. No change is made
to the shared dataclasses or the live agent. F5 integration and a win-rate claim are outside F4.

## What the builder returns

`build_payoff_table(battle, beliefs, class_probs)` retains its existing signature. Rows are
legal ordinary moves followed by legal switches, in the supplied order. Columns are sorted
candidate moves followed by an aggregate switch column, matching the original stub. This
uses concrete move columns rather than the issue's ambiguous three-column wording. New
voluntary tera rows are deferred; already changed types are retained in the calculation.

The switch column is `Action(kind="switch", name="__opponent_switch__")`. This is a label
inside the opponent model, never an order to send to Showdown. `Decision.chosen` still refers
only to our legal row actions. No opponent sentinel is added to training labels or features.

Candidate moves come from the supplied active-species inclusion beliefs and revealed moves
in our live view. Revealed moves get inclusion weight one. The candidate set is not claimed
to be exhaustive or guaranteed legal under every hidden restriction. No moves means an
unavailable table, even if a switch column could be made: absence of evidence cannot turn
an unknown attack into a harmless switch-only opponent.

Class mass is distributed in proportion to positive inclusion weights inside each class.
These weights describe move presence, **not** calibrated move-selection probabilities. Known
and possible unknown reserves contribute to the aggregate switch column with equal weight per
possible team member. The observed team-size metadata is used, defaulting to six. This is a
heuristic switch-target prior, not a prediction of a strong player's preferred replacement.
Unseen targets receive zero outgoing damage credit, a conservative bound rather than an
invented immunity or guessed species. Revealed reserves use their observed species and HP.

Invalid class distributions, positive mass with no candidates, impossible positive switch
mass, invalid state, and failed calculations raise `PayoffUnavailable`. Missing forecasts
produce `opp_probs=None` for F1. F2 needs an explicit distribution. No mass is silently
redistributed to compensate for missing candidates; F5 must choose a labelled legal fallback.

## Scoring and chance

The utility unit is a fraction of each affected Pokemon's maximum HP, treating team members
equally in that unit. It is a one-turn heuristic, not a probability of winning:

    opponent HP lost - our HP lost + our HP recovered - opponent HP recovered
        + PAYOFF_KO_BONUS * (opponent KO - our KO)

The default bonus is 0.25 in `config.py`. Unit examples compare 0, 0.25 and 0.5 without tuning
on held-out games. The design's twelve examples cover the core arithmetic and error contracts.

For each action, use a miss branch and two equally weighted on-hit damage endpoints from
the pinned calculator. This is deliberately not an exact average over all rolls. Branch on
order and hits **before** capping damage/recovery or awarding a KO bonus. An earlier KO
cancels the later action. Both sides' HP after supported entry damage is considered first.
All calculations use copied teams; missing combatant stats reuse F3's default-spread helper.
The incoming teammate, not the outgoing one, receives an opposing attack after a switch.

## Supported immediate effects and limits

| Area | Implemented | Known approximation or explicit fallback |
|---|---|---|
| Damage | Pinned poke-env calculator for both directions and chosen switch targets; KO/HP caps | Range endpoints and damage mechanics are evaluated at the scenario's starting state, not recalculated after each earlier event. HP-dependent power, drain, recoil, Life Orb self-damage, criticals and secondaries can be misvalued. |
| Order | Base move priority, Prankster status priority; Speed stages, Scarf, paralysis, Tailwind and common weather speed abilities; Trick Room; 50/50 Speed ties | Other ability/item/effect modifiers and integer rounding are not fully simulated. Recognized random/late-order items raise an unavailable-table error. |
| Accuracy | Listed accuracy plus accuracy/evasion stages; always-hit metadata; No Guard; paralysis action failure | Weather-specific accuracy, other abilities/items, confusion and conditional move failures are not fully simulated. Sleep/freeze actions raise an unavailable-table error. |
| Protection | Protect/Detect with a consecutive-use success approximation; blocks protectable damage unless bypassed | Other protection moves with contact punishment or other side effects are explicit unsupported cases. |
| Healing | Fixed recovery metadata, capped to missing HP when the action occurs | Weather-dependent/delayed recovery and other status utility are omitted. |
| Switch entry | Stealth Rock and Spikes with integer damage, type matchup, known Boots/Magic Guard and simple grounding | Sticky Web, Toxic Spikes, temporary grounding effects, Regenerator, other switch-in/out abilities and ability suppression are omitted. No extra arbitrary switch penalty is added. |
| Status/setup | Zero beyond supported immediate recovery/protection | Poisoning, stat boosts, newly laid hazards, blocking/forcing moves and long-term value are not simulated. Zero score is this policy's blind spot, not a claim these moves do nothing. |
| Special actions | Recharge contributes no attack damage; existing tera state affects typing | Charge-turn and self-KO attacks raise an unavailable-table error; new voluntary tera choices and pivot follow-up switches are deferred. |

Every successful build emits `PayoffApproximationWarning` summarising these omissions and
pointing here. Callers may filter that warning after explicitly accepting this policy, but
must not swallow `PayoffUnavailable` as zero damage. Unknown item/ability distributions are
not averaged yet; only tracked knowledge and the supplied move beliefs are used. As in F3,
default spreads and disguised/transformed opponents can be misestimated.

## Validation and how to review

Fast tests exercise the proposed arithmetic, real damage calls, forecast-to-column mapping,
priority/Speed/Trick Room, hazard immunities, existing tera typing, error propagation and input
isolation. `tests/test_payoff_live.py` follows heuristics play and builds tables only from
revealed moves; it reports supported tables, unavailable states and unexpected failures.
It does not train a model, access a peer's hidden information, or claim the observer plays F5.

Run `pytest` for fast tests and `pytest -m showdown -s` with the local server for live checks.
The printed table sample gives Raghav a concrete comparison to inspect, including switch
rows and immunities. Simulator traces for every omitted mechanic are not claimed to exist.
Before merging, reviewers should accept/revise the provisional policy, verify that a live
table makes game sense, and reconcile feedback from PRs #40-#43. Keep F4 open until then.
