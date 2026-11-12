# Plan

**Rule:** version 1 is a small, complete, impressive system. It must be fully finished,
tested, evaluated and explainable before anyone starts version 2 work. A complete small system
beats an incomplete big one, in the grade and in the viva.

## Version 1 milestones

Milestones are in dependency order, but most roles work in parallel from day 1 using fake
inputs (see [02-roles.md](02-roles.md)).

| # | Milestone | Owner | Done when |
|---|---|---|---|
| M0 | Setup | A | Every member runs `scripts/check_setup.py` and `scripts/play_baselines.py` successfully on their own laptop |
| M1 | Data | B | Filtered battles, `splits.json` and the data report are in place; a 5,000-battle sample exists for development |
| M2 | Parser | C | Replay and live parsing agree on 1,000 local games; the leakage test passes; decision points exist for the full training set. **Critical** |
| M3 | Beliefs and features | D | Belief tests pass; beliefs visibly sharpen over a game; the feature table is built and every feature is documented |
| M4 | Opponent model | E | Logistic regression and LightGBM both beat the naive baseline on validation; calibrated; results written up |
| M5 | Agent plays | F | The agent plays full games with both strategies, using beliefs and the opponent model, and beats random and max power. **Once this works, the project is safe** |
| M6 | Evaluation | A | 500+ games against each baseline with confidence intervals; ablations for beliefs, opponent model, and best response vs LP |
| M7 | Explain and present | all | Explanation log demo, failure analysis of lost games, report, and a full mock viva |

## Version 2 (only after M7)

Pick from these in roughly this order:

1. **Value model:** a classifier predicting who wins from a state, to give the payoff table a
   sense of the long term.
2. **Predicting exact moves** instead of action types.
3. **Explanation dashboard** in Streamlit with SHAP values.
4. **Public ladder games** against human players (check Showdown's current bot policy first).
5. **Behaviour cloning:** a model that imitates human choices, as another agent to compare.
6. **Deep learning comparison:** a small transformer over the turn history, as a contrast to
   the interpretable models.
7. **Opponent play styles:** clustering opponents (k-means or Gaussian mixtures) and
   visualising them with PCA.

## If something goes badly wrong

| Problem | Fallback |
|---|---|
| The dataset is unusable or too slow to download | Generate games locally by bot self-play, or use Metamon's parsed Gen 9 data |
| Writing a parser takes too long | Reuse poke-env's battle tracking; parse fewer event types |
| Beliefs are not ready | Use revealed information plus the most common role |
| LightGBM gives trouble | Logistic regression only |
| The LP is not working | Best response only; add LP afterwards |

## Weekly rhythm

- **One short sync a week:** each owner shows what works (running code, not slides), what is
  blocked, and what is next. Blockers get an owner and a fix before the call ends.
- **One owner explains their module to the team each week** as practice for the viva. Questions
  nobody could answer go into [06-viva-prep.md](06-viva-prep.md).
- Progress is tracked as GitHub issues, one per task, labelled by role and milestone.
