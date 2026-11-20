# F6 integration: awaiting the trained E5 model

F6's code connects C3's live public parser, D2's belief updates, D4's feature builder and
the `OpponentModel.predict_proba(row)` contract to both decision strategies. It does not
implement or train Role E's classifier. The repository's E5 model remains a stub, so the
full F6 milestone is **not complete** and the PR remains a draft pending that dependency.

## Use a supplied model

```python
agent = ExplainableAgent(
    battle_format=BATTLE_FORMAT,
    mode="model",
    opponent_model=trained_model,
    strategy="best_response",  # or "lp"
    explanation_stream=stream,  # caller owns and closes the text stream
)
```

`mode="model"` requires an explicit model instance. Flat mode rejects one, so an intended
learned run cannot silently use thirds instead. The classifier receives only D4's public
feature row from C3's snapshot before the current turn's choices. Our private legal moves
and stats are used only by the payoff engine, never passed into model features. Forced
replacements bypass the simultaneous model/table and receive their own explanation.

An explanation names the battle and turn, public HP, role beliefs, the original three-class
forecast, feasible row expectations for best response, or row minima and mixture for minimax.
Minimax explicitly ignores the forecast. Its guarantee concerns the mixture's expected
payoff in this approximate table, not a win probability. Table/model failures have labelled
fallback explanations; unsupported programming errors remain visible.

The agent-side cosmetic-form adapter from F5 also conditions the base-species prior on
public observations when D's pending PR #48 returns empty cosmetic beliefs. No parser,
belief, feature, model or shared-interface source is rewritten. C's known Illusion team-count
limitation and F4's one-turn approximations remain; see their review notes and policy docs.

## Check the wiring without claiming a trained model

```sh
python scripts/check_agent_pipeline.py --server-dir ../pokemon-showdown
```

This runs 100 games with each strategy against random using an **untrained fixed-thirds test
predictor** through the real parser, beliefs and feature path. Every explanation is counted;
one table and one fallback example per strategy are retained with source and client-error
diagnostics. Results are explicitly labelled smoke-only. This is not a substitute for E5,
not model-performance evidence, and does not close issue #39.

After E5 is supplied: run 100 games per strategy with that actual model, preserve model and
source provenance, verify error/coverage counts and inspect explanation examples. Only then
mark F6 ready. The draft is stacked on `f/f6-foundation`, which combines the pending F5, C3
and D4 dependency branches without merging their PRs into main.
