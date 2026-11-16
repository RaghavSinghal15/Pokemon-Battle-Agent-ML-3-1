# F5 development acceptance: flat-forecast agent

These are measured local development games, not a trained-model result or the final frozen evaluation.

| Opponent | Wins | Losses | Ties | Games | Win rate | 95% Wilson interval |
|---|---:|---:|---:|---:|---:|---|
| random | 200 | 0 | 0 | 200 | 100.0% | [98.1%, 100.0%] |
| max_power | 177 | 23 | 0 | 200 | 88.5% | [83.3%, 92.2%] |

Both lower bounds exceed 50%, satisfying the F5 development criterion.

## Inputs and coverage

- Raw records: [random](vs-random.json), [max power](vs-max_power.json).
- [Summary and fallback counts](summary.json).
- 8,730 normal table decisions; three normal-turn greedy fallbacks for an unsupported action-order item.
- 1,103 forced replacements used the separately documented healthiest-reserve rule.
- No server-default orders were needed. All 400 games completed.

## Reproduction and limits

Command: `python scripts/evaluate_agent_v0.py --server-dir ../pokemon-showdown`.
Strategy: best response to a flat three-class forecast, with public set priors.
Evaluator: `55b6bb37dcebe18eb2253a371f6a2ffda5b0ccf1`, retained by tag `eval-f5-2026-11-17`.
Showdown: `5c28e341ef9f9730a35874390cc3bc2aead7a16e`. Both tracked source flags were clean.
Seeds: 464 (random) and 465 (max power). Seeds control Python/NumPy choices, not server teams.
The agent issued the challenges; directions were not balanced in this milestone check.
No human replay data or held-out test set was used. No model was fitted and no payoff constants were tuned.
The policy remains one-turn and approximate; see [F5 policy](../../../docs/f5-agent-v0.md) and [F4 limits](../../../docs/f4-implementation.md).
