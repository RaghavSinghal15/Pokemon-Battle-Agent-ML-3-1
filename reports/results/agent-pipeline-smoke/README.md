# F6 public-pipeline smoke check

**Untrained fixed-thirds forecast. Not E5, not learned-model performance, and not completion of F6.**

Both strategies ran through the real C3 public parser, D2 beliefs, D4 features, model API and F4 table.

| Strategy | Completed games | Table choices | Explanations | Client errors |
|---|---:|---:|---:|---:|
| best_response | 100 | 2493 | 2699 | 0 |
| lp | 100 | 2950 | 3112 | 0 |

Every action produced one explanation. Thirty normal turns used explicit fallbacks for missing move evidence or unsupported action ordering; 338 forced replacements used the separate replacement rule.

## Saved inputs and demos

- [Summary, model calls and fallback counts](summary.json).
- Best response: [game record](best_response.json), [table and fallback explanations](best_response-demo.txt).
- Minimax: [game record](lp.json), [table and fallback explanations](lp-demo.txt).

## Reproduce and interpret

Command: `python scripts/check_agent_pipeline.py --server-dir ../pokemon-showdown`.
Evaluator: `ad4850f67c787b1f2eef359f5ad1c26653d38d74`, retained by tag `smoke-f6-2026-11-17`.
Showdown: `5c28e341ef9f9730a35874390cc3bc2aead7a16e`. Both tracked source flags were clean.
Seeds 464/465 control Python/NumPy actions, not server teams. The opponent was random.
Win counts are incidental smoke-run outcomes; no trained-model or comparative-performance claim is made.
The actual E5 model still needs 100 games per strategy with provenance and readable explanations before issue #39 can close.
See [F6 integration](../../../docs/f6-integration.md).
