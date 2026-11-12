# Tests

Run all fast tests (no server or data needed), which is what CI runs on every pull request:

```bash
pytest
```

Run tests that need the local Showdown server or the downloaded data:

```bash
pytest -m showdown
pytest -m data
```

## One test file per module

| File | Owner | Must check at least |
|---|---|---|
| `test_interfaces.py` | everyone | shared structures serialise and round-trip |
| `test_data.py` | B | no battle id in two splits; filters drop what they claim to |
| `test_parser.py` | C | **leakage**: cutting the log after turn N leaves the turn-N state unchanged; replay and live parsing agree on the same game |
| `test_beliefs.py` | D | revealing a move unique to one role makes that role probability 1; probabilities sum to 1 |
| `test_features.py` | D | every row has exactly `FEATURE_NAMES`; no feature uses the action or the result |
| `test_models.py` | E | probabilities sum to 1; model beats the class-frequency baseline on a toy dataset |
| `test_decision.py` | F | LP gives 1/3 each on rock-paper-scissors; best response picks an obviously dominant row |
| `test_eval.py` | A | confidence interval is correct for known values |
| `test_live_server.py` | A, F | the agent finishes full games (needs the server) |

Build small fake inputs by hand (see `make_state()` in `test_interfaces.py`) so your tests do
not depend on another module being finished.
