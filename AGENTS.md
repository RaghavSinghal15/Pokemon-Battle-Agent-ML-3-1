# Instructions for AI coding assistants

This file is read by AI coding assistants (Claude Code, Codex, Cursor, Copilot, Gemini and
others) working in this repository. Human contributors: see [CONTRIBUTING.md](CONTRIBUTING.md).

## The project

An explainable agent for Pokémon Showdown Gen 9 Random Battles, built by a team of six
third-year students for the BITS F464 Machine Learning course. The agent tracks beliefs about
the opponent's hidden team (Bayes' rule), predicts the opponent's next action type with a
calibrated classifier (logistic regression vs LightGBM) trained on human replays, picks a move
from a payoff table by best response or a minimax mixed strategy (linear programming), and
explains every decision. Read [docs/01-project-explained.md](docs/01-project-explained.md) for
the full picture and [docs/04-architecture.md](docs/04-architecture.md) for the design.

## Who you are helping, and how to behave

The people you are helping are **students who must defend every line of this project in an oral
exam (viva)** where the examiner cross-questions hard. Their understanding matters more than
your speed. Most of them have never played Pokémon.

1. **Find out which role the person owns** (A to F, see [docs/02-roles.md](docs/02-roles.md))
   if it isn't clear, and work inside that role's module.
2. **Explain before you code.** For any non-trivial task, first explain the concept and your
   plan in plain language, and check the person follows. Introduce every term, library and
   technique the first time it comes up. Then write the code.
3. **Go one step at a time.** Do the step that was asked; don't run ahead and implement the
   next three. Small, reviewable changes.
4. **Prefer simple, readable code** the person could rewrite themselves: explicit loops over
   clever one-liners, standard libraries over exotic ones. Comment the *why*.
5. **After writing code, summarise what it does and why** in two or three sentences the person
   could say in the viva, and point out anything they should double-check.
6. **Be honest about uncertainty.** If you are not sure how a game mechanic or a library API
   works, say so and check (see "Getting facts right" below). Never invent a mechanic or an API.

## Repository map

| Path | Role | Purpose |
|---|---|---|
| `src/pokeagent/interfaces.py` | all | Shared data structures. **The contract between modules.** |
| `src/pokeagent/config.py` | all | Paths, format, seed, constants |
| `src/pokeagent/data/` | B | Download, filter, split replays |
| `src/pokeagent/parser/` | C | Replay log / live battle → `PublicState` and `DecisionPoint` |
| `src/pokeagent/beliefs/` | D | Bayesian beliefs over opponent roles, moves, items, tera |
| `src/pokeagent/features/` | D | `PublicState` + beliefs → `FeatureRow` |
| `src/pokeagent/models/` | E | Opponent model, calibration |
| `src/pokeagent/decision/` | F | Payoff table, best response, LP |
| `src/pokeagent/agent/` | F | poke-env `Player` subclass, explanation log |
| `src/pokeagent/eval/` | A | Baselines, tournaments, statistics |
| `scripts/` | A | Command-line entry points |
| `tests/` | all | One test file per module |
| `docs/` | all | Explanations, plan, decisions, viva prep |

## Commands

```bash
pip install -e ".[dev]"                          # install (inside .venv)
ruff format . && ruff check .                     # format and lint
pytest                                            # fast tests (what CI runs)
pytest -m showdown                                # tests needing the local server
python scripts/check_setup.py                     # environment check
python scripts/play_baselines.py --games 20       # bots play each other
```

The local Showdown server runs separately: `node pokemon-showdown start --no-security` in the
pokemon-showdown folder, listening on `localhost:8000`. Node.js 22.18+ is required.

## Hard rules

- **No leakage.** A `PublicState` for turn N contains only what was visible before turn N's
  choices. Features never use the action taken, the turn's result or the winner. Splits come
  from `data/splits.json` only. Fit scalers and calibrators on train/validation, never test.
  Check every change against the leakage checklist in docs/04-architecture.md.
- **Public information only.** States hold only what a spectator could see, for both sides
  (reason in docs/04-architecture.md). Don't add our private team knowledge to model inputs.
- **Respect module boundaries.** Edit only the module of the role you are helping. Changing
  `interfaces.py` needs every affected owner's approval: tell the person, don't just do it.
- **Never commit data or models.** `data/` and `artifacts/` are git-ignored. Never upload or
  redistribute the replay dataset.
- **Version 1 scope.** No `torch`, `shap` or `streamlit` imports in version 1 code, and no new
  heavy dependencies without the team agreeing. See [docs/05-plan.md](docs/05-plan.md).
- **Constants in config.** No hard-coded paths, format names or seeds; use `config.py`.
- **Tests with every change.** Use small hand-built fake inputs (see `make_state()` in
  `tests/test_interfaces.py`) so tests don't depend on unfinished modules.
- **Record design choices** in [docs/decisions.md](docs/decisions.md) with the alternatives and
  the reason.

## Getting facts right

Pokémon mechanics and poke-env's API are easy to get confidently wrong. Before relying on one:

- **poke-env** (version 0.16+): read the installed source (`python -c "import poke_env;
  print(poke_env.__file__)"`) rather than assuming an API from memory. Battle objects are in
  `poke_env.battle`, players in `poke_env.player`, the damage calculator in
  `poke_env.calc.damage_calc_gen9`.
- **Random Battle sets**: `data/random-battles/gen9/sets.json` in the pokemon-showdown repo
  (roles with move pools, abilities and tera types per species; items are chosen in
  `data/random-battles/gen9/teams.ts`). Usage statistics: <https://pkmn.github.io/randbats/>.
- **Battle log protocol**: `sim/SIM-PROTOCOL.md` in the pokemon-showdown repo.
- **Mechanics**: <https://dex.pokemonshowdown.com> and <https://calc.pokemonshowdown.com>.
- Gen 9 Random Battles have no natures in play (all neutral); don't build features on them.

## Writing style

Docs, docstrings and comments are read by teammates from other roles and by the examiner:
plain language, short sentences, every term explained on first use, no unexplained jargon.
