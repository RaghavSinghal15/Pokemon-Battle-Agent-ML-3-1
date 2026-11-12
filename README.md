# Pokémon Battle Agent (BITS F464 Machine Learning, 2026-27 Sem 1)

An **explainable AI agent for Pokémon Showdown Random Battles** (Gen 9), built for the
BITS F464 Machine Learning course project at BITS Pilani K.K. Birla Goa Campus.

The agent learns from recorded games between human players. On every turn it:

1. **reasons about hidden information** (what the opponent's Pokémon is probably holding),
2. **predicts what the opponent will do** (a calibrated classifier trained on human games),
3. **chooses a move** using expected value or game theory (linear programming),
4. **explains its decision** in plain text.

No Pokémon knowledge is needed to work on most of this repo. Start with
[`docs/README.md`](docs/README.md), which tells you what to read and in which order.

---

## Why this is hard (the 30-second version)

- **Both players move at the same time**, like rock-paper-scissors. There is often no single
  best move, only a good mix of moves.
- **The opponent's team is hidden** until it is revealed, like poker.
- **Outcomes are random**: damage varies, attacks miss, critical hits happen.
- **The game is mechanically huge**: 18 types, hundreds of characters, moves, abilities and
  items, all interacting.

Simple bots that always pick the strongest-looking attack lose badly to average human players.
The full explanation is in [`docs/01-project-explained.md`](docs/01-project-explained.md).

## Team

| Name | BITS ID | Role |
|---|---|---|
| Raghav Singhal | 2024ADPS0827G | E. Opponent model; domain reviewer; integration |
| Bhavsar Devam Jatin | 2024ADPS0831G | TBD |
| Rishi Kumar Motwani | 2024ADPS0838G | TBD |
| Yash Saxena | 2024ADPS0868G | TBD |
| Krrish Amit Bagaria | 2024A7PS0507G | TBD |
| Vyom Sehgal | 2024A3PS0347G | TBD |

Roles and what each one involves are described in [`docs/02-roles.md`](docs/02-roles.md).

## Quickstart

Full instructions (Windows, macOS, Linux) are in [`docs/07-setup.md`](docs/07-setup.md).
The short version:

```bash
# 1. This repo
git clone https://github.com/RaghavSinghal15/Pokemon-Battle-Agent-ML-3-1.git
cd Pokemon-Battle-Agent-ML-3-1
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 2. A local Pokémon Showdown server (in a separate folder, outside this repo)
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security    # leave this terminal running

# 3. Back in this repo, in a new terminal: check everything works
python scripts/check_setup.py
python scripts/play_baselines.py --games 20
```

If the last command prints win counts for two bots, your setup is complete.

## Repository layout

```
docs/                    Everything explained: project, roles, Pokémon primer, plan, viva prep
src/pokeagent/           The Python package
  interfaces.py          Shared data structures every module agrees on (read this first)
  config.py              Paths, format name, random seed, constants
  data/                  Role B: download, filter, store replays, train/test splits
  parser/                Role C: replay log -> public game state per turn
  beliefs/               Role D: probabilities over the opponent's hidden sets
  features/              Role D: game state -> row of numbers for the models
  models/                Role E: opponent model (logistic regression, LightGBM), calibration
  decision/              Role F: payoff table, best response, linear-programming strategy
  agent/                 Role F: the poke-env player that plays live, plus the explanation log
  eval/                  Role A: baselines, tournaments, win-rate statistics
scripts/                 Command-line entry points (setup check, baseline games, pipelines)
tests/                   pytest tests; every module has its own test file
notebooks/               Exploration only; see notebooks/README.md for the rules
data/                    Local data (git-ignored, never committed)
reports/                 Figures, result tables and the final report
.claude/agents/          AI helper agents (leakage auditor, viva examiner, Pokémon explainer)
AGENTS.md / CLAUDE.md    Rules for AI coding assistants working in this repo
CONTRIBUTING.md          How we work: branches, pull requests, reviews, AI usage
```

## Status

Version 1 is the complete, fully working core. Version 2 extensions start only after
version 1 is done. See [`docs/05-plan.md`](docs/05-plan.md).

| Milestone | Status |
|---|---|
| M0 Setup: local server, bots play each other | Not started |
| M1 Data: filtered replays, splits, data report | Not started |
| M2 Parser: validated, leakage-free states | Not started |
| M3 Beliefs and features | Not started |
| M4 Opponent model with calibration | Not started |
| M5 Agent plays full games, both strategies | Not started |
| M6 Evaluation: tournaments and ablations | Not started |
| M7 Report, explanation log, mock viva | Not started |

## Data and licences

- Replays: the Hugging Face dataset
  [`HolidayOugi/pokemon-showdown-replays`](https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays).
  It states no licence, so we use it for coursework only and **never commit or redistribute it**.
- Random Battle set data: Pokémon Showdown's source (`data/random-battles/gen9/sets.json`) and
  the public randbats stats.
- Libraries: [poke-env](https://github.com/hsahovic/poke-env) (MIT),
  [Pokémon Showdown](https://github.com/smogon/pokemon-showdown) (MIT).
