# Contributing

How we work together on this repo. Short version: **branch, small pull request, one review,
tests pass, you can explain every line.**

## The workflow

1. **Pick or create an issue** for the task, assigned to you, labelled with your role and
   milestone. One issue = one piece of work someone could review in under 30 minutes.
2. **Create a branch** from the latest `main`:
   ```bash
   git checkout main
   git pull
   git checkout -b <role>/<short-description>      # e.g. parser/switch-events, data/splits
   ```
3. **Commit small and often**, with messages that say what the commit does:
   `Add Wilson confidence interval`, `Parse terastallize events`, not `update` or `fix stuff`.
4. **Run the checks before pushing:**
   ```bash
   ruff format .
   ruff check .
   pytest
   ```
5. **Push and open a pull request** (PR) into `main`, filling in the template. Link the issue
   (`Closes #12`).
6. **Get one approving review** from your reviewer (the table in
   [docs/02-roles.md](docs/02-roles.md#suggested-pairing)). Game-related changes also need the
   domain reviewer.
7. **Squash and merge** once approved and CI is green. Delete the branch.

**Never push directly to `main`.** Never force-push to a branch someone else is using.

## What a good pull request looks like

- Does one thing. Under ~400 changed lines where possible; split bigger work.
- Has tests for the new behaviour (see [tests/README.md](tests/README.md)).
- Changes only your own module, unless agreed with that module's owner.
- Updates the docs it affects: your module's docstrings, [docs/decisions.md](docs/decisions.md)
  for any design choice, and the README status table when a milestone completes.
- Fills in **"What I'd say in the viva"** in the template: two or three sentences, in your own
  words, explaining what the change does and why.

## Reviewing

Reviewing is part of the job, and the fastest way to learn the parts of the system you don't
own. As a reviewer:

- Run the code, don't only read it.
- Ask "why?" about anything unclear. If the author can't explain it, it isn't ready.
- Check the leakage checklist in [docs/04-architecture.md](docs/04-architecture.md) for
  anything touching data, parser, features or models.
- Approve, or request changes with specific, kind comments. Aim to review within a day.

## Changing shared interfaces

[`src/pokeagent/interfaces.py`](src/pokeagent/interfaces.py) is the contract between modules.
To change it: open a PR that changes only that file (plus the code it breaks), tag the owners of
every module that uses the field, and wait for all of them to approve.

## Code style

- Python 3.10+, formatted and linted with `ruff` (line length 100).
- Type hints on every public function. A docstring on every module and public function saying
  what it does and why, in plain language a teammate from another role can follow.
- Clear over clever: readable loops beat dense one-liners. Comment the *why*, not the *what*.
- No magic numbers or paths: constants go in [`config.py`](src/pokeagent/config.py).
- Seed everything random with `config.SEED`.
- Version 1 code must not import `torch`, `shap` or `streamlit`.

## Data, models and results

- Never commit data, trained models or large files. `data/` and `artifacts/` are git-ignored;
  share the script that makes them, not the files.
- Never upload or redistribute the replay dataset anywhere (no Drive links).
- Every number in the report must be reproducible: save results to `reports/results/` with the
  commit hash and seed used.
- Notebooks follow [notebooks/README.md](notebooks/README.md).

## Using AI tools

AI coding assistants (Claude, Copilot, ChatGPT, Cursor and others) are allowed and encouraged.
The repo includes instructions for them in [`AGENTS.md`](AGENTS.md) and
[`CLAUDE.md`](CLAUDE.md), which tell them to explain before coding and to respect module
boundaries. The rules:

1. **You own every line you merge.** "The AI wrote it" is not an answer in the viva or in a
   review. If you can't explain a line, don't merge it.
2. **Understand first, then generate.** Ask the assistant to explain the concept and the plan
   before it writes code. The examiner will ask about the concept, not the code.
3. **Verify everything it claims**, especially Pokémon mechanics and library APIs, which
   assistants often get confidently wrong. Check the dex, the poke-env source, or ask the
   domain reviewer.
4. **Keep it in your module.** Don't accept AI edits to other people's modules or to
   `interfaces.py` without following the rules above.
5. **Use the helper agents** in `.claude/agents/` (if you use Claude Code):
   - `leakage-auditor` before any PR touching data, parser, features or models,
   - `viva-examiner` to quiz yourself on your module,
   - `pokemon-explainer` when you hit a game mechanic you don't understand.

## Communication

- Questions about the code: comment on the issue or PR, so the answer stays findable.
- Quick coordination: the WhatsApp group.
- Blocked for more than a day: say so in the group and tag whoever can unblock you.
