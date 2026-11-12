# Setup

By the end of this page you will have run your first bot-vs-bot games on your own laptop.
If anything fails, post the exact error in the group and tag the role A owner.

## 1. Install the prerequisites (once)

| Tool | Version | Check with | Get it from |
|---|---|---|---|
| Git | any recent | `git --version` | <https://git-scm.com/downloads> |
| Python | 3.11 or 3.12 (3.10+ works) | `python --version` | <https://www.python.org/downloads/> (Windows: tick **"Add python.exe to PATH"**) |
| Node.js | **22.18 or newer** (Showdown requires it) | `node --version` | <https://nodejs.org> (choose the current LTS) |

On Windows, use **Git Bash** or **PowerShell** for the commands below. Where a command differs
on Windows, both versions are shown.

## 2. Get this repo and its Python packages

```bash
git clone https://github.com/RaghavSinghal15/Pokemon-Battle-Agent-ML-3-1.git
cd Pokemon-Battle-Agent-ML-3-1

python -m venv .venv
source .venv/bin/activate        # macOS / Linux / Git Bash
# .venv\Scripts\Activate.ps1     # Windows PowerShell

pip install -e ".[dev]"
```

What these do:

- A **virtual environment** (`.venv`) is a private Python installation for this project, so
  its packages don't clash with anything else on your laptop. You must **activate** it (the
  `source` / `Activate.ps1` line) in every new terminal before working; your prompt then
  starts with `(.venv)`.
- `pip install -e ".[dev]"` installs everything listed in `pyproject.toml`, plus our own code
  in "editable" mode: changes you make in `src/` take effect immediately, without reinstalling.

## 3. Run a local Pokémon Showdown server

Clone it **next to** this repo, not inside it:

```bash
cd ..
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js     # Windows PowerShell: copy config\config-example.js config\config.js
node pokemon-showdown start --no-security
```

Leave this terminal open: the server runs as long as it does. It should print
`Test your server at http://localhost:8000`. `--no-security` turns off login checks so bots can
connect without accounts; it is only for local use.

You can open <http://localhost:8000> in a browser to watch games on your own server.

## 4. Check everything works

In a **new** terminal, inside this repo, with the virtual environment activated:

```bash
python scripts/check_setup.py
python scripts/play_baselines.py --games 20
python scripts/play_baselines.py --p1 heuristics --p2 max_power --games 100
pytest
```

Expected: every line of `check_setup.py` says `OK`, the baseline games print win counts (the
heuristics bot should win most games against max power), and `pytest` reports all tests passed.

**M0 is done for you when all four commands work.** Tell the group.

## 5. Optional: AI coding assistants

The repo has instructions for AI assistants in [`AGENTS.md`](../AGENTS.md) and
[`CLAUDE.md`](../CLAUDE.md), and three helper agents in `.claude/agents/`. Read
[`CONTRIBUTING.md`](../CONTRIBUTING.md#using-ai-tools) for the rules on using them.

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found on Windows | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python` |
| PowerShell refuses to run `Activate.ps1` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once |
| `npm install` fails or Showdown complains about the Node version | Install Node 22.18 or newer, open a new terminal, retry |
| `check_setup.py` says no server | The Showdown terminal must stay open and show "Test your server at http://localhost:8000" |
| Games hang and never finish | Restart the Showdown server; make sure only one server is running |
| `ModuleNotFoundError: pokeagent` | Activate the virtual environment, then `pip install -e ".[dev]"` again |
