@AGENTS.md

## Claude Code specifics

Helper subagents live in `.claude/agents/`. Suggest them when they fit:

- **leakage-auditor**: before any pull request that touches `data/`, `parser/`, `features/` or
  `models/`. Reviews the diff for information leaking from the future or from labels.
- **viva-examiner**: when someone wants to test their understanding of a module. Asks
  examiner-style questions one at a time and gives feedback.
- **pokemon-explainer**: when a game mechanic comes up that the person doesn't know, or to check
  whether code handles a mechanic correctly.
