# Decision log

Every design choice that someone might ask "why?" about, newest first. One entry per decision:
what we chose, what else we considered, and why. Keep entries short. These become the
"Design choices" section of the report and the answers to viva questions.

Template:

```
## YYYY-MM-DD: Short title (role, author)
**Decision:** what we chose.
**Alternatives:** what else we considered.
**Why:** the reason, with evidence (a number, a test, a plot) where possible.
```

---

## 2026-11-13: Initial architecture (all)
**Decision:** public-information states for both sides; split by battle with the newest
battles as test; three action classes for the opponent model; one-turn payoffs; forced
switches excluded; a walking-skeleton agent that always plays.
**Alternatives:** full player view for our own side; random turn-level split; exact-move
prediction; multi-turn search.
**Why:** see [04-architecture.md](04-architecture.md), section "Key design decisions".
