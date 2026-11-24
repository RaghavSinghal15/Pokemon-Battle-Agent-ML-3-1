"""Ablations (A6, #35): switch one component off at a time and measure what it was worth.

Every variant plays SimpleHeuristicsPlayer. Results are saved per chunk of games (so a long
run can be split), and the report adds up every chunk recorded for each variant.
"""

import json
import pickle
from contextlib import contextmanager
from pathlib import Path

from pokeagent.agent import player
from pokeagent.agent.player import ExplainableAgent
from pokeagent.config import (
    ARTIFACTS_DIR,
    OPPONENT_MODEL_PATH,
    REPORTS_DIR,
    VALUE_UTILITY_MODEL_PATH,
)
from pokeagent.decision import payoff
from pokeagent.decision._payoff_state import HiddenBoosts
from pokeagent.eval.stats import win_rate_ci

ABLATION_RESULTS_DIR = REPORTS_DIR / "results" / "ablations"
ABLATION_REPORT_PATH = REPORTS_DIR / "results" / "ablations.md"
NAIVE_MODEL_PATH = ARTIFACTS_DIR / "opponent_model_naive_hidden.pkl"

# name -> (what is switched off, agent settings). "full" is the reference.
VARIANTS = {
    "full": ("nothing (reference)", {}),
    "lp": ("strategy: minimax LP instead of best response", {"strategy": "lp"}),
    "hp_utility": ("V4: HP traded + KO bonus instead of change in P(win)", {"utility": "hp"}),
    "no_lasting_effects": ("F8-F10: setup, status and hazards valued at 0", {"effects": False}),
    "no_opponent_model": ("E: forecast = class frequencies", {"model": "naive"}),
    "no_belief_updates": ("D: beliefs stay at the Random Battle prior", {"beliefs": False}),
    "damage_only": ("everything: highest calculated damage", {"damage_only": True}),
    "no_tera": ("F14: we never terastallize", {"tera": False}),
    "no_attack_weighting": (
        "F15: their attacks weighted by belief only, not by damage to us",
        {"attack_weighting": False},
    ),
    "lp_no_stand_ins": (
        "F13: minimax, with no damage credit against unseen switch-ins",
        {"strategy": "lp", "stand_ins": False},
    ),
    "no_hidden_boosts": (
        "F16: unrevealed Scarf, Gale Wings and damage items ignored",
        {"hidden_boosts": False},
    ),
    "no_stall_guard": ("F18: no switch out of a stall", {"stall_guard": False}),
    # Not an ablation but an addition: F12, one more turn of lookahead (off by default).
    "two_turn": ("nothing; adds F12 two-turn search", {"search": True}),
    # Tuning (#120), not ablations: the tera cost the reference uses is PAYOFF_TERA_COST.
    "tera_cost_0.5": ("F17 tuning: tera cost 0.5 of a Pokemon's HP", {"tera_cost": 0.5}),
    "tera_cost_1.0": ("F17 tuning: tera cost 1.0 of a Pokemon's HP", {"tera_cost": 1.0}),
}


class DamageOnlyAgent(ExplainableAgent):
    """Always the move with the highest estimated damage (the agent's own fallback)."""

    def choose_move(self, battle):
        if battle.force_switch:
            return super().choose_move(battle)
        return self._fallback(battle, "damage-only ablation")


def _load(path: Path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def build_agent(variant: str, agent_class=None, **player_kwargs) -> ExplainableAgent:
    """The full agent with this variant's component switched off.

    agent_class replaces ExplainableAgent (dl.sequence_agent's GRU agent uses this)."""
    settings = VARIANTS[variant][1]
    model_path = NAIVE_MODEL_PATH if settings.get("model") == "naive" else OPPONENT_MODEL_PATH
    options = {
        "mode": "model",
        "opponent_model": _load(model_path),
        "strategy": settings.get("strategy", "best_response"),
        "belief_updates": settings.get("beliefs", True),
        "search": settings.get("search", False),
    }
    if settings.get("utility", "value") == "value":
        options.update(utility="value", value_model=_load(VALUE_UTILITY_MODEL_PATH))
    if agent_class is None:
        agent_class = DamageOnlyAgent if settings.get("damage_only") else ExplainableAgent
    return agent_class(**options, **player_kwargs)


@contextmanager
def lasting_effects(enabled: bool):
    """Temporarily set both lasting-effect weights to 0 (they are module settings)."""
    saved = payoff.PAYOFF_EFFECT_TURNS, payoff.PAYOFF_HAZARD_WEIGHT
    if not enabled:
        payoff.PAYOFF_EFFECT_TURNS = payoff.PAYOFF_HAZARD_WEIGHT = 0.0
    try:
        yield
    finally:
        payoff.PAYOFF_EFFECT_TURNS, payoff.PAYOFF_HAZARD_WEIGHT = saved


@contextmanager
def switched_off(variant: str):
    """Temporarily switch off the parts of this variant that are module settings, not agent
    options: lasting effects (F8-F10), our tera (F14), attack weighting (F15), minimax's
    stand-ins for unseen switch-ins (F13), hidden boosts (F16), the stall guard (F18) and the
    tera cost being tuned (F17). Everything is restored afterwards."""
    settings = VARIANTS[variant][1]
    saved = (
        payoff.our_tera_type,
        payoff.PAYOFF_ATTACK_TEMPERATURE,
        player.PAYOFF_STAND_INS_FOR,
        payoff.hidden_boosts,
        player.PAYOFF_STALL_TURNS,
        payoff.PAYOFF_TERA_COST,
    )
    if not settings.get("hidden_boosts", True):
        payoff.hidden_boosts = lambda mon, belief: HiddenBoosts()
    if not settings.get("stall_guard", True):
        player.PAYOFF_STALL_TURNS = 10**9  # never reached
    if "tera_cost" in settings:
        payoff.PAYOFF_TERA_COST = settings["tera_cost"]
    if not settings.get("tera", True):
        payoff.our_tera_type = lambda battle: None
    if not settings.get("attack_weighting", True):
        payoff.PAYOFF_ATTACK_TEMPERATURE = 0.0
    if not settings.get("stand_ins", True):
        player.PAYOFF_STAND_INS_FOR = ()
    try:
        with lasting_effects(settings.get("effects", True)):
            yield
    finally:
        (
            payoff.our_tera_type,
            payoff.PAYOFF_ATTACK_TEMPERATURE,
            player.PAYOFF_STAND_INS_FOR,
            payoff.hidden_boosts,
            player.PAYOFF_STALL_TURNS,
            payoff.PAYOFF_TERA_COST,
        ) = saved


def summarise(results_dir: Path = ABLATION_RESULTS_DIR) -> list[dict]:
    """Add up the saved chunks per variant, in VARIANTS order."""
    totals = {name: {"games": 0, "wins": 0, "commits": set()} for name in VARIANTS}
    for path in sorted(results_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        total = totals[record["variant"]]
        total["games"] += record["games"]
        total["wins"] += record["wins"]
        total["commits"].add(record["source"]["commit"][:7])
    rows = []
    for name, total in totals.items():
        if not total["games"]:
            continue
        rate, low, high = win_rate_ci(total["wins"], total["games"])
        rows.append(
            {
                "variant": name,
                "switched_off": VARIANTS[name][0],
                "games": total["games"],
                "wins": total["wins"],
                "win_rate": rate,
                "low": low,
                "high": high,
                "commits": sorted(total["commits"]),
            }
        )
    return rows


def markdown(rows: list[dict]) -> str:
    reference = next((r for r in rows if r["variant"] == "full"), None)
    lines = [
        "# Ablations against SimpleHeuristicsPlayer",
        "",
        "Each row switches one component off; 95% Wilson intervals. Generated by",
        "`python scripts/run_ablations.py --report`.",
        "",
        "| Variant | Switched off | Games | Wins | Win rate | 95% interval | vs full |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        change = ""
        if reference and r is not reference:
            change = f"{100 * (r['win_rate'] - reference['win_rate']):+.1f} pts"
        lines.append(
            f"| {r['variant']} | {r['switched_off']} | {r['games']} | {r['wins']} |"
            f" {r['win_rate']:.1%} | {r['low']:.1%} to {r['high']:.1%} | {change} |"
        )
    commits = sorted({c for r in rows for c in r["commits"]})
    lines += ["", f"Source commits: {', '.join(commits)}."]
    return "\n".join(lines) + "\n"
