"""DL3 (#78): classical vs deep opponent models on the test split, and in agent games.

    python -m pokeagent.dl.compare              # after E5 and the dl.play_games chunks

Test split: the classical models' numbers are read from E5's report (test_evaluation.json),
not scored again. The two deep models (DL1 MLP, DL2 GRU) are scored here, once, on the
same rows, with the same metrics and the same paired bootstrap over games against the
model the agent uses (LightGBM with the hidden class).

Training time: the whole fitting procedure, search included, on the same 2-CPU machine.
The classical times are measured here by rerunning the search on train and validation; the
deep ones were recorded when they were trained (dl_mlp_val.json, dl_sequence_val.json).

Games: the full agent with the MLP forecast (dl.play_games) against the same agent with
LightGBM (the "full" ablation chunks) and with class frequencies ("no_opponent_model"), all
against SimpleHeuristicsPlayer.

Writes reports/results/dl_comparison.json and dl_comparison.md (metrics only).
"""

import json
import pickle
import time
from math import erf, sqrt

import numpy as np
import pandas as pd

from pokeagent.config import ARTIFACTS_DIR, REPORTS_DIR
from pokeagent.dl.play_games import GAMES_DIR
from pokeagent.eval.ablations import ABLATION_RESULTS_DIR
from pokeagent.eval.stats import win_rate_ci
from pokeagent.interfaces import ACTION_CLASSES
from pokeagent.models.calibration import expected_calibration_error, reliability_table
from pokeagent.models.evaluate_test import REPORT_JSON as E5_REPORT
from pokeagent.models.evaluate_test import paired_bootstrap
from pokeagent.models.opponent import (
    LABEL,
    WITH_HIDDEN,
    all_turn_losses,
    evaluate,
    evaluate_all_turns,
    load_split,
    train_lightgbm,
    train_logistic_regression,
)

REPORT_JSON = REPORTS_DIR / "results" / "dl_comparison.json"
REPORT_MD = REPORTS_DIR / "results" / "dl_comparison.md"
REFERENCE = "lightgbm_hidden"  # the model the agent uses
DEEP = {"mlp": "opponent_model_mlp", "gru": "opponent_model_gru"}
# How a viva examiner could get a reason out of each model for one prediction.
EXPLAINABILITY = {
    "naive_hidden": "trivial: the same frequencies every turn",
    "logistic_regression": "high: one weight per feature and class, readable directly",
    "lightgbm": "medium: gain per feature overall, tree paths per prediction",
    REFERENCE: "medium: gain per feature overall, tree paths per prediction",
    "mlp": "low: 2 hidden layers, needs post-hoc tools (e.g. SHAP) per prediction",
    "gru": "lowest: depends on the whole game so far through a recurrent state",
}


class _Precomputed:
    """Predictions made once on the whole table, looked up by row index.

    evaluate() drops hidden rows before predicting; for the GRU that would also cut them out
    of each game's history, so it predicts on the full table first (as in train_sequence).
    """

    def __init__(self, model, table):
        self.name = model.name
        self.probs = pd.DataFrame(model.predict_proba_table(table), index=table.index)

    def predict_proba_table(self, table):
        return self.probs.loc[table.index].to_numpy()


def _load(name: str):
    with (ARTIFACTS_DIR / f"{name}.pkl").open("rb") as stream:
        return pickle.load(stream)


def deep_test_results() -> tuple[dict, dict]:
    """Test metrics of the two deep models, and their paired bootstrap against REFERENCE."""
    test_all = load_split("test", allow_test=True, with_hidden=True)
    games = test_all["battle_id"].to_numpy()
    switched = (test_all[LABEL] == "switch").to_numpy()
    column = ACTION_CLASSES.index("switch")
    reference_losses = all_turn_losses(_load(f"opponent_model_{REFERENCE}"), test_all)
    results, bootstrap = {}, {}
    for key, artifact in DEEP.items():
        fixed = _Precomputed(_load(artifact), test_all)
        results[key] = {
            **evaluate_all_turns(fixed, test_all),
            "visible_only": evaluate(fixed, test_all),
            "switch_ece": expected_calibration_error(
                reliability_table(fixed.predict_proba_table(test_all)[:, column], switched)
            ),
        }
        bootstrap[f"{REFERENCE} - {key}"] = paired_bootstrap(
            reference_losses, all_turn_losses(fixed, test_all), games
        )
    return results, bootstrap


def classical_minutes() -> dict:
    """Rerun the classical searches (train + validation only) to time them."""
    train, val = load_split("train"), load_split("val")
    train_all = load_split("train", with_hidden=True)
    val_all = load_split("val", with_hidden=True)
    minutes = {}
    start = time.time()
    train_logistic_regression(train, val)
    minutes["logistic_regression"] = (time.time() - start) / 60
    start = time.time()
    train_lightgbm(train_all, val_all, classes=WITH_HIDDEN)
    minutes[REFERENCE] = (time.time() - start) / 60
    return minutes


def deep_minutes() -> dict:
    mlp = json.loads((REPORTS_DIR / "results" / "dl_mlp_val.json").read_text(encoding="utf-8"))
    gru = json.loads((REPORTS_DIR / "results" / "dl_sequence_val.json").read_text("utf-8"))
    return {"mlp": mlp["mlp_minutes"], "gru": gru["training"]["full"]["minutes"]}


def two_proportion_p(wins_a: int, games_a: int, wins_b: int, games_b: int) -> float:
    """Two-sided p-value of a pooled two-proportion z-test."""
    pooled = (wins_a + wins_b) / (games_a + games_b)
    se = sqrt(pooled * (1 - pooled) * (1 / games_a + 1 / games_b))
    z = (wins_a / games_a - wins_b / games_b) / se
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def _totals(paths) -> tuple[int, int]:
    games = wins = 0
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        games, wins = games + record["games"], wins + record["wins"]
    return games, wins


def game_results(games_dir=GAMES_DIR, ablation_dir=ABLATION_RESULTS_DIR) -> dict:
    rows = {
        "mlp": _totals(sorted(games_dir.glob("mlp-*.json"))),
        REFERENCE: _totals(sorted(ablation_dir.glob("full-*.json"))),
        "naive_hidden": _totals(sorted(ablation_dir.glob("no_opponent_model-*.json"))),
    }
    result = {}
    for key, (games, wins) in rows.items():
        if games == 0:
            continue
        rate, low, high = win_rate_ci(wins, games)
        result[key] = {"games": games, "wins": wins, "win_rate": rate, "ci95": [low, high]}
    if "mlp" in result and REFERENCE in result:
        r = result[REFERENCE]
        m = result["mlp"]
        result["mlp_vs_reference_p"] = two_proportion_p(
            m["wins"], m["games"], r["wins"], r["games"]
        )
    return result


def markdown(report: dict) -> str:
    test, minutes = report["test"], report["training_minutes"]
    names = {
        "naive_hidden": "Naive (class frequencies)",
        "logistic_regression": "Logistic regression",
        "lightgbm": "LightGBM",
        REFERENCE: "LightGBM, hidden class (used)",
        "mlp": "MLP (DL1)",
        "gru": "GRU over the game (DL2)",
    }
    lines = [
        "# Classical vs deep opponent models (DL3)",
        "",
        "Test split, scored once; all-turns log loss is the headline metric (uniform guess",
        f"{np.log(3):.4f}). Generated by `python -m pokeagent.dl.compare`.",
        "",
        "| Model | All-turns log loss | Accuracy | Macro F1 | Switch ECE | Training (min)"
        " | Explainability |",
        "|---|---|---|---|---|---|---|",
    ]
    for key, name in names.items():
        r = test[key]
        v = r["visible_only"]
        train = f"{minutes[key]:.1f}" if key in minutes else "-"
        lines.append(
            f"| {name} | {r['all_turns_log_loss']:.4f} | {v['accuracy']:.3f} |"
            f" {v['macro_f1']:.3f} | {r['switch_ece']:.3f} | {train} | {EXPLAINABILITY[key]} |"
        )
    lines += ["", "Paired bootstrap over games (positive = the used LightGBM is worse):", ""]
    for name, b in report["paired_bootstrap"].items():
        low, high = b["ci95"]
        lines.append(f"- {name}: {b['difference']:+.4f} (95% interval {low:+.4f} to {high:+.4f})")
    games = report["games"]
    lines += [
        "",
        "## Agent games against SimpleHeuristicsPlayer (best response, everything else equal)",
        "",
        "| Opponent model | Games | Wins | Win rate | 95% interval |",
        "|---|---|---|---|---|",
    ]
    for key in ("mlp", REFERENCE, "naive_hidden"):
        if key in games:
            g = games[key]
            low, high = g["ci95"]
            lines.append(
                f"| {names[key]} | {g['games']} | {g['wins']} | {g['win_rate']:.1%} |"
                f" {low:.1%} to {high:.1%} |"
            )
    if "mlp_vs_reference_p" in games:
        lines += [
            "",
            f"MLP vs LightGBM: two-proportion z-test p = {games['mlp_vs_reference_p']:.2f}.",
        ]
    lines += [
        "",
        "The GRU is not played: it needs the opponent's earlier actions as a sequence, which the",
        "live agent does not build (docs/decisions.md, DL3).",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    e5 = json.loads(E5_REPORT.read_text(encoding="utf-8"))["opponent_model"]
    test = {key: e5["results"][key] for key in ("naive_hidden", "logistic_regression", "lightgbm")}
    test[REFERENCE] = e5["results"][REFERENCE]
    deep, bootstrap = deep_test_results()
    test.update(deep)
    bootstrap = {**{k: v for k, v in e5["paired_bootstrap_all_turns"].items()}, **bootstrap}
    report = {
        "split": "test",
        "test": test,
        "paired_bootstrap": bootstrap,
        "training_minutes": {**classical_minutes(), **deep_minutes()},
        "games": game_results(),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    REPORT_MD.write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
