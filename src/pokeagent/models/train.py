"""Train the opponent models, compare them on validation and save the chosen one.

    python -m pokeagent.models.train

Reads data/processed/features.parquet and hidden_features.parquet (train and validation only).
Writes every model to artifacts/ (git-ignored), the chosen one to artifacts/opponent_model.pkl,
the validation report to reports/results/opponent_model_val.json (metrics only, safe to commit)
and a switch reliability plot to reports/figures/opponent_switch_reliability.png.

The headline metric is log loss on ALL turns, visible and hidden (models.opponent:
evaluate_all_turns). Log loss on visible turns only is reported too, but it rewards the
label bias, so it is not used to choose the model.
"""

import json
import pickle

import numpy as np

from pokeagent.config import ARTIFACTS_DIR, OPPONENT_MODEL_PATH, REPORTS_DIR
from pokeagent.interfaces import ACTION_CLASSES
from pokeagent.models.calibration import expected_calibration_error, reliability_table
from pokeagent.models.opponent import (
    LABEL,
    WITH_HIDDEN,
    evaluate,
    evaluate_all_turns,
    load_split,
    train_lightgbm,
    train_logistic_regression,
    train_naive,
)

REPORT_PATH = REPORTS_DIR / "results" / "opponent_model_val.json"
FIGURE_PATH = REPORTS_DIR / "figures" / "opponent_switch_reliability.png"
# Game phases for the by-turn breakdown: switching is much more common early in a game.
TURN_BUCKETS = ((1, 1), (2, 5), (6, 15), (16, 30), (31, 10_000))


def by_turn(models, table) -> list[dict]:
    """All-turns log loss of each model within each game phase."""
    rows = []
    for low, high in TURN_BUCKETS:
        part = table[(table["turn"] >= low) & (table["turn"] <= high)]
        if part.empty:
            continue
        row = {
            "turns": f"{low}-{high}" if high < 10_000 else f"{low}+",
            "rows": len(part),
            "switch_share": float((part[LABEL] == "switch").mean()),
        }
        for key, model in models.items():
            row[key] = evaluate_all_turns(model, part)["all_turns_log_loss"]
        rows.append(row)
    return rows


def switch_reliability(models, table) -> dict:
    """Reliability of P(switch) on all turns: a switch is visible whenever it happens."""
    switched = (table[LABEL] == "switch").to_numpy()
    column = ACTION_CLASSES.index("switch")
    result = {}
    for key, model in models.items():
        rows = reliability_table(model.predict_proba_table(table)[:, column], switched)
        result[key] = {"ece": expected_calibration_error(rows), "bins": rows}
    return result


def plot_reliability(reliability: dict, names: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 0.7], [0, 0.7], color="grey", linestyle="--", label="perfect calibration")
    for key, data in reliability.items():
        xs = [row["mean_predicted"] for row in data["bins"]]
        ys = [row["observed_rate"] for row in data["bins"]]
        ax.plot(xs, ys, marker="o", label=f"{names[key]} (error {data['ece']:.3f})")
    ax.set_xlabel("Predicted probability of a switch")
    ax.set_ylabel("How often they actually switched")
    ax.set_title("Switch forecast on validation, all turns")
    ax.legend(fontsize=8)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train, val = load_split("train"), load_split("val")
    train_all = load_split("train", with_hidden=True)
    val_all = load_split("val", with_hidden=True)
    print(
        f"train {len(train):,} visible + {len(train_all) - len(train):,} hidden rows; "
        f"validation {len(val):,} + {len(val_all) - len(val):,}"
    )

    models = {"naive": train_naive(train), "naive_hidden": train_naive(train_all, WITH_HIDDEN)}
    models["logistic_regression"], lr_tried = train_logistic_regression(train, val)
    models["lightgbm"], gbm_tried = train_lightgbm(train, val)
    models["lightgbm_hidden"], gbm_hidden_tried = train_lightgbm(
        train_all, val_all, classes=WITH_HIDDEN
    )

    results = {}
    for key, model in models.items():
        visible = evaluate(model, val)
        results[key] = {**evaluate_all_turns(model, val_all), "visible_only": visible}
    chosen = min(
        ("logistic_regression", "lightgbm", "lightgbm_hidden"),
        key=lambda key: results[key]["all_turns_log_loss"],
    )
    reliability = switch_reliability(
        {key: models[key] for key in ("lightgbm", "lightgbm_hidden")}, val_all
    )
    report = {
        "split": "val",
        "classes": list(ACTION_CLASSES),
        "uniform_log_loss": float(np.log(len(ACTION_CLASSES))),
        "chosen": chosen,
        "results": results,
        "switch_reliability": reliability,
        "searches": {
            "logistic_regression": lr_tried,
            "lightgbm": gbm_tried,
            "lightgbm_hidden": gbm_hidden_tried,
        },
        "lightgbm_hidden_gain_share": dict(
            sorted(
                models["lightgbm_hidden"].estimator.feature_importance().items(),
                key=lambda kv: -kv[1],
            )
        ),
        "by_turn": by_turn(models, val_all),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, model in models.items():
        with (ARTIFACTS_DIR / f"opponent_model_{key}.pkl").open("wb") as stream:
            pickle.dump(model, stream)
    with OPPONENT_MODEL_PATH.open("wb") as stream:
        pickle.dump(models[chosen], stream)
    plot_reliability(reliability, {key: models[key].name for key in reliability})

    print(f"\n{'model':42} {'all turns':>9} {'visible':>8} {'accuracy':>9} {'P(switch)':>10}")
    print(f"{'uniform (1/3 each)':42} {'':>9} {report['uniform_log_loss']:8.4f}")
    for key, r in results.items():
        print(
            f"{models[key].name:42} {r['all_turns_log_loss']:9.4f}"
            f" {r['visible_only']['log_loss']:8.4f} {r['visible_only']['accuracy']:9.4f}"
            f" {r['mean_predicted_switch']:10.4f}"
        )
    print(f"(actual switch share on all turns: {results['naive']['actual_switch_share']:.4f})")
    for key, data in reliability.items():
        print(f"switch calibration error, {models[key].name}: {data['ece']:.4f}")
    print("\nAll-turns log loss by game phase:")
    for row in report["by_turn"]:
        print(
            f"  turns {row['turns']:>6}  rows {row['rows']:>7,}  switch {row['switch_share']:.2f}"
            + "".join(f"  {key} {row[key]:.4f}" for key in models)
        )
    print(f"\nChosen: {models[chosen].name} -> {OPPONENT_MODEL_PATH}")
    print(f"Wrote {REPORT_PATH} and {FIGURE_PATH}")


if __name__ == "__main__":
    main()
