"""Train the value models, compare them on validation and save the chosen one (V2, #72).

    python -m pokeagent.models.train_value

Reads data/processed/value_features.parquet (train and validation only). Writes every model
to artifacts/, the chosen one to artifacts/value_model.pkl, and the validation report to
reports/results/value_model_val.json (metrics only, safe to commit).
"""

import json
import pickle

from pokeagent.config import ARTIFACTS_DIR, REPORTS_DIR, VALUE_MODEL_PATH
from pokeagent.features.value import LABEL
from pokeagent.models.value import (
    evaluate_value,
    gain_shares,
    load_value_split,
    mirror_gap,
    train_team_hp_baseline,
    train_value_lightgbm,
    train_value_logistic,
)

REPORT_PATH = REPORTS_DIR / "results" / "value_model_val.json"
# Game phases: early positions are close to a coin flip, late ones are mostly decided.
TURN_BUCKETS = ((1, 5), (6, 15), (16, 30), (31, 10_000))


def main() -> None:
    train, val = load_value_split("train"), load_value_split("val")
    print(f"train {len(train):,} rows, validation {len(val):,} rows")
    models = {"team_hp": train_team_hp_baseline(train)}
    models["logistic_regression"], lr_tried = train_value_logistic(train, val)
    models["lightgbm"], gbm_tried = train_value_lightgbm(train, val)

    results = {
        key: {**evaluate_value(model, val), "mirror_gap": mirror_gap(model, val)}
        for key, model in models.items()
    }
    chosen = min(("logistic_regression", "lightgbm"), key=lambda k: results[k]["log_loss"])
    by_turn = []
    for low, high in TURN_BUCKETS:
        part = val[(val["turn"] >= low) & (val["turn"] <= high)]
        row = {"turns": f"{low}-{high}" if high < 10_000 else f"{low}+", "rows": len(part)}
        for key, model in models.items():
            row[key] = evaluate_value(model, part)["log_loss"]
        by_turn.append(row)
    report = {
        "split": "val",
        "base_rate": float(train[LABEL].mean()),
        "chosen": chosen,
        "results": results,
        "searches": {"logistic_regression": lr_tried, "lightgbm": gbm_tried},
        "lightgbm_gain_share": gain_shares(models["lightgbm"]),
        "by_turn": by_turn,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, model in models.items():
        with (ARTIFACTS_DIR / f"value_model_{key}.pkl").open("wb") as stream:
            pickle.dump(model, stream)
    with VALUE_MODEL_PATH.open("wb") as stream:
        pickle.dump(models[chosen], stream)

    print(
        f"\n{'model':32} {'log loss':>9} {'Brier':>7} {'acc':>6} {'AUC':>6} {'ECE':>6}"
        f" {'mirror':>7}"
    )
    print(f"{'coin flip (0.5)':32} {0.6931:9.4f} {0.25:7.4f}")
    for key, r in results.items():
        print(
            f"{models[key].name:32} {r['log_loss']:9.4f} {r['brier']:7.4f} {r['accuracy']:6.3f}"
            f" {r['auc']:6.3f} {r['ece']:6.3f} {r['mirror_gap']:7.3f}"
        )
    print("\nLog loss by game phase:")
    for row in by_turn:
        print(
            f"  turns {row['turns']:>6} rows {row['rows']:>7,}"
            + "".join(f"  {key} {row[key]:.4f}" for key in models)
        )
    top = list(report["lightgbm_gain_share"].items())[:8]
    print("\nLightGBM top features by gain: " + ", ".join(f"{k} {v:.0%}" for k, v in top))
    print(f"\nChosen: {models[chosen].name} -> {VALUE_MODEL_PATH}\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
