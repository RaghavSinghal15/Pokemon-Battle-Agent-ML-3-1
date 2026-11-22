"""Train the opponent models and report them on validation.

    python -m pokeagent.models.train

Reads data/processed/features.parquet (train and validation only), writes the chosen model to
artifacts/ (git-ignored) and the validation report to reports/results/opponent_model_val.json
(counts and metrics only, safe to commit).
"""

import json
import pickle

import numpy as np
from sklearn.metrics import log_loss

from pokeagent.config import ARTIFACTS_DIR, REPORTS_DIR
from pokeagent.interfaces import ACTION_CLASSES
from pokeagent.models.opponent import (
    _y,
    evaluate,
    load_split,
    train_lightgbm,
    train_logistic_regression,
    train_naive,
)

REPORT_PATH = REPORTS_DIR / "results" / "opponent_model_val.json"
MODEL_PATHS = {
    "logistic_regression": ARTIFACTS_DIR / "opponent_model_lr.pkl",
    "lightgbm": ARTIFACTS_DIR / "opponent_model_lgbm.pkl",
}
# Game phases for the by-turn breakdown: switching is much more common early in a game.
TURN_BUCKETS = ((1, 1), (2, 5), (6, 15), (16, 30), (31, 10_000))


def by_turn(models, table) -> list[dict]:
    """Log loss of each model within each game phase."""
    rows = []
    for low, high in TURN_BUCKETS:
        part = table[(table["turn"] >= low) & (table["turn"] <= high)]
        if part.empty:
            continue
        truth = _y(part)
        row = {
            "turns": f"{low}-{high}" if high < 10_000 else f"{low}+",
            "rows": len(part),
            "switch_share": float((truth == ACTION_CLASSES.index("switch")).mean()),
        }
        for model in models:
            probs = model.predict_proba_table(part)
            row[model.name] = float(log_loss(truth, probs, labels=[0, 1, 2]))
        rows.append(row)
    return rows


def main() -> None:
    train, val = load_split("train"), load_split("val")
    print(f"train {len(train):,} rows, validation {len(val):,} rows")

    naive = train_naive(train)
    lr, tried = train_logistic_regression(train, val)
    for record in tried:
        print(f"  C={record['C']:<6g} validation log loss {record['val_log_loss']:.4f}")
    gbm, gbm_tried = train_lightgbm(train, val)
    for record in gbm_tried:
        print(
            f"  {record['leaves']:>3} leaves, {record['trees']:>4} trees:"
            f" validation log loss {record['val_log_loss']:.4f}"
        )

    models = (naive, lr, gbm)
    results = [evaluate(model, val) for model in models]
    report = {
        "split": "val",
        "classes": list(ACTION_CLASSES),
        "train_class_frequencies": dict(
            zip(ACTION_CLASSES, np.round(naive.class_frequencies, 4).tolist(), strict=True)
        ),
        "uniform_log_loss": float(np.log(len(ACTION_CLASSES))),
        "models": results,
        "lr_c_search": tried,
        "lightgbm_search": gbm_tried,
        "lightgbm_gain_share": dict(
            sorted(gbm.estimator.feature_importance().items(), key=lambda kv: -kv[1])
        ),
        "by_turn": by_turn(models, val),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, model in (("logistic_regression", lr), ("lightgbm", gbm)):
        with MODEL_PATHS[key].open("wb") as stream:
            pickle.dump(model, stream)

    print(f"\n{'model':38} {'log loss':>9} {'accuracy':>9} {'macro-F1':>9}")
    print(f"{'uniform (1/3 each)':38} {report['uniform_log_loss']:9.4f}")
    for r in results:
        print(f"{r['model']:38} {r['log_loss']:9.4f} {r['accuracy']:9.4f} {r['macro_f1']:9.4f}")
    for model, result in zip(models[1:], results[1:], strict=True):
        print(f"\nConfusion matrix of {model.name} (rows true, columns predicted):")
        for name, row in zip(ACTION_CLASSES, result["confusion_matrix"], strict=True):
            print(f"  {name:13} {row}")
    print("\nValidation log loss by game phase:")
    for row in report["by_turn"]:
        print(
            f"  turns {row['turns']:>6}  rows {row['rows']:>7,}  switch {row['switch_share']:.2f}"
            f"  naive {row[naive.name]:.4f}  LR {row[lr.name]:.4f}  LightGBM {row[gbm.name]:.4f}"
        )
    print("\nLightGBM: share of total gain per feature (top 10):")
    for name, share in list(report["lightgbm_gain_share"].items())[:10]:
        print(f"  {name:36} {share:.3f}")
    print(f"\nWrote {REPORT_PATH} and {', '.join(str(p) for p in MODEL_PATHS.values())}")


if __name__ == "__main__":
    main()
