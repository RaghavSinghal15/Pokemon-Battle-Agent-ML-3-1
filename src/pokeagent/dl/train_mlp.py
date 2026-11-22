"""Train the DL1 network and compare it with the classical opponent models on validation.

    pip install -e ".[dl]"
    python -m pokeagent.models.train          # first: the classical models it is compared with
    python -m pokeagent.dl.train_mlp

Writes artifacts/opponent_model_mlp.pkl and reports/results/dl_mlp_val.json. Test untouched.
"""

import json
import pickle
import time

from pokeagent.config import ARTIFACTS_DIR, REPORTS_DIR
from pokeagent.dl.mlp import train_mlp
from pokeagent.models.opponent import WITH_HIDDEN, evaluate, evaluate_all_turns, load_split

REPORT_PATH = REPORTS_DIR / "results" / "dl_mlp_val.json"
COMPARED = ("naive_hidden", "logistic_regression", "lightgbm_hidden")


def main() -> None:
    train, val = load_split("train", with_hidden=True), load_split("val", with_hidden=True)
    start = time.time()
    mlp, tried = train_mlp(train, val, WITH_HIDDEN)
    minutes = (time.time() - start) / 60
    with (ARTIFACTS_DIR / "opponent_model_mlp.pkl").open("wb") as stream:
        pickle.dump(mlp, stream)

    models = {"mlp": mlp}
    for key in COMPARED:
        path = ARTIFACTS_DIR / f"opponent_model_{key}.pkl"
        if path.is_file():
            with path.open("rb") as stream:
                models[key] = pickle.load(stream)
    results = {}
    for key, model in models.items():
        visible = evaluate(model, val)
        results[key] = {**evaluate_all_turns(model, val), "visible_only": visible}
    report = {"split": "val", "mlp_search": tried, "mlp_minutes": minutes, "results": results}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"MLP search ({minutes:.1f} min): {tried}")
    print(f"\n{'model':40} {'all turns':>9} {'accuracy':>9} {'macro-F1':>9} {'P(switch)':>10}")
    for key, r in results.items():
        v = r["visible_only"]
        print(
            f"{models[key].name:40} {r['all_turns_log_loss']:9.4f} {v['accuracy']:9.4f}"
            f" {v['macro_f1']:9.4f} {r['mean_predicted_switch']:10.4f}"
        )
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
