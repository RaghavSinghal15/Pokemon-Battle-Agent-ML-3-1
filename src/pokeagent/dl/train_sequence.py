"""Train the DL2 whole-game GRU, plus the same network with no memory (the ablation), and
compare with the per-turn models on validation.

    python -m pokeagent.dl.train_sequence

Writes artifacts/opponent_model_gru.pkl and reports/results/dl_sequence_val.json.
"""

import json
import pickle
import time

import pandas as pd

from pokeagent.config import ARTIFACTS_DIR, REPORTS_DIR
from pokeagent.dl.sequence import train_sequence_model
from pokeagent.models.opponent import evaluate, evaluate_all_turns, load_split

REPORT_PATH = REPORTS_DIR / "results" / "dl_sequence_val.json"
COMPARED = ("lightgbm_hidden", "mlp")


class _Precomputed:
    """Predictions made once on the whole validation table, looked up by row.

    evaluate() drops hidden rows before predicting; for the GRU that would also remove them
    from each game's history. Predicting on the full table first keeps the history intact.
    """

    def __init__(self, model, table):
        self.name = model.name
        self.probs = pd.DataFrame(model.predict_proba_table(table), index=table.index)

    def predict_proba_table(self, table):
        return self.probs.loc[table.index].to_numpy()


def main() -> None:
    train, val = load_split("train", with_hidden=True), load_split("val", with_hidden=True)
    models, records = {}, {}
    for history in ("full", "none"):
        start = time.time()
        model, record = train_sequence_model(train, val, history=history)
        record["minutes"] = (time.time() - start) / 60
        models[f"gru_{history}"], records[history] = model, record
        print(f"trained {model.name}: {record}")
    with (ARTIFACTS_DIR / "opponent_model_gru.pkl").open("wb") as stream:
        pickle.dump(models["gru_full"], stream)
    for key in COMPARED:
        path = ARTIFACTS_DIR / f"opponent_model_{key}.pkl"
        if path.is_file():
            with path.open("rb") as stream:
                models[key] = pickle.load(stream)

    results = {}
    for key, model in models.items():
        fixed = _Precomputed(model, val)
        results[key] = {**evaluate_all_turns(fixed, val), "visible_only": evaluate(fixed, val)}
    report = {"split": "val", "training": records, "results": results}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

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
