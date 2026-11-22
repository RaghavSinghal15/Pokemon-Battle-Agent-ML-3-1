"""E6 (#79): predict the opponent's exact move, not just its action type.

For each decision where the opponent used a move, the candidates are the moves the beliefs
(D) allow for their active Pokemon, plus moves it has already revealed. Each (decision,
candidate) pair gets a row describing that move in this position; a LightGBM classifier scores
"was this the move they used", and the scores are normalised within each decision into a
distribution over the candidates. The baseline is the belief prior alone (inclusion
probabilities renormalised over the candidates).

If the move they used is not among the candidates (the beliefs ruled it out), the decision
counts as a miss: it is in the log loss with a small floor probability, and in "coverage".

    python -m pokeagent.models.exact_move            # build pairs, train, report (validation)
"""

import json
from dataclasses import replace
from functools import cache
from math import prod

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from poke_env.data import GenData, to_id_str

from pokeagent.beliefs.tracker import beliefs_for
from pokeagent.config import (
    DECISION_POINTS_PATH,
    MOVE_PAIRS_PATH,
    RANDOM_BATTLE_GENERATION,
    REPORTS_DIR,
    SEED,
    SPLITS_PATH,
)
from pokeagent.features.build import _PIVOT_MOVES, _base_types, _move_signals, _status_move
from pokeagent.features.export import _split_by_battle
from pokeagent.interfaces import PokemonView, PublicState

PAIRS_PATH = MOVE_PAIRS_PATH
REPORT_PATH = REPORTS_DIR / "results" / "exact_move_val.json"
FLOOR = 1e-4  # probability given to a used move that is not among the candidates
_HAZARDS = {"stealthrock", "spikes", "toxicspikes", "stickyweb"}
PAIR_FEATURES = (
    "belief_probability",
    "revealed",
    "used_last_turn",
    "is_attack",
    "base_power",
    "accuracy",
    "priority",
    "effectiveness",
    "stab",
    "is_setup",
    "is_recovery",
    "is_hazard",
    "is_protect",
    "is_pivot",
    "inflicts_status",
    "user_hp",
    "target_hp",
    "user_boost_sum",
    "turn",
    "n_candidates",
)


@cache
def _dex() -> GenData:
    return GenData.from_gen(RANDOM_BATTLE_GENERATION)


@cache
def _move_facts(move_id: str) -> dict | None:
    entry = _dex().moves.get(move_id)
    if entry is None:
        return None
    accuracy = entry.get("accuracy", True)
    setup, recovery = _move_signals(move_id)
    return {
        "is_attack": float(entry.get("category") != "Status"),
        "base_power": float(entry.get("basePower", 0)),
        "accuracy": 1.0 if accuracy is True else accuracy / 100,
        "priority": float(entry.get("priority", 0)),
        "type": entry.get("type", "").upper(),
        "is_setup": float(setup),
        "is_recovery": float(recovery),
        "is_hazard": float(move_id in _HAZARDS),
        "is_protect": float(bool(entry.get("stallingMove"))),
        "is_pivot": float(move_id in _PIVOT_MOVES),
        "inflicts_status": float(_status_move(move_id) is not None),
    }


def _effectiveness(move_type: str, target: PokemonView | None) -> float:
    dex = _dex()
    types = _base_types(dex, target)
    if target and target.terastallized and target.tera_type:
        types = (target.tera_type.upper(),)
    if not move_type or move_type not in dex.type_chart or not types:
        return 1.0
    return prod(dex.type_chart[t][move_type] for t in types if t in dex.type_chart)


def candidate_rows(state: PublicState, last_move: str | None) -> list[dict]:
    """One row per candidate move of `state.me.active` against `state.opp.active`.

    `state` is from the mover's side (me = the player whose move we predict). Their beliefs
    are built from the opponent's point of view, as in the opponent-model features.
    """
    user, target = state.me.active, state.opp.active
    if user is None:
        return []
    observer = replace(state, perspective=state.opp.player, me=state.opp, opp=state.me)
    belief = beliefs_for(observer).get(user.species)
    probs = dict(belief.move_probs) if belief else {}
    for move in user.revealed_moves:
        probs[move] = 1.0
    user_types = _base_types(_dex(), user)
    rows = []
    for move, probability in sorted(probs.items()):
        facts = _move_facts(move)
        if facts is None or probability <= 0:
            continue
        rows.append(
            {
                "move": move,
                "belief_probability": float(probability),
                "revealed": float(move in user.revealed_moves),
                "used_last_turn": float(move == last_move),
                **{k: v for k, v in facts.items() if k != "type"},
                "effectiveness": _effectiveness(facts["type"], target)
                if facts["is_attack"]
                else 1.0,
                "stab": float(facts["is_attack"] and facts["type"] in user_types),
                "user_hp": user.hp_fraction,
                "target_hp": target.hp_fraction if target else 1.0,
                "user_boost_sum": float(sum(user.boosts.values())),
                "turn": float(state.turn),
            }
        )
    for row in rows:
        row["n_candidates"] = float(len(rows))
    return rows


def build_pairs(decision_points=DECISION_POINTS_PATH, splits=SPLITS_PATH, out=PAIRS_PATH) -> int:
    """Write one row per (decision where a move was used, candidate move). Returns decisions."""
    split_of = _split_by_battle(splits)
    frames, decisions = [], 0
    columns = ["battle_id", "turn", "perspective", "state", "action"]
    with pq.ParquetFile(decision_points) as source:
        for batch in source.iter_batches(columns=columns):
            rows = []
            for record in batch.to_pylist():
                action = record["action"]
                if isinstance(action, str):
                    action = json.loads(action)
                if not action or action["kind"] != "move":
                    continue
                state = PublicState.from_dict(json.loads(record["state"]))
                last = state.my_last_action
                last_move = last.name if last is not None and last.kind == "move" else None
                decision = f"{record['battle_id']}:{record['turn']}:{record['perspective']}"
                candidates = candidate_rows(state, last_move)
                used = to_id_str(action["name"])
                for row in candidates:
                    row.update(decision=decision, chosen=row["move"] == used)
                if not any(row["chosen"] for row in candidates):
                    # The used move is not a candidate: keep a marker row so it counts as a miss.
                    candidates.append(
                        {"decision": decision, "move": used, "chosen": True, "missing": True}
                    )
                for row in candidates:
                    row["split"] = split_of[record["battle_id"]]
                rows.extend(candidates)
                decisions += 1
            frames.append(pd.DataFrame(rows))
    table = pd.concat(frames, ignore_index=True)
    table["missing"] = table.get("missing", False)
    table["missing"] = table["missing"].fillna(False).astype(bool)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out)
    return decisions


def _normalise(table: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    """Scores -> probabilities that sum to 1 within each decision (marker rows get FLOOR)."""
    scores = np.where(table["missing"].to_numpy(), 0.0, scores)
    decision = table["decision"]
    keys = decision.cat.codes.to_numpy() if decision.dtype == "category" else decision.to_numpy()
    sums = pd.Series(scores).groupby(keys).transform("sum").to_numpy()
    probs = np.divide(scores, sums, out=np.zeros_like(scores), where=sums > 0)
    return np.where(table["missing"].to_numpy(), FLOOR, np.maximum(probs, FLOOR))


def evaluate_moves(table: pd.DataFrame, probs: np.ndarray) -> dict:
    """Top-1 / top-3 accuracy, log loss of the used move, and candidate coverage."""
    frame = table[["decision", "chosen", "missing"]].assign(p=probs)
    frame["rank"] = frame.groupby("decision", observed=True)["p"].rank(
        ascending=False, method="first"
    )
    used = frame[frame["chosen"]]
    real = ~used["missing"]
    return {
        "decisions": int(len(used)),
        "coverage": float(real.mean()),
        "top1": float(((used["rank"] == 1) & real).mean()),
        "top3": float(((used["rank"] <= 3) & real).mean()),
        "log_loss": float(-np.log(used["p"]).mean()),
        "mean_candidates": float(frame.groupby("decision", observed=True).size().mean()),
    }


def train_and_report() -> dict:
    import lightgbm as lgb

    columns = [*PAIR_FEATURES, "decision", "chosen", "missing", "split"]
    # Test is never read here. float32 and categories keep 3.5 million rows small.
    table = pd.read_parquet(
        PAIRS_PATH, columns=columns, filters=[("split", "in", ["train", "val"])]
    )
    table[list(PAIR_FEATURES)] = table[list(PAIR_FEATURES)].astype("float32")
    table["decision"] = table["decision"].astype("category")
    train, val = table[table["split"] == "train"], table[table["split"] == "val"]
    fit_train, fit_val = train[~train["missing"]], val[~val["missing"]]
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(
        fit_train[list(PAIR_FEATURES)],
        fit_train["chosen"],
        eval_set=[(fit_val[list(PAIR_FEATURES)], fit_val["chosen"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    filled = val[list(PAIR_FEATURES)].fillna(0)
    scores = model.predict_proba(filled)[:, 1]
    results = {
        "uniform over candidates": evaluate_moves(val, _normalise(val, np.ones(len(val)))),
        "belief prior": evaluate_moves(
            val, _normalise(val, val["belief_probability"].fillna(0).to_numpy())
        ),
        "LightGBM on (state, move) pairs": evaluate_moves(val, _normalise(val, scores)),
    }
    gains = model.booster_.feature_importance(importance_type="gain")
    report = {
        "split": "val",
        "results": results,
        "trees": int(model.best_iteration_),
        "gain_share": {
            name: float(g / gains.sum())
            for name, g in sorted(zip(PAIR_FEATURES, gains, strict=True), key=lambda kv: -kv[1])
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    if not PAIRS_PATH.is_file():
        print(f"Built pairs for {build_pairs():,} decisions -> {PAIRS_PATH}")
    report = train_and_report()
    print(f"\n{'model':34} {'top-1':>7} {'top-3':>7} {'log loss':>9} {'coverage':>9}")
    for name, r in report["results"].items():
        print(
            f"{name:34} {r['top1']:7.3f} {r['top3']:7.3f} {r['log_loss']:9.4f} {r['coverage']:9.3f}"
        )
    top = list(report["gain_share"].items())[:6]
    print("Top features by gain: " + ", ".join(f"{k} {v:.0%}" for k, v in top))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
