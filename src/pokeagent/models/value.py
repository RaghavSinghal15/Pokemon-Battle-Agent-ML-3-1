"""Value model (V2, #72): P(this player wins) from a start-of-turn position.

Binary classification on the V1 table (features/value.py). Three models are compared on
validation log loss:
  team HP only     logistic regression on one number, our team HP minus theirs (a baseline)
  logistic         logistic regression on all value features, standardised
  LightGBM         gradient-boosted trees with early stopping
The test split is not touched here.
"""

import inspect

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pokeagent.config import SEED, VALUE_FEATURES_PATH
from pokeagent.features.value import LABEL, VALUE_FEATURE_NAMES
from pokeagent.models.calibration import expected_calibration_error, reliability_table
from pokeagent.models.opponent import (
    LGBM_EARLY_STOP,
    LGBM_LEARNING_RATE,
    LGBM_LEAVES_GRID,
    LGBM_MAX_TREES,
    LR_C_GRID,
)

TEAM_HP_LEAD = "team_hp_lead"


def load_value_split(name: str, path=VALUE_FEATURES_PATH, allow_test: bool = False):
    """One split of the value table. The test split needs explicit permission."""
    if name == "test" and not allow_test:
        raise PermissionError("The test split is used once, at the end (pass allow_test=True)")
    table = pd.read_parquet(path)
    missing = [c for c in [*VALUE_FEATURE_NAMES, LABEL, "split"] if c not in table.columns]
    if missing:
        raise ValueError(f"Value table is missing columns: {', '.join(missing)}")
    return table[table["split"] == name].reset_index(drop=True)


class ValueModel:
    """Wraps a fitted binary classifier and the columns it reads."""

    def __init__(self, name: str, estimator, columns: list[str]):
        self.name, self.estimator, self.columns = name, estimator, list(columns)

    def _x(self, table: pd.DataFrame) -> pd.DataFrame:
        table = table.copy()
        if TEAM_HP_LEAD in self.columns and TEAM_HP_LEAD not in table:
            table[TEAM_HP_LEAD] = table["my_team_hp"] - table["opp_team_hp"]
        return table[self.columns].astype(float)

    def win_probability_table(self, table: pd.DataFrame) -> np.ndarray:
        """P(win) for every row."""
        return self.estimator.predict_proba(self._x(table))[:, 1]

    def win_probability(self, row: dict) -> float:
        """P(win) for one feature row (a dict with every value feature)."""
        missing = [n for n in VALUE_FEATURE_NAMES if n not in row]
        if missing:
            raise ValueError(f"Feature row is missing {', '.join(missing)}")
        return float(self.win_probability_table(pd.DataFrame([row]))[0])


def evaluate_value(model: ValueModel, table: pd.DataFrame) -> dict:
    """Log loss, Brier score, accuracy at 0.5, ROC AUC and calibration error."""
    won = table[LABEL].to_numpy(dtype=int)
    p = model.win_probability_table(table)
    return {
        "rows": len(table),
        "log_loss": float(log_loss(won, p, labels=[0, 1])),
        "brier": float(brier_score_loss(won, p)),
        "accuracy": float(((p >= 0.5) == won).mean()),
        "auc": float(roc_auc_score(won, p)),
        "ece": expected_calibration_error(reliability_table(p, won)),
    }


def mirror_gap(model: ValueModel, table: pd.DataFrame) -> float:
    """Mean |P(p1 wins) + P(p2 wins) - 1| over turns with a row for both players.

    Exactly one player wins, so a consistent model gives two numbers that add up to 1.
    """
    frame = table[["battle_id", "turn", "perspective"]].copy()
    frame["p"] = model.win_probability_table(table)
    pairs = frame.pivot_table(index=["battle_id", "turn"], columns="perspective", values="p")
    pairs = pairs.dropna()
    if pairs.empty:
        return float("nan")
    return float((pairs["p1"] + pairs["p2"] - 1).abs().mean())


def _y(table: pd.DataFrame) -> np.ndarray:
    return table[LABEL].to_numpy(dtype=int)


def train_team_hp_baseline(train: pd.DataFrame) -> ValueModel:
    """One-feature logistic regression on the team HP lead: how far a simple count gets."""
    model = ValueModel("team HP lead only", None, [TEAM_HP_LEAD])
    model.estimator = LogisticRegression(random_state=SEED).fit(model._x(train), _y(train))
    return model


def train_value_logistic(train, val, c_grid=LR_C_GRID) -> tuple[ValueModel, list[dict]]:
    """Logistic regression on all value features; C chosen by validation log loss."""
    best, tried = None, []
    for c in c_grid:
        pipeline = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=2000, random_state=SEED)
        )
        model = ValueModel(f"logistic regression (C={c:g})", pipeline, VALUE_FEATURE_NAMES)
        pipeline.fit(model._x(train), _y(train))
        loss = evaluate_value(model, val)["log_loss"]
        tried.append({"C": c, "val_log_loss": loss})
        if best is None or loss < best[1]:
            best = (model, loss)
    return best[0], tried


def train_value_lightgbm(train, val, leaves_grid=LGBM_LEAVES_GRID) -> tuple[ValueModel, list]:
    """LightGBM with early stopping on validation log loss, for each tree size."""
    import lightgbm as lgb

    columns = list(VALUE_FEATURE_NAMES)
    x_train, x_val = train[columns].astype(float), val[columns].astype(float)
    best, tried = None, []
    for leaves in leaves_grid:
        booster = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=LGBM_MAX_TREES,
            learning_rate=LGBM_LEARNING_RATE,
            num_leaves=leaves,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=SEED,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        # LightGBM 4.6+ takes eval_X/eval_y and deprecates eval_set; older versions need eval_set.
        if "eval_X" in inspect.signature(booster.fit).parameters:
            validation = {"eval_X": (x_val,), "eval_y": (_y(val),)}
        else:
            validation = {"eval_set": [(x_val, _y(val))]}
        booster.fit(
            x_train,
            _y(train),
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)],
            **validation,
        )
        model = ValueModel(f"LightGBM ({leaves} leaves)", booster, columns)
        loss = evaluate_value(model, val)["log_loss"]
        tried.append({"leaves": leaves, "trees": booster.best_iteration_, "val_log_loss": loss})
        if best is None or loss < best[1]:
            best = (model, loss)
    return best[0], tried


def gain_shares(model: ValueModel) -> dict[str, float]:
    """LightGBM split gain per feature, as shares of the total, largest first."""
    gains = model.estimator.booster_.feature_importance(importance_type="gain")
    shares = {name: float(g / gains.sum()) for name, g in zip(model.columns, gains, strict=True)}
    return dict(sorted(shares.items(), key=lambda kv: -kv[1]))
