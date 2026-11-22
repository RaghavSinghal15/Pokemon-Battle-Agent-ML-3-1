"""Training and using the opponent model.

The model reads the 29 features of one decision point (features/build.py) and returns the
probability of each action class (interfaces.ACTION_CLASSES). Training uses the train split;
model choices (e.g. regularisation strength) are made on validation. The test split is read
only by the final evaluation (E5), never here.
"""

import inspect

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pokeagent.config import FEATURES_PATH, SEED
from pokeagent.features.build import FEATURE_NAMES
from pokeagent.interfaces import ACTION_CLASSES, FeatureRow

LABEL = "action_class"
# Candidate inverse regularisation strengths for logistic regression, chosen on validation.
LR_C_GRID = (0.01, 0.1, 1.0, 10.0)
# LightGBM: leaves per tree to try, and the boosting settings shared by every try.
LGBM_LEAVES_GRID = (15, 63, 255)
LGBM_LEARNING_RATE = 0.05
LGBM_MAX_TREES = 3000
LGBM_EARLY_STOP = 50  # stop after this many trees without a validation improvement


def load_split(name: str, path=FEATURES_PATH, allow_test: bool = False) -> pd.DataFrame:
    """Rows of one split of the feature table. Reading "test" must be asked for explicitly."""
    if name == "test" and not allow_test:
        raise PermissionError("The test split is used once, in the final evaluation (E5)")
    table = pd.read_parquet(path, filters=[("split", "==", name)])
    missing = [c for c in [*FEATURE_NAMES, LABEL] if c not in table.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    return table


def _x(table: pd.DataFrame) -> np.ndarray:
    return table[list(FEATURE_NAMES)].to_numpy(dtype=float)


def _y(table: pd.DataFrame) -> np.ndarray:
    """Labels as integers 0, 1, 2 in ACTION_CLASSES order."""
    index = {name: i for i, name in enumerate(ACTION_CLASSES)}
    return table[LABEL].map(index).to_numpy(dtype=int)


class OpponentModel:
    """Wraps a trained classifier so the rest of the system only sees probabilities."""

    def __init__(self, name: str, estimator=None, class_frequencies: np.ndarray | None = None):
        if (estimator is None) == (class_frequencies is None):
            raise ValueError("Give either a fitted estimator or class frequencies")
        self.name = name
        self.estimator = estimator
        self.class_frequencies = class_frequencies

    def predict_proba_table(self, table: pd.DataFrame) -> np.ndarray:
        """One row of probabilities per decision point, columns in ACTION_CLASSES order."""
        if self.estimator is None:
            return np.tile(self.class_frequencies, (len(table), 1))
        probs = self.estimator.predict_proba(_x(table))
        # scikit-learn orders columns by the sorted labels it saw; ours are 0, 1, 2.
        if list(self.estimator.classes_) != list(range(len(ACTION_CLASSES))):
            raise RuntimeError("The model was not trained on all three action classes")
        return probs

    def predict_proba(self, row: FeatureRow) -> dict[str, float]:
        """Probability of each action class, e.g. {"attack": 0.6, "non_damaging": 0.1,
        "switch": 0.3}. Sums to 1."""
        missing = [name for name in FEATURE_NAMES if name not in row]
        if missing:
            raise ValueError(f"Feature row is missing: {missing}")
        table = pd.DataFrame([{name: float(row[name]) for name in FEATURE_NAMES}])
        probs = self.predict_proba_table(table)[0]
        return {name: float(p) for name, p in zip(ACTION_CLASSES, probs, strict=True)}


def train_naive(train: pd.DataFrame) -> OpponentModel:
    """The minimum bar: always predict the training class frequencies, ignoring the state."""
    counts = np.bincount(_y(train), minlength=len(ACTION_CLASSES))
    return OpponentModel("naive (class frequencies)", class_frequencies=counts / counts.sum())


def evaluate(model: OpponentModel, table: pd.DataFrame) -> dict:
    """Log loss (the main metric), accuracy, macro-F1 and the confusion matrix."""
    probs = model.predict_proba_table(table)
    truth = _y(table)
    predicted = probs.argmax(axis=1)
    labels = list(range(len(ACTION_CLASSES)))
    return {
        "model": model.name,
        "rows": len(table),
        "log_loss": float(log_loss(truth, probs, labels=labels)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro")),
        # Rows: true class; columns: predicted class (both in ACTION_CLASSES order).
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "mean_predicted": {
            name: float(p) for name, p in zip(ACTION_CLASSES, probs.mean(axis=0), strict=True)
        },
    }


def train_logistic_regression(
    train: pd.DataFrame, val: pd.DataFrame, c_grid=LR_C_GRID
) -> tuple[OpponentModel, list[dict]]:
    """Multinomial logistic regression on standardised features; C chosen by validation log loss.

    The scaler is part of the pipeline, so it is fitted on the training rows only.
    Returns the best model and one validation record per C tried.
    """
    x_train, y_train = _x(train), _y(train)
    best, tried = None, []
    for c in c_grid:
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, random_state=SEED),
        )
        pipeline.fit(x_train, y_train)
        model = OpponentModel(f"logistic regression (C={c:g})", estimator=pipeline)
        record = evaluate(model, val)
        tried.append({"C": c, "val_log_loss": record["log_loss"]})
        if best is None or record["log_loss"] < best[1]:
            best = (model, record["log_loss"])
    return best[0], tried


def train_lightgbm(
    train: pd.DataFrame, val: pd.DataFrame, leaves_grid=LGBM_LEAVES_GRID
) -> tuple[OpponentModel, list[dict]]:
    """Gradient-boosted trees (course topic: ensembles), multiclass.

    For each tree size, trees are added until validation log loss stops improving for
    LGBM_EARLY_STOP rounds (early stopping); the size with the best validation log loss wins.
    Trees need no feature scaling. Returns the best model and one record per size tried.
    """
    import lightgbm as lgb

    x_train, y_train, x_val, y_val = _x(train), _y(train), _x(val), _y(val)
    best, tried = None, []
    for leaves in leaves_grid:
        booster = lgb.LGBMClassifier(
            objective="multiclass",
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
        x_val_named = pd.DataFrame(x_val, columns=list(FEATURE_NAMES))
        # LightGBM 4.6+ takes eval_X/eval_y and deprecates eval_set; older versions need eval_set.
        if "eval_X" in inspect.signature(booster.fit).parameters:
            validation = {"eval_X": (x_val_named,), "eval_y": (y_val,)}
        else:
            validation = {"eval_set": [(x_val_named, y_val)]}
        booster.fit(
            pd.DataFrame(x_train, columns=list(FEATURE_NAMES)),
            y_train,
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)],
            **validation,
        )
        model = OpponentModel(f"LightGBM ({leaves} leaves)", estimator=_Named(booster))
        record = evaluate(model, val)
        tried.append(
            {"leaves": leaves, "trees": booster.best_iteration_, "val_log_loss": record["log_loss"]}
        )
        if best is None or record["log_loss"] < best[1]:
            best = (model, record["log_loss"])
    return best[0], tried


class _Named:
    """Gives LightGBM its column names back, so it does not warn about unnamed inputs."""

    def __init__(self, booster):
        self.booster = booster
        self.classes_ = booster.classes_

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.booster.predict_proba(pd.DataFrame(x, columns=list(FEATURE_NAMES)))

    def feature_importance(self) -> dict[str, float]:
        """Total loss reduction ("gain") from splits on each feature, as shares of the total."""
        gains = self.booster.booster_.feature_importance(importance_type="gain")
        total = gains.sum()
        return {name: float(g / total) for name, g in zip(FEATURE_NAMES, gains, strict=True)}
