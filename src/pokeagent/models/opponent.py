"""Training and using the opponent model.

The model reads the 29 features of one decision point (features/build.py) and returns the
probability of each action class (interfaces.ACTION_CLASSES). Training uses the train split;
model choices (e.g. regularisation strength) are made on validation. The test split is read
only by the final evaluation (E5), never here.

Label bias (E4, docs/08 check 4): about 10% of choices are hidden in the log (the Pokemon
fainted before acting, or couldn't move). Switches happen first, so they are never hidden, and
a model trained on visible choices only over-predicts switching. The fix: train with "hidden"
as a fourth class on all turns, then fold the hidden share back into attack and non-damaging
(fold_hidden). The predicted switch probability is then unbiased, because a switch is always
visible.
"""

import inspect

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pokeagent.config import FEATURES_PATH, HIDDEN_FEATURES_PATH, SEED
from pokeagent.features.build import FEATURE_NAMES
from pokeagent.interfaces import ACTION_CLASSES, FeatureRow

LABEL = "action_class"
HIDDEN = "hidden"  # label of a turn whose choice the log does not show
WITH_HIDDEN = (*ACTION_CLASSES, HIDDEN)
# Candidate inverse regularisation strengths for logistic regression, chosen on validation.
LR_C_GRID = (0.01, 0.1, 1.0, 10.0)
# LightGBM: leaves per tree to try, and the boosting settings shared by every try.
LGBM_LEAVES_GRID = (15, 63, 255)
LGBM_LEARNING_RATE = 0.05
LGBM_MAX_TREES = 3000
LGBM_EARLY_STOP = 50  # stop after this many trees without a validation improvement


def load_split(
    name: str,
    path=FEATURES_PATH,
    allow_test: bool = False,
    with_hidden: bool = False,
    hidden_path=HIDDEN_FEATURES_PATH,
) -> pd.DataFrame:
    """Rows of one split of the feature table. Reading "test" must be asked for explicitly.

    with_hidden=True adds that split's hidden-choice turns, labelled "hidden".
    """
    if name == "test" and not allow_test:
        raise PermissionError("The test split is used once, in the final evaluation (E5)")
    table = pd.read_parquet(path, filters=[("split", "==", name)])
    if with_hidden:
        hidden = pd.read_parquet(hidden_path, filters=[("split", "==", name)])
        if set(hidden[LABEL]) - {HIDDEN}:
            raise ValueError("The hidden-choice table must contain only hidden rows")
        table = pd.concat([table, hidden], ignore_index=True)
    missing = [c for c in [*FEATURE_NAMES, LABEL] if c not in table.columns]
    if missing:
        raise ValueError(f"Feature table is missing columns: {missing}")
    return table


def _x(table: pd.DataFrame) -> np.ndarray:
    return table[list(FEATURE_NAMES)].to_numpy(dtype=float)


def _y(table: pd.DataFrame, classes=ACTION_CLASSES) -> np.ndarray:
    """Labels as integers 0, 1, 2 (3 = hidden) in the order of `classes`."""
    index = {name: i for i, name in enumerate(classes)}
    labels = table[LABEL].map(index)
    if labels.isna().any():
        raise ValueError(f"Labels outside {classes}: {sorted(set(table[LABEL]) - set(classes))}")
    return labels.to_numpy(dtype=int)


def fold_hidden(probs: np.ndarray) -> np.ndarray:
    """Turn probabilities over (attack, non_damaging, switch, hidden) into the three classes.

    A switch is always visible, so P(switch) is the switch column as it is. A hidden choice
    was an attack or a non-damaging move; it is shared between them in the ratio the model
    gives the visible ones for the same state. (In bot games with known choices, 91% of
    hidden choices were attacks against 90% of visible non-switch choices: docs/08.)
    """
    attack, non_damaging, switch, hidden = probs.T
    visible = attack + non_damaging
    share = np.divide(attack, visible, out=np.full_like(attack, 0.5), where=visible > 0)
    folded = np.column_stack([attack + hidden * share, non_damaging + hidden * (1 - share), switch])
    return folded / folded.sum(axis=1, keepdims=True)


class OpponentModel:
    """Wraps a trained classifier so the rest of the system only sees probabilities."""

    def __init__(
        self,
        name: str,
        estimator=None,
        class_frequencies: np.ndarray | None = None,
        classes: tuple[str, ...] = ACTION_CLASSES,
    ):
        if (estimator is None) == (class_frequencies is None):
            raise ValueError("Give either a fitted estimator or class frequencies")
        if classes not in (ACTION_CLASSES, WITH_HIDDEN):
            raise ValueError(f"Unsupported classes {classes}")
        self.name = name
        self.estimator = estimator
        self.class_frequencies = class_frequencies
        self.classes = classes  # what it was trained on; it always answers in ACTION_CLASSES

    def __setstate__(self, state: dict) -> None:
        """Models saved before the hidden class existed were trained on the three classes."""
        state.setdefault("classes", ACTION_CLASSES)
        self.__dict__.update(state)

    def raw_proba_table(self, table: pd.DataFrame) -> np.ndarray:
        """Probabilities over the classes the model was trained on (hidden not yet folded)."""
        if self.estimator is None:
            return np.tile(self.class_frequencies, (len(table), 1))
        probs = self.estimator.predict_proba(_x(table))
        # scikit-learn orders columns by the sorted labels it saw; ours are 0, 1, 2(, 3).
        if list(self.estimator.classes_) != list(range(len(self.classes))):
            raise RuntimeError(f"The model was not trained on all of {self.classes}")
        return probs

    def predict_proba_table(self, table: pd.DataFrame) -> np.ndarray:
        """One row of probabilities per decision point, columns in ACTION_CLASSES order."""
        probs = self.raw_proba_table(table)
        return fold_hidden(probs) if self.classes == WITH_HIDDEN else probs

    def predict_proba(self, row: FeatureRow) -> dict[str, float]:
        """Probability of each action class, e.g. {"attack": 0.6, "non_damaging": 0.1,
        "switch": 0.3}. Sums to 1."""
        missing = [name for name in FEATURE_NAMES if name not in row]
        if missing:
            raise ValueError(f"Feature row is missing: {missing}")
        table = pd.DataFrame([{name: float(row[name]) for name in FEATURE_NAMES}])
        probs = self.predict_proba_table(table)[0]
        return {name: float(p) for name, p in zip(ACTION_CLASSES, probs, strict=True)}


def train_naive(train: pd.DataFrame, classes=ACTION_CLASSES) -> OpponentModel:
    """The minimum bar: always predict the training class frequencies, ignoring the state."""
    counts = np.bincount(_y(train, classes), minlength=len(classes))
    name = "naive (class frequencies" + (", hidden folded)" if HIDDEN in classes else ")")
    return OpponentModel(name, class_frequencies=counts / counts.sum(), classes=classes)


def evaluate_all_turns(model: OpponentModel, table: pd.DataFrame) -> dict:
    """Score the three-class forecast on every turn, visible and hidden.

    On a visible turn the score is -log P(what they did). On a hidden turn we only know it was
    not a switch, so the score is -log(1 - P(switch)). This is the log likelihood of what the
    log actually shows, so it does not reward the bias that dropping hidden turns creates.
    """
    probs = model.predict_proba_table(table)
    hidden = (table[LABEL] == HIDDEN).to_numpy()
    switch = ACTION_CLASSES.index("switch")
    visible_truth = _y(table[~hidden])
    p_visible = probs[~hidden][np.arange(len(visible_truth)), visible_truth]
    p_hidden = 1 - probs[hidden][:, switch]
    eps = 1e-15
    losses = -np.log(np.clip(np.concatenate([p_visible, p_hidden]), eps, 1))
    switched = (table[LABEL] == "switch").to_numpy()
    return {
        "model": model.name,
        "rows": len(table),
        "hidden_rows": int(hidden.sum()),
        "all_turns_log_loss": float(losses.mean()),
        "switch_log_loss": float(log_loss(switched, probs[:, switch], labels=[False, True])),
        "actual_switch_share": float(switched.mean()),
        "mean_predicted_switch": float(probs[:, switch].mean()),
    }


def evaluate(model: OpponentModel, table: pd.DataFrame) -> dict:
    """Log loss (the main metric), accuracy, macro-F1 and the confusion matrix.

    Visible turns only (hidden rows, if present, are left out); see evaluate_all_turns.
    """
    table = table[table[LABEL] != HIDDEN]
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


def _selection_loss(model: OpponentModel, val: pd.DataFrame) -> float:
    """Validation log loss over the classes the model was trained on (hidden included)."""
    return float(
        log_loss(
            _y(val, model.classes),
            model.raw_proba_table(val),
            labels=list(range(len(model.classes))),
        )
    )


def train_logistic_regression(
    train: pd.DataFrame, val: pd.DataFrame, c_grid=LR_C_GRID, classes=ACTION_CLASSES
) -> tuple[OpponentModel, list[dict]]:
    """Multinomial logistic regression on standardised features; C chosen by validation log loss.

    The scaler is part of the pipeline, so it is fitted on the training rows only.
    Returns the best model and one validation record per C tried.
    """
    x_train, y_train = _x(train), _y(train, classes)
    best, tried = None, []
    for c in c_grid:
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=2000, random_state=SEED),
        )
        pipeline.fit(x_train, y_train)
        suffix = ", hidden folded" if HIDDEN in classes else ""
        model = OpponentModel(
            f"logistic regression (C={c:g}{suffix})", estimator=pipeline, classes=classes
        )
        loss = _selection_loss(model, val)
        tried.append({"C": c, "val_log_loss": loss})
        if best is None or loss < best[1]:
            best = (model, loss)
    return best[0], tried


def train_lightgbm(
    train: pd.DataFrame, val: pd.DataFrame, leaves_grid=LGBM_LEAVES_GRID, classes=ACTION_CLASSES
) -> tuple[OpponentModel, list[dict]]:
    """Gradient-boosted trees (course topic: ensembles), multiclass.

    For each tree size, trees are added until validation log loss stops improving for
    LGBM_EARLY_STOP rounds (early stopping); the size with the best validation log loss wins.
    Trees need no feature scaling. Returns the best model and one record per size tried.
    """
    import lightgbm as lgb

    x_train, y_train, x_val, y_val = _x(train), _y(train, classes), _x(val), _y(val, classes)
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
        suffix = ", hidden folded" if HIDDEN in classes else ""
        model = OpponentModel(
            f"LightGBM ({leaves} leaves{suffix})", estimator=_Named(booster), classes=classes
        )
        loss = _selection_loss(model, val)
        tried.append({"leaves": leaves, "trees": booster.best_iteration_, "val_log_loss": loss})
        if best is None or loss < best[1]:
            best = (model, loss)
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
