"""Opponent-model pipeline on a small synthetic feature table (E2, #15)."""

import numpy as np
import pandas as pd
import pytest

from pokeagent.features.build import FEATURE_NAMES
from pokeagent.interfaces import ACTION_CLASSES
from pokeagent.models.opponent import (
    evaluate,
    load_split,
    train_lightgbm,
    train_logistic_regression,
    train_naive,
)


def synthetic(rows: int, seed: int) -> pd.DataFrame:
    """Opponents switch when their type matchup is bad, otherwise mostly attack."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(rng.normal(size=(rows, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    bad = table["opp_type_matchup"] < -0.5
    labels = np.where(bad, "switch", np.where(rng.random(rows) < 0.8, "attack", "non_damaging"))
    table["action_class"] = labels
    return table


@pytest.fixture
def feature_file(tmp_path):
    parts = []
    for split, seed in (("train", 0), ("val", 1), ("test", 2)):
        part = synthetic(600, seed)
        part["split"] = split
        parts.append(part)
    path = tmp_path / "features.parquet"
    pd.concat(parts).to_parquet(path)
    return path


def test_test_split_needs_explicit_permission(feature_file):
    assert set(load_split("train", feature_file)["split"]) == {"train"}
    with pytest.raises(PermissionError):
        load_split("test", feature_file)
    assert len(load_split("test", feature_file, allow_test=True)) == 600


def test_naive_model_predicts_training_frequencies():
    train = synthetic(600, 0)
    naive = train_naive(train)
    expected = train["action_class"].value_counts(normalize=True)
    for name, p in naive.predict_proba(train.iloc[0].to_dict()).items():
        assert p == pytest.approx(expected.get(name, 0.0))


def test_logistic_regression_learns_what_naive_cannot():
    train, val = synthetic(600, 0), synthetic(600, 1)
    lr, tried = train_logistic_regression(train, val, c_grid=(0.1, 1.0))
    assert [t["C"] for t in tried] == [0.1, 1.0]
    assert evaluate(lr, val)["log_loss"] < evaluate(train_naive(train), val)["log_loss"] - 0.1


def test_predict_proba_returns_the_three_classes_summing_to_one():
    train = synthetic(600, 0)
    lr, _ = train_logistic_regression(train, train, c_grid=(1.0,))
    probs = lr.predict_proba(train.iloc[0][list(FEATURE_NAMES)].to_dict())
    assert list(probs) == list(ACTION_CLASSES)
    assert sum(probs.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="missing"):
        lr.predict_proba({"turn": 1.0})


def test_evaluate_reports_a_three_by_three_confusion_matrix():
    train = synthetic(600, 0)
    result = evaluate(train_naive(train), train)
    assert np.array(result["confusion_matrix"]).shape == (3, 3)
    assert result["accuracy"] == pytest.approx((train["action_class"] == "attack").mean())


def test_lightgbm_learns_and_records_its_search():
    train, val = synthetic(600, 0), synthetic(600, 1)
    gbm, tried = train_lightgbm(train, val, leaves_grid=(7,))
    assert tried[0]["leaves"] == 7 and tried[0]["trees"] >= 1
    assert evaluate(gbm, val)["log_loss"] < evaluate(train_naive(train), val)["log_loss"] - 0.1
    shares = gbm.estimator.feature_importance()
    assert sum(shares.values()) == pytest.approx(1.0)
    assert max(shares, key=shares.get) == "opp_type_matchup"  # the only informative feature
