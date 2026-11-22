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


# --- E4: hidden-choice label bias ---------------------------------------------------------


def test_fold_hidden_keeps_switch_and_shares_hidden_by_visible_ratio():
    from pokeagent.models.opponent import fold_hidden

    folded = fold_hidden(np.array([[0.45, 0.15, 0.2, 0.2]]))[0]
    # Switch is untouched; the 0.2 hidden splits 3:1 like attack 0.45 : non-damaging 0.15.
    assert folded == pytest.approx([0.45 + 0.15, 0.15 + 0.05, 0.2])
    # With no visible move mass, hidden is shared evenly.
    assert fold_hidden(np.array([[0.0, 0.0, 0.5, 0.5]]))[0] == pytest.approx([0.25, 0.25, 0.5])


def with_hidden_rows(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Hide some non-switch choices, as a log does when a Pokemon faints before acting."""
    rng = np.random.default_rng(seed)
    table = table.copy()
    hide = (table["action_class"] != "switch").to_numpy() & (rng.random(len(table)) < 0.3)
    table.loc[hide, "action_class"] = "hidden"
    return table


def test_hidden_class_removes_the_switch_bias():
    from pokeagent.models.opponent import WITH_HIDDEN, evaluate_all_turns

    truth = synthetic(4000, 3)
    train, val = with_hidden_rows(synthetic(4000, 0), 1), with_hidden_rows(truth, 2)
    biased = train_naive(train[train["action_class"] != "hidden"])
    fixed = train_naive(train, WITH_HIDDEN)
    true_share = (truth["action_class"] == "switch").mean()
    biased_result, fixed_result = evaluate_all_turns(biased, val), evaluate_all_turns(fixed, val)
    assert biased_result["mean_predicted_switch"] > true_share + 0.05
    assert fixed_result["mean_predicted_switch"] == pytest.approx(true_share, abs=0.02)
    assert fixed_result["all_turns_log_loss"] < biased_result["all_turns_log_loss"]


def test_load_split_adds_hidden_rows_only_when_asked(tmp_path):
    visible = synthetic(50, 0).assign(split="train")
    hidden = synthetic(20, 1).assign(split="train", action_class="hidden")
    visible.to_parquet(tmp_path / "features.parquet")
    hidden.to_parquet(tmp_path / "hidden.parquet")
    kwargs = {"path": tmp_path / "features.parquet", "hidden_path": tmp_path / "hidden.parquet"}
    assert len(load_split("train", **kwargs)) == 50
    both = load_split("train", with_hidden=True, **kwargs)
    assert len(both) == 70 and (both["action_class"] == "hidden").sum() == 20
    hidden.assign(action_class="attack").to_parquet(tmp_path / "hidden.parquet")
    with pytest.raises(ValueError, match="only hidden"):
        load_split("train", with_hidden=True, **kwargs)


def test_reliability_table_and_calibration_error():
    from pokeagent.models.calibration import expected_calibration_error, reliability_table

    probs = np.repeat([0.1, 0.9], 50)
    happened = np.concatenate([np.repeat([1, 0], [5, 45]), np.repeat([1, 0], [45, 5])])
    table = reliability_table(probs, happened, bins=2)
    assert [round(row["observed_rate"], 2) for row in table] == [0.1, 0.9]
    assert expected_calibration_error(table) == pytest.approx(0.0)
    overconfident = reliability_table(np.repeat([0.0, 1.0], 50), happened, bins=2)
    assert expected_calibration_error(overconfident) == pytest.approx(0.1)
