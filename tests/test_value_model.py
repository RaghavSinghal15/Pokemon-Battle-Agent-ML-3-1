"""Value model (V2, #72) on a small synthetic table: whoever has more team HP tends to win."""

import numpy as np
import pandas as pd
import pytest

from pokeagent.features.value import LABEL, VALUE_FEATURE_NAMES
from pokeagent.models.value import (
    evaluate_value,
    load_value_split,
    mirror_gap,
    train_team_hp_baseline,
    train_value_lightgbm,
    train_value_logistic,
)


def synthetic(battles: int, seed: int) -> pd.DataFrame:
    """Two mirrored rows per turn; the side with more team HP wins more often."""
    rng = np.random.default_rng(seed)
    rows = []
    for b in range(battles):
        for turn in range(1, 6):
            base = dict(
                zip(VALUE_FEATURE_NAMES, rng.normal(size=len(VALUE_FEATURE_NAMES)), strict=True)
            )
            mine, theirs = rng.uniform(0, 6, size=2)
            p1_wins = rng.random() < 1 / (1 + np.exp(-(mine - theirs)))
            for side, (a, c), won in (
                ("p1", (mine, theirs), p1_wins),
                ("p2", (theirs, mine), not p1_wins),
            ):
                rows.append(
                    {
                        **base,
                        "battle_id": f"b{seed}-{b}",
                        "perspective": side,
                        "turn": float(turn),
                        "my_team_hp": a,
                        "opp_team_hp": c,
                        LABEL: won,
                    }
                )
    return pd.DataFrame(rows)


def test_models_beat_a_coin_flip_and_learn_the_team_hp_signal():
    train, val = synthetic(300, 0), synthetic(100, 1)
    baseline = train_team_hp_baseline(train)
    lr, tried = train_value_logistic(train, val, c_grid=(0.1, 1.0))
    gbm, _ = train_value_lightgbm(train, val, leaves_grid=(7,))
    assert [t["C"] for t in tried] == [0.1, 1.0]
    for model in (baseline, lr, gbm):
        result = evaluate_value(model, val)
        assert result["log_loss"] < np.log(2) - 0.05 and result["auc"] > 0.7
    assert baseline.win_probability(val.iloc[0].to_dict()) == pytest.approx(
        baseline.win_probability_table(val.iloc[:1])[0]
    )


def test_mirror_gap_is_small_for_a_symmetric_model():
    train, val = synthetic(300, 0), synthetic(50, 1)
    # The team-HP lead flips sign between the two rows, so the logistic model is exactly symmetric
    # apart from its intercept.
    assert mirror_gap(train_team_hp_baseline(train), val) < 0.05


def test_test_split_needs_permission_and_rows_need_every_feature(tmp_path):
    table = synthetic(5, 0).assign(split="test")
    table.to_parquet(tmp_path / "v.parquet")
    with pytest.raises(PermissionError):
        load_value_split("test", tmp_path / "v.parquet")
    assert len(load_value_split("test", tmp_path / "v.parquet", allow_test=True)) == len(table)
    model = train_team_hp_baseline(synthetic(50, 0))
    with pytest.raises(ValueError, match="missing"):
        model.win_probability({"my_team_hp": 3.0})
