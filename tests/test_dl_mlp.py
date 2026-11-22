"""DL1 (#76): the per-turn network learns the synthetic signal and is repeatable.

Skipped where torch is not installed (CI installs only the core extras).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_opponent_model import synthetic  # noqa: E402

from pokeagent.dl.mlp import train_mlp  # noqa: E402
from pokeagent.interfaces import ACTION_CLASSES  # noqa: E402
from pokeagent.models.opponent import evaluate, train_naive  # noqa: E402


def test_mlp_beats_naive_and_is_seeded():
    train, val = synthetic(2000, 0), synthetic(600, 1)
    first, tried = train_mlp(
        train, val, ACTION_CLASSES, sizes_grid=((16,),), max_epochs=8, batch_size=64
    )
    second, _ = train_mlp(
        train, val, ACTION_CLASSES, sizes_grid=((16,),), max_epochs=8, batch_size=64
    )
    assert tried[0]["sizes"] == [16] and 1 <= tried[0]["epochs"] <= 8
    assert evaluate(first, val)["log_loss"] < evaluate(train_naive(train), val)["log_loss"] - 0.1
    row = val.iloc[0].to_dict()
    assert first.predict_proba(row) == pytest.approx(second.predict_proba(row))
    assert sum(first.predict_proba(row).values()) == pytest.approx(1)
    assert np.all(first.predict_proba_table(val) >= 0)
