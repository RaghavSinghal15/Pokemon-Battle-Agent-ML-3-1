"""DL2 (#77): the whole-game GRU uses history, and never sees the future."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from pokeagent.dl.sequence import train_sequence_model  # noqa: E402
from pokeagent.features.build import FEATURE_NAMES  # noqa: E402
from pokeagent.interfaces import ACTION_CLASSES  # noqa: E402

LABELS = list(ACTION_CLASSES)


def games(n: int, seed: int) -> pd.DataFrame:
    """Each player repeats their previous action 90% of the time; features are pure noise.

    Only a model that remembers the game can beat the class frequencies.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(n):
        action = rng.integers(3)
        for turn in range(1, 13):
            if rng.random() > 0.9:
                action = rng.integers(3)
            features = dict(zip(FEATURE_NAMES, rng.normal(size=len(FEATURE_NAMES)), strict=True))
            rows.append(
                {
                    **features,
                    "turn": float(turn),
                    "battle_id": f"g{g}",
                    "perspective": "p1",
                    "action_class": LABELS[action],
                }
            )
    return pd.DataFrame(rows).sample(frac=1, random_state=seed)  # order must not matter


@pytest.fixture(scope="module")
def trained():
    train, val = games(300, 0), games(80, 1)
    full, record = train_sequence_model(train, val, ACTION_CLASSES, hidden=16, max_epochs=10)
    none, _ = train_sequence_model(train, val, ACTION_CLASSES, "none", hidden=16, max_epochs=10)
    return val, full, none, record


def log_loss(model, table):
    probs = model.predict_proba_table(table)
    truth = table["action_class"].map(LABELS.index).to_numpy()
    return -np.log(probs[np.arange(len(table)), truth]).mean()


def test_history_helps_when_behaviour_repeats(trained):
    val, full, none, record = trained
    assert record["epochs"] >= 1
    assert log_loss(full, val) < log_loss(none, val) - 0.1


def test_predictions_do_not_depend_on_later_turns(trained):
    val, full, _, _ = trained
    probs = pd.DataFrame(full.predict_proba_table(val), index=val.index)
    cut = val[val["turn"] <= 5]
    cut_probs = pd.DataFrame(full.predict_proba_table(cut), index=cut.index)
    assert np.allclose(cut_probs.to_numpy(), probs.loc[cut.index].to_numpy(), atol=1e-6)
