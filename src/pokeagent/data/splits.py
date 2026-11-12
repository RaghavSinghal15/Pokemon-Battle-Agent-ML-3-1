"""Train / validation / test split, made ONCE and shared by everyone."""

import pandas as pd

from pokeagent.config import SEED


def make_splits(
    battles: pd.DataFrame, test_frac: float = 0.15, val_frac: float = 0.15, seed: int = SEED
) -> dict[str, list[str]]:
    """Split battle ids into train, validation and test sets.

    Rules (these are viva questions, so understand why):
    - Split by BATTLE, never by turn. All turns of one game must land in the same set; otherwise
      the model is tested on games it has partly seen in training.
    - The test set is the NEWEST battles by upload time. This checks that the model works on
      games played after the ones it learned from, like it will when playing live.
    - Validation is a random sample of the remaining battles, used for tuning and calibration.
    - Save the result to config.SPLITS_PATH and never regenerate it after models are trained.

    TODO (role B): implement, and add a test that no battle id appears in two sets.
    """
    raise NotImplementedError
