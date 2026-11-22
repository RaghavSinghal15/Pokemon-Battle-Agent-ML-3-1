"""DL2 (#77): a whole-game sequence model for the opponent's next action.

The per-turn models (LR, LightGBM, DL1's MLP) see one turn's features. This model reads the
game so far for the player being predicted: at each of their decision turns, that turn's
start-of-turn features plus what they did on their previous decision turn (attack,
non_damaging, switch, or hidden). A GRU (gated recurrent unit: a small network that updates a
running summary vector one step at a time) carries the history forward, and a linear layer
turns the summary at turn N into class probabilities.

No leakage: the input at turn N holds turn N's start-of-turn features and actions from
EARLIER turns only. The GRU only runs forwards, so the prediction at N cannot see later turns
(test_dl_sequence.py checks this by cutting the game after turn N).

The question it answers: does knowing what this player did earlier help beyond the position?
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from pokeagent.config import SEED
from pokeagent.features.build import FEATURE_NAMES
from pokeagent.models.opponent import HIDDEN, LABEL, WITH_HIDDEN, fold_hidden

GROUP = ["battle_id", "perspective"]  # one sequence per game and predicted player
HIDDEN_SIZE = 64
LEARNING_RATE = 2e-3
BATCH_GAMES = 64
MAX_EPOCHS = 30
PATIENCE = 3


class _Network(nn.Module):
    def __init__(self, n_inputs: int, n_classes: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(n_inputs, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        summaries, _ = self.gru(inputs)  # one summary per step, each from that step and earlier
        return self.out(summaries)


def _sequences(table: pd.DataFrame, mean, std, classes, history: str = "full"):
    """Per game: (row positions in `table`, inputs [steps x features], labels [steps]).

    history="full" is the model; "none" blanks the previous-action inputs and resets the GRU
    every step (an ablation: the same network with no memory of the game).
    """
    n_classes = len(classes)
    ordered = table.reset_index(drop=True)
    ordered = ordered.assign(_row=np.arange(len(ordered))).sort_values([*GROUP, "turn"])
    result = []
    for _, game in ordered.groupby(GROUP, sort=False):
        x = (game[list(FEATURE_NAMES)].to_numpy(dtype=float) - mean) / std
        y = game[LABEL].map({c: i for i, c in enumerate(classes)}).to_numpy()
        previous = np.zeros((len(game), n_classes + 1))
        previous[0, n_classes] = 1.0  # "first decision of the game"
        for step in range(1, len(game)):
            previous[step, y[step - 1]] = 1.0
        if history == "none":
            previous[:] = 0.0
        inputs = np.concatenate([x, previous], axis=1)
        result.append((game["_row"].to_numpy(), inputs.astype(np.float32), y))
    return result


class SequenceOpponentModel:
    """Like OpponentModel, but predict_proba_table needs every row of each game, in any order."""

    def __init__(self, name, network, mean, std, classes=WITH_HIDDEN, history="full"):
        self.name, self.network, self.mean, self.std = name, network, mean, std
        self.classes, self.history = classes, history

    def raw_proba_table(self, table: pd.DataFrame, sequences=None) -> np.ndarray:
        """sequences: the table's games already prepared by _sequences (saves rebuilding)."""
        if sequences is None:
            sequences = _sequences(table, self.mean, self.std, self.classes, self.history)
        probs = np.zeros((len(table), len(self.classes)))
        self.network.eval()
        with torch.no_grad():
            for rows, inputs, _ in sequences:
                x = torch.tensor(inputs)
                if self.history == "none":
                    logits = self.network(x[:, None, :])[:, 0, :]  # each step on its own
                else:
                    logits = self.network(x[None])[0]
                probs[rows] = torch.softmax(logits.double(), dim=1).numpy()
        return probs

    def predict_proba_table(self, table: pd.DataFrame) -> np.ndarray:
        probs = self.raw_proba_table(table)
        return fold_hidden(probs) if HIDDEN in self.classes else probs


def _batches(sequences, generator):
    order = torch.randperm(len(sequences), generator=generator).tolist()
    for start in range(0, len(order), BATCH_GAMES):
        chosen = [sequences[i] for i in order[start : start + BATCH_GAMES]]
        length = max(len(s[2]) for s in chosen)
        inputs = np.zeros((len(chosen), length, chosen[0][1].shape[1]), dtype=np.float32)
        labels = np.full((len(chosen), length), -100)  # -100: padding, ignored by the loss
        for i, (_, x, y) in enumerate(chosen):
            inputs[i, : len(y)], labels[i, : len(y)] = x, y
        yield torch.tensor(inputs), torch.tensor(labels)


def _val_loss(model: SequenceOpponentModel, val: pd.DataFrame, sequences) -> float:
    probs = model.raw_proba_table(val, sequences)
    truth = val[LABEL].map({c: i for i, c in enumerate(model.classes)}).to_numpy()
    return float(-np.log(np.clip(probs[np.arange(len(val)), truth], 1e-15, 1)).mean())


def train_sequence_model(
    train, val, classes=WITH_HIDDEN, history="full", hidden=HIDDEN_SIZE, max_epochs=MAX_EPOCHS
):
    """Train the GRU with early stopping on validation log loss (over the trained classes).

    Standardisation is fitted on training rows only. Returns the model and a training record.
    """
    x_train = train[list(FEATURE_NAMES)].to_numpy(dtype=float)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0
    sequences = _sequences(train, mean, std, classes, history)
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    network = _Network(sequences[0][1].shape[1], len(classes), hidden)
    model = SequenceOpponentModel(
        f"GRU over the game ({history} history)", network, mean, std, classes, history
    )
    optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    val_sequences = _sequences(val, mean, std, classes, history)  # built once, not every epoch
    best_loss, best_state, stale, epochs = np.inf, None, 0, 0
    for epoch in range(max_epochs):
        network.train()
        for inputs, labels in _batches(sequences, generator):
            optimiser.zero_grad()
            if history == "none":
                # No memory: every step is a sequence of length one.
                flat = inputs.reshape(-1, 1, inputs.shape[-1])
                logits = network(flat).reshape(*labels.shape, -1)
            else:
                logits = network(inputs)
            loss = loss_fn(logits.reshape(-1, len(classes)), labels.reshape(-1))
            loss.backward()
            optimiser.step()
        loss = _val_loss(model, val, val_sequences)
        epochs = epoch + 1
        if loss < best_loss - 1e-5:
            best_loss, stale = loss, 0
            best_state = {k: v.clone() for k, v in network.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    network.load_state_dict(best_state)
    return model, {"history": history, "epochs": epochs, "val_log_loss": best_loss}
