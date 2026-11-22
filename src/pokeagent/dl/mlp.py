"""DL1 (#76): a per-turn neural network (multilayer perceptron) for the opponent's next action.

It sees exactly what logistic regression and LightGBM see: the 36 per-turn features of one
decision point. So comparing them isolates the model type. Like the chosen LightGBM, it is
trained with the fourth "hidden" class (E4) and evaluated on all turns.

An MLP here: standardise the features, then two fully connected layers with ReLU (max(0, x))
and dropout (randomly zeroing 10% of units while training, to reduce overfitting), then a
softmax over the classes. Trained with Adam on cross-entropy (= log loss), in mini-batches,
stopping when validation log loss has not improved for a few epochs (early stopping).
"""

import numpy as np
import torch
from torch import nn

from pokeagent.config import SEED
from pokeagent.models.opponent import OpponentModel, _selection_loss, _x, _y

HIDDEN_SIZES_GRID = ((64, 64), (256, 128))
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 40
PATIENCE = 3  # epochs without a validation improvement before stopping
DROPOUT = 0.1


class MLPEstimator:
    """A trained network with the scikit-learn style predict_proba / classes_ the model needs."""

    def __init__(self, network: nn.Module, mean: np.ndarray, std: np.ndarray, n_classes: int):
        self.network, self.mean, self.std = network, mean, std
        self.classes_ = np.arange(n_classes)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.network.eval()
        with torch.no_grad():
            inputs = torch.tensor((x - self.mean) / self.std, dtype=torch.float32)
            # Softmax in double precision, so each row sums to 1 to within rounding.
            return torch.softmax(self.network(inputs).double(), dim=1).numpy()


def _network(n_features: int, sizes: tuple[int, ...], n_classes: int) -> nn.Module:
    layers, width = [], n_features
    for size in sizes:
        layers += [nn.Linear(width, size), nn.ReLU(), nn.Dropout(DROPOUT)]
        width = size
    layers.append(nn.Linear(width, n_classes))  # softmax is applied in the loss / prediction
    return nn.Sequential(*layers)


def train_mlp(
    train,
    val,
    classes,
    sizes_grid=HIDDEN_SIZES_GRID,
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
):
    """Train one network per layer-size setting; keep the one with the best validation loss.

    The standardisation is fitted on training rows only. Returns the best OpponentModel and
    one record per setting (epochs trained, validation log loss).
    """
    x_train, y_train = _x(train), _y(train, classes)
    mean, std = x_train.mean(axis=0), x_train.std(axis=0)
    std[std == 0] = 1.0  # a constant feature: leave it unscaled rather than divide by zero
    inputs = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    targets = torch.tensor(y_train, dtype=torch.long)
    best, tried = None, []
    for sizes in sizes_grid:
        torch.manual_seed(SEED)
        generator = torch.Generator().manual_seed(SEED)
        network = _network(inputs.shape[1], sizes, len(classes))
        optimiser = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        estimator = MLPEstimator(network, mean, std, len(classes))
        model = OpponentModel(f"MLP {sizes}", estimator=estimator, classes=classes)
        best_loss, best_state, stale, epochs = np.inf, None, 0, 0
        for epoch in range(max_epochs):
            network.train()
            order = torch.randperm(len(inputs), generator=generator)
            for start in range(0, len(inputs), batch_size):
                batch = order[start : start + batch_size]
                optimiser.zero_grad()
                loss = loss_fn(network(inputs[batch]), targets[batch])
                loss.backward()
                optimiser.step()
            val_loss = _selection_loss(model, val)
            epochs = epoch + 1
            if val_loss < best_loss - 1e-5:
                best_loss, stale = val_loss, 0
                best_state = {k: v.clone() for k, v in network.state_dict().items()}
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        network.load_state_dict(best_state)
        tried.append({"sizes": list(sizes), "epochs": epochs, "val_log_loss": float(best_loss)})
        if best is None or best_loss < best[1]:
            best = (model, best_loss)
    return best[0], tried
