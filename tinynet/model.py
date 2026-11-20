"""The ``Sequential`` model: a stack of layers with a training loop."""

from __future__ import annotations

import time
from typing import Callable, Sequence

import numpy as np

from .layers import Layer
from .losses import MSE, Loss, probabilities
from .metrics import accuracy
from .optimizers import Optimizer


class History(dict):
    """Per-epoch training metrics, e.g. ``history["loss"]``, ``history["val_acc"]``."""

    def best(self, key: str = "val_loss") -> tuple[int, float]:
        values = self[key]
        idx = int(np.argmin(values) if "loss" in key else np.argmax(values))
        return idx, values[idx]


class Sequential:
    """A feed-forward network built from a list of layers.

    >>> from tinynet import Sequential, Dense, ReLU
    >>> net = Sequential([Dense(2, 16), ReLU(), Dense(16, 3)])
    """

    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = list(layers)

    # ------------------------------------------------------------------ core
    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    __call__ = forward

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def trainable_layers(self) -> list[Layer]:
        return [l for l in self.layers if l.params]

    def regularization_loss(self) -> float:
        return sum(l.regularization_loss() for l in self.layers)

    def num_params(self) -> int:
        return sum(l.num_params() for l in self.layers)

    def train_step(self, x: np.ndarray, y: np.ndarray, loss: Loss, optimizer: Optimizer) -> float:
        """One forward/backward/update pass on a mini-batch. Returns the batch loss."""
        out = self.forward(x, training=True)
        value = loss.forward(out, y) + self.regularization_loss()
        self.backward(loss.backward())
        optimizer.step(self.trainable_layers())
        return value

    # ------------------------------------------------------------- training
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss: Loss,
        optimizer: Optimizer,
        epochs: int = 10,
        batch_size: int = 32,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        lr_schedule: Callable[[int], float] | None = None,
        early_stopping: int | None = None,
        shuffle: bool = True,
        rng: np.random.Generator | None = None,
        verbose: int = 1,
        log_every: int = 1,
    ) -> History:
        """Train with mini-batch gradient descent.

        early_stopping : stop after this many epochs without validation-loss
            improvement and restore the best weights (requires validation_data).
        verbose : 0 = silent, 1 = one line every ``log_every`` epochs.
        """
        rng = rng if rng is not None else np.random.default_rng()
        n = X.shape[0]
        classify = not isinstance(loss, MSE)
        history = History(loss=[], lr=[], **({"acc": []} if classify else {}))
        if validation_data is not None:
            history.update(val_loss=[], **({"val_acc": []} if classify else {}))
        best_val, best_weights, patience = np.inf, None, 0

        for epoch in range(epochs):
            if lr_schedule is not None:
                optimizer.lr = lr_schedule(epoch)
            start = time.perf_counter()
            idx = rng.permutation(n) if shuffle else np.arange(n)
            total = 0.0
            for i in range(0, n, batch_size):
                b = idx[i : i + batch_size]
                total += self.train_step(X[b], y[b], loss, optimizer) * len(b)

            train_loss = total / n
            history["loss"].append(train_loss)
            history["lr"].append(optimizer.lr)
            msg = f"epoch {epoch + 1:>4}/{epochs}  loss {train_loss:.4f}"
            if classify:
                history["acc"].append(accuracy(self.forward(X), y))
                msg += f"  acc {history['acc'][-1]:.3f}"

            if validation_data is not None:
                v_loss, v_acc = self.evaluate(*validation_data, loss=loss)
                history["val_loss"].append(v_loss)
                msg += f"  val_loss {v_loss:.4f}"
                if classify:
                    history["val_acc"].append(v_acc)
                    msg += f"  val_acc {v_acc:.3f}"
                if early_stopping is not None:
                    if v_loss < best_val - 1e-6:
                        best_val, patience = v_loss, 0
                        best_weights = self.get_weights()
                    else:
                        patience += 1
                        if patience >= early_stopping:
                            if verbose:
                                print(msg + f"\nearly stopping at epoch {epoch + 1}; restoring best weights")
                            self.set_weights(best_weights)
                            break

            if verbose and ((epoch + 1) % log_every == 0 or epoch == 0 or epoch == epochs - 1):
                print(msg + f"  ({time.perf_counter() - start:.2f}s)")
        return history

    # ------------------------------------------------------------ inference
    def evaluate(self, X: np.ndarray, y: np.ndarray, loss: Loss) -> tuple[float, float]:
        """Return ``(data loss, accuracy)``; accuracy is NaN for regression (MSE).

        The L2 penalty is *not* included, so validation losses are comparable
        between regularized and unregularized models.
        """
        out = self.forward(X)
        acc = float("nan") if isinstance(loss, MSE) else accuracy(out, y)
        return loss.forward(out, y), acc

    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Predicted class labels (argmax for multi-class, logit > 0 for binary)."""
        out = np.concatenate([self.forward(X[i : i + batch_size]) for i in range(0, len(X), batch_size)])
        if out.shape[1] == 1:
            return (out[:, 0] > 0).astype(int)
        return out.argmax(axis=1)

    def predict_proba(self, X: np.ndarray, loss: Loss) -> np.ndarray:
        return probabilities(loss, self.forward(X))

    # ------------------------------------------------------- serialization
    def get_weights(self) -> dict[str, np.ndarray]:
        return {f"{i}.{k}": v.copy() for i, l in enumerate(self.layers) for k, v in l.params.items()}

    def set_weights(self, weights: dict[str, np.ndarray]) -> None:
        for i, layer in enumerate(self.layers):
            for k in layer.params:
                key = f"{i}.{k}"
                if weights[key].shape != layer.params[k].shape:
                    raise ValueError(f"shape mismatch for {key}")
                layer.params[k][...] = weights[key]

    def save(self, path: str) -> None:
        np.savez(path, **self.get_weights())

    def load(self, path: str) -> None:
        with np.load(path) as data:
            self.set_weights(dict(data))

    def summary(self) -> str:
        lines = [f"{'#':<3} {'layer':<32} {'params':>10}", "-" * 47]
        for i, l in enumerate(self.layers):
            lines.append(f"{i:<3} {repr(l):<32} {l.num_params():>10,}")
        lines += ["-" * 47, f"{'total trainable parameters':<36} {self.num_params():>10,}"]
        return "\n".join(lines)

    def __repr__(self):
        return "Sequential([\n  " + ",\n  ".join(map(repr, self.layers)) + "\n])"
