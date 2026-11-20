"""Loss functions.

Each loss exposes ``forward(output, target) -> float`` (mean over the batch)
and ``backward() -> dL/d(output)``. Classification losses take raw *logits*
and fuse the final activation into the loss for numerical stability.
"""

from __future__ import annotations

import numpy as np

from .layers import sigmoid, softmax


class Loss:
    def forward(self, output: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, output, target):
        return self.forward(output, target)


class SoftmaxCrossEntropy(Loss):
    """Softmax + categorical cross-entropy on logits.

    ``target`` may be integer class labels of shape (N,) or one-hot (N, C).
    The gradient simplifies to ``(softmax(logits) - onehot) / N``.
    """

    def __init__(self, label_smoothing: float = 0.0) -> None:
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.label_smoothing = label_smoothing

    def forward(self, output, target):
        n, c = output.shape
        if target.ndim == 1:
            onehot = np.zeros((n, c))
            onehot[np.arange(n), target.astype(int)] = 1.0
        else:
            onehot = target.astype(float)
        if self.label_smoothing:
            onehot = onehot * (1.0 - self.label_smoothing) + self.label_smoothing / c
        z = output - output.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        self._probs = np.exp(log_probs)
        self._onehot = onehot
        return float(-(onehot * log_probs).sum() / n)

    def backward(self):
        n = self._probs.shape[0]
        return (self._probs - self._onehot) / n


class BinaryCrossEntropy(Loss):
    """Sigmoid + binary cross-entropy on logits (shape (N, 1) or (N,))."""

    def forward(self, output, target):
        target = target.reshape(output.shape).astype(float)
        # log(1 + exp(-|z|)) formulation avoids overflow for large |z|
        loss = np.maximum(output, 0) - output * target + np.log1p(np.exp(-np.abs(output)))
        self._out, self._target = output, target
        return float(loss.sum() / output.shape[0])

    def backward(self):
        return (sigmoid(self._out) - self._target) / self._out.shape[0]


class MSE(Loss):
    """Mean squared error: ``0.5 * sum((y_hat - y)^2) / N``."""

    def forward(self, output, target):
        target = target.reshape(output.shape).astype(float)
        self._diff = output - target
        return float(0.5 * np.sum(self._diff ** 2) / output.shape[0])

    def backward(self):
        return self._diff / self._diff.shape[0]


def probabilities(loss: Loss, logits: np.ndarray) -> np.ndarray:
    """Convert raw network output into probabilities for the given loss."""
    if isinstance(loss, SoftmaxCrossEntropy):
        return softmax(logits)
    if isinstance(loss, BinaryCrossEntropy):
        return sigmoid(logits)
    return logits
