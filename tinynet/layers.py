"""Neural network layers.

Every layer implements ``forward(x, training)`` and ``backward(grad)``.
``backward`` receives dL/d(output) and returns dL/d(input), storing the
gradients of its own parameters in ``self.grads`` (same keys as ``self.params``).
"""

from __future__ import annotations

import numpy as np


class Layer:
    """Base class. Layers without parameters leave ``params`` empty."""

    def __init__(self) -> None:
        self.params: dict[str, np.ndarray] = {}
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def regularization_loss(self) -> float:
        return 0.0

    def num_params(self) -> int:
        return int(sum(p.size for p in self.params.values()))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# --------------------------------------------------------------------------- #
# Fully connected layer
# --------------------------------------------------------------------------- #
class Dense(Layer):
    """Fully connected layer: ``y = x @ W + b``.

    Parameters
    ----------
    in_features, out_features : layer shape.
    init : "he" (good for ReLU), "xavier" (good for tanh/sigmoid) or "normal".
    l2 : L2 weight-decay strength (adds ``0.5 * l2 * ||W||^2`` to the loss).
    dtype : parameter dtype; np.float32 halves memory and roughly doubles speed.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        init: str = "he",
        l2: float = 0.0,
        rng: np.random.Generator | None = None,
        dtype=np.float64,
    ) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        if init == "he":
            std = np.sqrt(2.0 / in_features)
        elif init in ("xavier", "glorot"):
            std = np.sqrt(2.0 / (in_features + out_features))
        elif init == "normal":
            std = 0.01
        else:
            raise ValueError(f"unknown init '{init}'")
        self.in_features = in_features
        self.out_features = out_features
        self.l2 = l2
        self.params["W"] = rng.normal(0.0, std, size=(in_features, out_features)).astype(dtype)
        self.params["b"] = np.zeros(out_features, dtype=dtype)
        self.grads["W"] = np.zeros_like(self.params["W"])
        self.grads["b"] = np.zeros_like(self.params["b"])
        self._x: np.ndarray | None = None

    def forward(self, x, training=False):
        self._x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, grad):
        W = self.params["W"]
        self.grads["W"] = self._x.T @ grad + self.l2 * W
        self.grads["b"] = grad.sum(axis=0)
        return grad @ W.T

    def regularization_loss(self):
        if self.l2 == 0.0:
            return 0.0
        return 0.5 * self.l2 * float(np.sum(self.params["W"] ** 2))

    def __repr__(self):
        extra = f", l2={self.l2}" if self.l2 else ""
        return f"Dense({self.in_features} -> {self.out_features}{extra})"


# --------------------------------------------------------------------------- #
# Activations
# --------------------------------------------------------------------------- #
class ReLU(Layer):
    def forward(self, x, training=False):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask


class LeakyReLU(Layer):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x, training=False):
        self._mask = x > 0
        return np.where(self._mask, x, self.alpha * x)

    def backward(self, grad):
        return np.where(self._mask, grad, self.alpha * grad)

    def __repr__(self):
        return f"LeakyReLU(alpha={self.alpha})"


class Sigmoid(Layer):
    def forward(self, x, training=False):
        self._out = sigmoid(x)
        return self._out

    def backward(self, grad):
        return grad * self._out * (1.0 - self._out)


class Tanh(Layer):
    def forward(self, x, training=False):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad):
        return grad * (1.0 - self._out ** 2)


class Dropout(Layer):
    """Inverted dropout: active only when ``training=True``."""

    def __init__(self, p: float = 0.5, rng: np.random.Generator | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability must be in [0, 1)")
        self.p = p
        self.rng = rng if rng is not None else np.random.default_rng()
        self._mask: np.ndarray | None = None

    def forward(self, x, training=False):
        if not training or self.p == 0.0:
            self._mask = None
            return x
        self._mask = ((self.rng.random(x.shape) >= self.p) / (1.0 - self.p)).astype(x.dtype, copy=False)
        return x * self._mask

    def backward(self, grad):
        return grad if self._mask is None else grad * self._mask

    def __repr__(self):
        return f"Dropout(p={self.p})"


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise numerically stable softmax."""
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
