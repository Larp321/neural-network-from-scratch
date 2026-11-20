"""Numerical gradient checking - verifies backpropagation is implemented correctly."""

from __future__ import annotations

import numpy as np

from .losses import Loss
from .model import Sequential


def numerical_gradient(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central-difference gradient of scalar function ``f()`` w.r.t. array ``x`` (modified in place)."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        plus = f()
        x[i] = old - eps
        minus = f()
        x[i] = old
        grad[i] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def check_model_gradients(model: Sequential, X: np.ndarray, y: np.ndarray, loss: Loss,
                          eps: float = 1e-5) -> dict[str, float]:
    """Compare analytic vs numerical gradients for every parameter.

    Returns ``{"<layer>.<param>": max relative error}``. Values below ~1e-6
    mean backprop is correct. Dropout layers must be disabled (training=False
    is used, so they are).
    """

    def total_loss() -> float:
        return loss.forward(model.forward(X, training=False), y) + model.regularization_loss()

    total_loss()
    model.backward(loss.backward())
    analytic = {(i, k): l.grads[k].copy() for i, l in enumerate(model.layers) for k in l.params}

    errors = {}
    for (i, k), g in analytic.items():
        num = numerical_gradient(total_loss, model.layers[i].params[k], eps)
        errors[f"{i}.{k}"] = relative_error(g, num)
    return errors


def check_input_gradient(model: Sequential, X: np.ndarray, y: np.ndarray, loss: Loss,
                         eps: float = 1e-5) -> float:
    """Max relative error of dL/dX (checks every layer's ``backward`` return value)."""
    X = X.astype(float).copy()
    loss.forward(model.forward(X), y)
    analytic = model.backward(loss.backward())
    num = numerical_gradient(lambda: loss.forward(model.forward(X), y), X, eps)
    return relative_error(analytic, num)
