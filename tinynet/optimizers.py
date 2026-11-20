"""Gradient-based optimizers and learning-rate schedules."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .layers import Layer


class Optimizer:
    def __init__(self, lr: float, clip_norm: float | None = None) -> None:
        self.lr = lr
        self.clip_norm = clip_norm
        self.t = 0  # number of steps taken
        self.state: dict[tuple[int, str], dict[str, np.ndarray]] = {}

    def step(self, layers: Iterable[Layer]) -> None:
        layers = list(layers)
        self.t += 1
        scale = 1.0
        if self.clip_norm is not None:
            total = math.sqrt(sum(float(np.sum(g ** 2)) for l in layers for g in l.grads.values()))
            if total > self.clip_norm:
                scale = self.clip_norm / (total + 1e-12)
        for layer in layers:
            for name, param in layer.params.items():
                grad = layer.grads[name] * scale
                state = self.state.setdefault((id(layer), name), {})
                self._update(param, grad, state)

    def _update(self, param: np.ndarray, grad: np.ndarray, state: dict) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional (Nesterov) momentum."""

    def __init__(self, lr=0.1, momentum=0.0, nesterov=False, clip_norm=None):
        super().__init__(lr, clip_norm)
        self.momentum = momentum
        self.nesterov = nesterov

    def _update(self, param, grad, state):
        if self.momentum == 0.0:
            param -= self.lr * grad
            return
        v = state.setdefault("v", np.zeros_like(param))
        v *= self.momentum
        v -= self.lr * grad
        if self.nesterov:
            param += self.momentum * v - self.lr * grad
        else:
            param += v

    def __repr__(self):
        return f"SGD(lr={self.lr}, momentum={self.momentum}, nesterov={self.nesterov})"


class RMSProp(Optimizer):
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8, clip_norm=None):
        super().__init__(lr, clip_norm)
        self.rho, self.eps = rho, eps

    def _update(self, param, grad, state):
        s = state.setdefault("s", np.zeros_like(param))
        s *= self.rho
        s += (1.0 - self.rho) * grad ** 2
        param -= self.lr * grad / (np.sqrt(s) + self.eps)

    def __repr__(self):
        return f"RMSProp(lr={self.lr}, rho={self.rho})"


class Adam(Optimizer):
    """Adam (Kingma & Ba, 2015) with bias correction."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, clip_norm=None):
        super().__init__(lr, clip_norm)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def _update(self, param, grad, state):
        m = state.setdefault("m", np.zeros_like(param))
        v = state.setdefault("v", np.zeros_like(param))
        m *= self.beta1
        m += (1.0 - self.beta1) * grad
        v *= self.beta2
        v += (1.0 - self.beta2) * grad ** 2
        m_hat = m / (1.0 - self.beta1 ** self.t)
        v_hat = v / (1.0 - self.beta2 ** self.t)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __repr__(self):
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2})"


# --------------------------------------------------------------------------- #
# Learning-rate schedules: callables mapping epoch (0-based) -> learning rate
# --------------------------------------------------------------------------- #
class StepDecay:
    def __init__(self, initial_lr: float, drop: float = 0.5, every: int = 10) -> None:
        self.initial_lr, self.drop, self.every = initial_lr, drop, every

    def __call__(self, epoch: int) -> float:
        return self.initial_lr * self.drop ** (epoch // self.every)


class CosineDecay:
    def __init__(self, initial_lr: float, total_epochs: int, min_lr: float = 0.0) -> None:
        self.initial_lr, self.total, self.min_lr = initial_lr, total_epochs, min_lr

    def __call__(self, epoch: int) -> float:
        progress = min(epoch / max(1, self.total - 1), 1.0)
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
