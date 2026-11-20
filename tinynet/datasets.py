"""Toy datasets, preprocessing helpers and an optional MNIST loader."""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np


def _rng(rng):
    return rng if rng is not None else np.random.default_rng()


def make_xor() -> tuple[np.ndarray, np.ndarray]:
    """The 4-point XOR truth table: the classic problem a linear model cannot solve."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    return X, y


def make_spiral(n_per_class: int = 100, n_classes: int = 3, noise: float = 0.2, rng=None):
    """Interleaved spiral arms - a hard, highly non-linear 2-D classification task."""
    rng = _rng(rng)
    X = np.zeros((n_per_class * n_classes, 2))
    y = np.zeros(n_per_class * n_classes, dtype=int)
    for k in range(n_classes):
        ix = slice(n_per_class * k, n_per_class * (k + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(k * 4, (k + 1) * 4, n_per_class) + rng.normal(0, noise, n_per_class)
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = k
    return X, y


def make_moons(n_samples: int = 400, noise: float = 0.1, rng=None):
    """Two interleaving half-circles."""
    rng = _rng(rng)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    X = np.vstack([
        np.c_[np.cos(t_out), np.sin(t_out)],
        np.c_[1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5],
    ])
    X += rng.normal(0, noise, X.shape)
    y = np.r_[np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)]
    return X, y


def make_circles(n_samples: int = 400, noise: float = 0.05, factor: float = 0.5, rng=None):
    """A small circle inside a larger one."""
    rng = _rng(rng)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    t_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    X = np.vstack([np.c_[np.cos(t_out), np.sin(t_out)], factor * np.c_[np.cos(t_in), np.sin(t_in)]])
    X += rng.normal(0, noise, X.shape)
    y = np.r_[np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)]
    return X, y


def make_sine(n_samples: int = 200, noise: float = 0.1, rng=None):
    """1-D regression: y = sin(x) + noise on x in [-pi, pi]."""
    rng = _rng(rng)
    X = np.sort(rng.uniform(-np.pi, np.pi, (n_samples, 1)), axis=0)
    y = np.sin(X) + rng.normal(0, noise, X.shape)
    return X, y


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def train_test_split(X, y, test_size: float = 0.2, rng=None):
    rng = _rng(rng)
    idx = rng.permutation(len(X))
    n_test = int(round(len(X) * test_size))
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]


def one_hot(y: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    n_classes = n_classes or int(y.max()) + 1
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


class Standardizer:
    """Zero-mean / unit-variance feature scaling fitted on training data."""

    def fit(self, X: np.ndarray) -> "Standardizer":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------- #
# MNIST (optional - downloads ~11 MB on first use)
# --------------------------------------------------------------------------- #
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _read_idx(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        _zero, _dtype_code, ndim = struct.unpack(">HBB", f.read(4))
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def load_mnist(data_dir: str = "data/mnist", download: bool = True):
    """Return ``(X_train, y_train, X_test, y_test)``; images flattened to 784 floats in [0, 1]."""
    os.makedirs(data_dir, exist_ok=True)
    arrays = {}
    for key, fname in MNIST_FILES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError(path)
            print(f"downloading {fname} ...")
            urllib.request.urlretrieve(MNIST_URL + fname, path)
        arrays[key] = _read_idx(path)
    X_train = arrays["train_images"].reshape(-1, 784).astype(np.float32) / 255.0
    X_test = arrays["test_images"].reshape(-1, 784).astype(np.float32) / 255.0
    return X_train, arrays["train_labels"].astype(int), X_test, arrays["test_labels"].astype(int)
