"""Classification metrics."""

from __future__ import annotations

import numpy as np


def _labels(output_or_labels: np.ndarray) -> np.ndarray:
    a = np.asarray(output_or_labels)
    if a.ndim == 2:
        return a.argmax(axis=1) if a.shape[1] > 1 else (a[:, 0] > 0).astype(int)
    return a.astype(int)


def accuracy(output: np.ndarray, target: np.ndarray) -> float:
    """Accuracy from logits (N, C) / binary logits (N, 1) or labels, vs labels or one-hot."""
    t = np.asarray(target)
    true = t.ravel().astype(int) if (t.ndim == 2 and t.shape[1] == 1) else _labels(t)
    return float(np.mean(_labels(output) == true))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """``cm[i, j]`` = number of samples of true class i predicted as class j."""
    y_true, y_pred = _labels(y_true), _labels(y_pred)
    k = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision_recall_f1(y_true, y_pred, n_classes=None):
    """Per-class precision, recall and F1 as three arrays."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / cm.sum(axis=0))
        recall = np.nan_to_num(tp / cm.sum(axis=1))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))
    return precision, recall, f1


def classification_report(y_true, y_pred, class_names=None) -> str:
    p, r, f = precision_recall_f1(y_true, y_pred)
    support = np.bincount(_labels(y_true), minlength=len(p))
    names = class_names or [str(i) for i in range(len(p))]
    lines = [f"{'class':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}"]
    for i, name in enumerate(names):
        lines.append(f"{name:>10} {p[i]:>10.3f} {r[i]:>10.3f} {f[i]:>10.3f} {support[i]:>10}")
    lines.append(f"{'accuracy':>10} {'':>10} {'':>10} {accuracy(np.asarray(y_pred), np.asarray(y_true)):>10.3f} {support.sum():>10}")
    return "\n".join(lines)
