"""Overfitting and how to fight it: no regularization vs L2 weight decay vs dropout.

A big network trained on a small, noisy dataset memorises the training set.
Validation loss reveals the problem; L2 and dropout reduce it.
"""
import os

import numpy as np

from _path import IMAGES
from tinynet import Adam, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy
from tinynet.datasets import make_moons, train_test_split
from tinynet.viz import decision_boundary, line_chart

rng = np.random.default_rng(11)
X, y = make_moons(n_samples=1000, noise=0.35, rng=rng)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.92, rng=rng)  # only 80 train points
print(f"{len(X_train)} training points, {len(X_val)} validation points\n")
loss = SoftmaxCrossEntropy()


def build(l2=0.0, dropout=0.0):
    r = np.random.default_rng(5)
    layers = []
    for i, o in [(2, 256), (256, 256)]:
        layers += [Dense(i, o, rng=r, l2=l2), ReLU()]
        if dropout:
            layers.append(Dropout(dropout, rng=r))
    return Sequential(layers + [Dense(256, 2, rng=r)])


configs = {"no regularization": {}, "L2 (1e-2)": {"l2": 1e-2}, "dropout (0.5)": {"dropout": 0.5}}
val_curves = {}
for name, kw in configs.items():
    model = build(**kw)
    h = model.fit(X_train, y_train, loss, Adam(lr=0.003), epochs=400, batch_size=16,
                  validation_data=(X_val, y_val), rng=np.random.default_rng(0), verbose=0)
    val_curves[name] = h["val_loss"]
    best_epoch, best = h.best("val_loss")
    print(f"{name:<18} train acc {h['acc'][-1]:.1%}   val acc {h['val_acc'][-1]:.1%}   "
          f"final val loss {h['val_loss'][-1]:.3f} (best {best:.3f} @ epoch {best_epoch + 1})")
    slug = name.split()[0].lower()
    decision_boundary(model, X_train, y_train, os.path.join(IMAGES, f"reg_{slug}.svg"), loss=loss,
                      title=name.capitalize() + " (train data)")

line_chart(val_curves, os.path.join(IMAGES, "regularization.svg"),
           title="Validation loss: regularization prevents overfitting", ylabel="validation cross-entropy")
