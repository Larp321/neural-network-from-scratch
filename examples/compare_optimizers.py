"""Same network, same initial weights, four optimizers: how fast does each converge?"""
import os

import numpy as np

from _path import IMAGES
from tinynet import SGD, Adam, Dense, ReLU, RMSProp, Sequential, SoftmaxCrossEntropy
from tinynet.datasets import make_spiral
from tinynet.viz import line_chart

rng = np.random.default_rng(7)
X, y = make_spiral(n_per_class=150, n_classes=3, noise=0.15, rng=rng)
loss = SoftmaxCrossEntropy()


def build():
    r = np.random.default_rng(123)  # identical initial weights for every optimizer
    return Sequential([Dense(2, 64, rng=r), ReLU(), Dense(64, 64, rng=r), ReLU(), Dense(64, 3, rng=r)])


optimizers = {
    "SGD": SGD(lr=0.05),
    "SGD + momentum": SGD(lr=0.05, momentum=0.9),
    "RMSProp": RMSProp(lr=0.003),
    "Adam": Adam(lr=0.005),
}
curves = {}
for name, opt in optimizers.items():
    model = build()
    h = model.fit(X, y, loss, opt, epochs=200, batch_size=32, rng=np.random.default_rng(0), verbose=0)
    curves[name] = h["loss"]
    print(f"{name:<16} final loss {h['loss'][-1]:.4f}  accuracy {h['acc'][-1]:.1%}")

line_chart(curves, os.path.join(IMAGES, "optimizers.svg"),
           title="Optimizer comparison on the spiral dataset", ylabel="training loss", log_y=True)
