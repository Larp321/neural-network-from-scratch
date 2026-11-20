"""XOR: the smallest problem that needs a hidden layer.

A single linear layer cannot separate XOR; one hidden layer of 4 tanh units can.
"""
import os

import numpy as np

from _path import IMAGES
from tinynet import SGD, BinaryCrossEntropy, Dense, Sequential, Tanh, sigmoid
from tinynet.datasets import make_xor
from tinynet.viz import decision_boundary, line_chart

rng = np.random.default_rng(0)
X, y = make_xor()

model = Sequential([Dense(2, 4, init="xavier", rng=rng), Tanh(), Dense(4, 1, init="xavier", rng=rng)])
loss = BinaryCrossEntropy()
print(model.summary(), "\n")

history = model.fit(X, y, loss, SGD(lr=0.5, momentum=0.9), epochs=300, batch_size=4,
                    rng=rng, verbose=1, log_every=50)

print("\n x1 x2 | target | P(y=1)")
for xi, yi, p in zip(X, y, sigmoid(model.forward(X))[:, 0]):
    print(f"  {int(xi[0])}  {int(xi[1])} |   {yi}    | {p:.3f}")

line_chart({"training loss": history["loss"]}, os.path.join(IMAGES, "xor_loss.svg"),
           title="XOR - training loss", ylabel="binary cross-entropy", log_y=True)
decision_boundary(model, X, y, os.path.join(IMAGES, "xor_boundary.svg"), loss=loss,
                  title="XOR", padding=0.5)
