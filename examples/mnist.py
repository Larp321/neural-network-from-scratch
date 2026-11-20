"""Handwritten digit recognition on MNIST (downloads ~11 MB on first run).

    python examples/mnist.py            # full run, ~2 minutes on a laptop CPU
    python examples/mnist.py --epochs 3
"""
import argparse
import os

import numpy as np

from _path import IMAGES, ROOT
from tinynet import Adam, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy, StepDecay
from tinynet.datasets import load_mnist
from tinynet.metrics import classification_report, confusion_matrix
from tinynet.viz import confusion_matrix_plot, image_grid, line_chart

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--hidden", type=int, default=256)
args = parser.parse_args()

X_train, y_train, X_test, y_test = load_mnist(os.path.join(ROOT, "data", "mnist"))
X_val, y_val = X_train[-5000:], y_train[-5000:]
X_train, y_train = X_train[:-5000], y_train[:-5000]
print(f"train {X_train.shape}  val {X_val.shape}  test {X_test.shape}\n")

rng = np.random.default_rng(0)
model = Sequential([
    Dense(784, args.hidden, rng=rng), ReLU(), Dropout(0.2, rng=rng),
    Dense(args.hidden, 128, rng=rng), ReLU(), Dropout(0.2, rng=rng),
    Dense(128, 10, rng=rng),
])
print(model.summary(), "\n")
loss = SoftmaxCrossEntropy(label_smoothing=0.05)
history = model.fit(X_train, y_train, loss, Adam(lr=1e-3), epochs=args.epochs, batch_size=args.batch_size,
                    validation_data=(X_val, y_val), lr_schedule=StepDecay(1e-3, drop=0.5, every=4),
                    early_stopping=4, rng=rng)

pred = model.predict(X_test)
test_acc = float(np.mean(pred == y_test))
print(f"\nTEST ACCURACY: {test_acc:.2%}\n")
print(classification_report(y_test, pred))

model.save(os.path.join(ROOT, "data", "mnist_mlp.npz"))
confusion_matrix_plot(confusion_matrix(y_test, pred, 10), os.path.join(IMAGES, "mnist_confusion.svg"),
                      title=f"MNIST test set - {test_acc:.2%} accuracy")
line_chart({"train": history["acc"], "validation": history["val_acc"]},
           os.path.join(IMAGES, "mnist_accuracy.svg"), title="MNIST - accuracy", ylabel="accuracy")
# What the first layer learned: each hidden unit's 784 incoming weights viewed as a 28x28 image.
W = model.layers[0].params["W"].T[:50].reshape(-1, 28, 28)
image_grid(W, os.path.join(IMAGES, "mnist_weights.png"), ncols=10, scale=3, diverging=True)
wrong = np.flatnonzero(pred != y_test)[:20]
image_grid(X_test[wrong].reshape(-1, 28, 28), os.path.join(IMAGES, "mnist_mistakes.png"), ncols=10, scale=3)
print("first misclassified test images (true -> predicted):",
      ", ".join(f"{y_test[i]}->{pred[i]}" for i in wrong))
