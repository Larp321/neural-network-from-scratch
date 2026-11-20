"""Three interleaved spirals: linear model vs a 2-hidden-layer MLP."""
import os

import numpy as np

from _path import IMAGES
from tinynet import Adam, Dense, ReLU, Sequential, SoftmaxCrossEntropy
from tinynet.datasets import make_spiral, train_test_split
from tinynet.metrics import classification_report
from tinynet.viz import decision_boundary, line_chart

rng = np.random.default_rng(42)
X, y = make_spiral(n_per_class=200, n_classes=3, noise=0.2, rng=rng)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, rng=rng)
loss = SoftmaxCrossEntropy()

# 1) A purely linear classifier (softmax regression) - it can only draw straight lines.
linear = Sequential([Dense(2, 3, rng=rng)])
linear.fit(X_train, y_train, loss, Adam(lr=0.05), epochs=100, batch_size=64, rng=rng, verbose=0)
lin_acc = decision_boundary(linear, X_test, y_test, os.path.join(IMAGES, "spiral_linear.svg"),
                            loss=loss, title="Linear model (no hidden layer)")

# 2) A multi-layer perceptron.
mlp = Sequential([
    Dense(2, 64, rng=rng, l2=1e-4), ReLU(),
    Dense(64, 64, rng=rng, l2=1e-4), ReLU(),
    Dense(64, 3, rng=rng),
])
print(mlp.summary(), "\n")
history = mlp.fit(X_train, y_train, loss, Adam(lr=0.01), epochs=300, batch_size=32,
                  validation_data=(X_test, y_test), rng=rng, log_every=50)
mlp_acc = decision_boundary(mlp, X_test, y_test, os.path.join(IMAGES, "spiral_mlp.svg"),
                            loss=loss, title="MLP 2-64-64-3")

print(f"\nlinear model test accuracy: {lin_acc:.1%}")
print(f"MLP          test accuracy: {mlp_acc:.1%}\n")
print(classification_report(y_test, mlp.predict(X_test)))

line_chart({"train": history["loss"], "validation": history["val_loss"]},
           os.path.join(IMAGES, "spiral_loss.svg"), title="Spiral - loss", ylabel="cross-entropy", log_y=True)
line_chart({"train": history["acc"], "validation": history["val_acc"]},
           os.path.join(IMAGES, "spiral_accuracy.svg"), title="Spiral - accuracy", ylabel="accuracy")
