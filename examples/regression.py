"""Function approximation: fit y = sin(x) from noisy samples with MSE loss."""
import os

import numpy as np

from _path import IMAGES
from tinynet import MSE, Adam, CosineDecay, Dense, Sequential, Tanh
from tinynet.datasets import make_sine
from tinynet.viz import line_chart, regression_plot

rng = np.random.default_rng(3)
X, y = make_sine(n_samples=200, noise=0.15, rng=rng)

model = Sequential([Dense(1, 32, init="xavier", rng=rng), Tanh(),
                    Dense(32, 32, init="xavier", rng=rng), Tanh(),
                    Dense(32, 1, init="xavier", rng=rng)])
epochs = 400
history = model.fit(X, y, MSE(), Adam(lr=0.01), epochs=epochs, batch_size=32,
                    lr_schedule=CosineDecay(0.01, epochs, min_lr=1e-4), rng=rng, verbose=1, log_every=100)

xs = np.linspace(-np.pi, np.pi, 500)[:, None]
rmse = float(np.sqrt(np.mean((model.forward(xs) - np.sin(xs)) ** 2)))
print(f"\nRMSE vs the true sin(x): {rmse:.4f}  (noise std was 0.15)")

regression_plot(X, y, model, os.path.join(IMAGES, "regression.svg"),
                title="Fitting sin(x) with a 1-32-32-1 tanh network", true_fn=np.sin)
line_chart({"training loss": history["loss"]}, os.path.join(IMAGES, "regression_loss.svg"),
           title="Regression - training loss", ylabel="MSE", log_y=True)
