# tinynet — a neural network from scratch in NumPy

![tests](https://img.shields.io/badge/tests-26%20passing-brightgreen) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![deps](https://img.shields.io/badge/dependencies-numpy%20only-orange) ![license](https://img.shields.io/badge/license-MIT-lightgrey)

A small, readable deep-learning library built **without any ML framework** — no
PyTorch, TensorFlow or scikit-learn. Every forward pass, every gradient and
every optimizer update is written by hand in NumPy (~1,100 lines including plotting), verified with
numerical gradient checking, and demonstrated on six experiments — including
**98.27 % test accuracy on MNIST** handwritten digits.

Even the charts below are drawn by a dependency-free SVG/PNG plotter written
for this project.

<p align="center">
  <img src="docs/images/spiral_linear.svg" width="45%" alt="linear model on spiral data">
  <img src="docs/images/spiral_mlp.svg" width="45%" alt="MLP on spiral data">
</p>
<p align="center"><em>Three interleaved spirals. A linear model can only draw straight lines (50.7% test accuracy);
a 2-hidden-layer network learns the curved boundary (99.3%).</em></p>

---

## Features

| | |
|---|---|
| **Layers** | `Dense` (He / Xavier init, L2 weight decay), `Dropout` (inverted) |
| **Activations** | `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh` (numerically stable) |
| **Losses** | `SoftmaxCrossEntropy` (with label smoothing), `BinaryCrossEntropy`, `MSE` — fused with their output activation for stability |
| **Optimizers** | `SGD`, momentum, Nesterov, `RMSProp`, `Adam`; global-norm gradient clipping |
| **Training** | mini-batch loop, validation tracking, early stopping with best-weight restore, `StepDecay` / `CosineDecay` LR schedules |
| **Utilities** | save/load weights (`.npz`), model summary, accuracy, confusion matrix, precision/recall/F1 report |
| **Data** | XOR, spirals, moons, circles, sine regression, MNIST loader, train/test split, standardisation |
| **Verification** | finite-difference gradient checker; 26 unit tests |
| **Plots** | loss curves, decision boundaries, regression fits, confusion matrices — pure-Python SVG + PNG encoder |

## Quick start

```bash
git clone https://github.com/Larp321/neural-network-from-scratch.git
cd neural-network-from-scratch
pip install -r requirements.txt          # just numpy
python examples/spiral.py                # train + write plots to docs/images/
python -m unittest discover -s tests -v  # run the test-suite
```

```python
import numpy as np
from tinynet import Sequential, Dense, ReLU, SoftmaxCrossEntropy, Adam
from tinynet.datasets import make_moons

X, y = make_moons(500, noise=0.2)
model = Sequential([Dense(2, 32), ReLU(), Dense(32, 32), ReLU(), Dense(32, 2)])
history = model.fit(X, y, SoftmaxCrossEntropy(), Adam(lr=0.01), epochs=100, batch_size=32)
print(model.summary())
print("accuracy:", np.mean(model.predict(X) == y))
```

## Experiments

All numbers below come from running the scripts in [`examples/`](examples) with fixed random seeds.

### 1. XOR — why hidden layers matter — [`examples/xor.py`](examples/xor.py)
XOR isn't linearly separable. A 2-4-1 tanh network (17 parameters) solves it in ~50 epochs:

| x₁ | x₂ | target | P(y=1) |
|:-:|:-:|:-:|:-:|
| 0 | 0 | 0 | 0.000 |
| 0 | 1 | 1 | 0.998 |
| 1 | 0 | 1 | 0.998 |
| 1 | 1 | 0 | 0.002 |

<p align="center"><img src="docs/images/xor_boundary.svg" width="40%"> <img src="docs/images/xor_loss.svg" width="50%"></p>

### 2. Spirals — linear vs deep — [`examples/spiral.py`](examples/spiral.py)
600 points, 3 classes, 75/25 split. Network: `2 → 64 → 64 → 3`, ReLU, L2 = 1e-4, Adam.

| model | test accuracy |
|---|---|
| Linear (softmax regression) | 50.7 % |
| MLP 2-64-64-3 (4,547 params) | **99.3 %** |

<p align="center"><img src="docs/images/spiral_loss.svg" width="48%"> <img src="docs/images/spiral_accuracy.svg" width="48%"></p>

### 3. Optimizer comparison — [`examples/compare_optimizers.py`](examples/compare_optimizers.py)
Identical architecture and initial weights; only the optimizer changes.

| optimizer | final training loss (200 epochs) |
|---|---|
| SGD | 0.0429 |
| SGD + momentum | 0.0261 |
| RMSProp | 0.0104 |
| Adam | 0.0114 |

<p align="center"><img src="docs/images/optimizers.svg" width="70%"></p>

Adaptive methods (RMSProp, Adam) reach a ~4× lower loss than plain SGD in the same number of steps.

### 4. Overfitting & regularization — [`examples/regularization.py`](examples/regularization.py)
A 2-256-256-2 network (~67k parameters) trained on only **80** noisy points, validated on 920.

| setting | train acc | validation acc | final val loss |
|---|---|---|---|
| no regularization | 98.8 % | 82.4 % | 0.911 |
| L2 weight decay (λ = 1e-2) | 87.5 % | **85.9 %** | **0.340** |
| dropout (p = 0.5) | 90.0 % | 85.1 % | 0.489 |

Without regularization the network memorises the training set — note the islands
drawn around individual noisy points — and its validation loss climbs after epoch 7.
L2 and dropout trade a little training accuracy for better generalisation.

<p align="center">
  <img src="docs/images/reg_no.svg" width="32%"> <img src="docs/images/reg_l2.svg" width="32%"> <img src="docs/images/reg_dropout.svg" width="32%">
</p>
<p align="center"><img src="docs/images/regularization.svg" width="70%"></p>

### 5. Regression — [`examples/regression.py`](examples/regression.py)
Fitting `sin(x)` from 200 noisy samples (noise σ = 0.15) with a 1-32-32-1 tanh network,
MSE loss and cosine learning-rate decay. RMSE against the true function: **0.019** —
the network recovers the signal, not the noise.

<p align="center"><img src="docs/images/regression.svg" width="70%"></p>

### 6. MNIST handwritten digits — [`examples/mnist.py`](examples/mnist.py)
The real-world benchmark: 70,000 28×28 images of handwritten digits.
Network `784 → 256 → 128 → 10` (235,146 parameters) with ReLU, dropout 0.2, label
smoothing 0.05, Adam + step learning-rate decay and early stopping.
Trained on 55,000 images, validated on 5,000, tested on the 10,000 unseen test images.

| | |
|---|---|
| **Test accuracy** | **98.27 %** (9,827 / 10,000) |
| Training time | ~2 minutes for 12 epochs on a laptop CPU (≈9 s/epoch) |
| Per-class F1 | 0.978 (digit 9) – 0.992 (digit 1) |

```bash
python examples/mnist.py          # downloads the dataset (~11 MB) on first run
```

<p align="center"><img src="docs/images/mnist_confusion.svg" width="48%"> <img src="docs/images/mnist_accuracy.svg" width="48%"></p>

**Where it fails.** The first 20 misclassified test digits. Many are hard even for a
person: a 4 that looks like a 2, an open 6 read as a 0, a 7 with a heavy hook:

<p align="center"><img src="docs/images/mnist_mistakes.png" width="80%" alt="misclassified MNIST digits"></p>

The most common confusions in the matrix (4→9, 9→4, 3→5, 7→2) are the classic MNIST look-alike pairs.

**What the first layer learned.** Each tile shows one hidden neuron's 784 input weights
as a 28×28 image (red = positive, blue = negative). Neurons develop stroke- and
curve-shaped detectors. Pixels at the border are always blank in MNIST, so their weights
never receive a gradient and keep their random initial values (the speckled frame).

<p align="center"><img src="docs/images/mnist_weights.png" width="80%" alt="first-layer weights"></p>

## How it works

The design mirrors the big frameworks at a size you can read in one sitting:

```
tinynet/
├── layers.py      Dense, activations, Dropout — each has forward() and backward()
├── losses.py      loss value + gradient w.r.t. the network output
├── model.py       Sequential: forward, backward (reverse order), fit/evaluate/predict, save/load
├── optimizers.py  SGD / Momentum / Nesterov / RMSProp / Adam + LR schedules
├── gradcheck.py   numerical gradient checking
├── metrics.py     accuracy, confusion matrix, precision / recall / F1
├── datasets.py    toy datasets, preprocessing, MNIST loader
└── viz.py         SVG charts + a tiny PNG encoder (no matplotlib)
```

One training step:

```python
out  = model.forward(x_batch, training=True)     # layer by layer
loss = loss_fn.forward(out, y_batch)             # scalar
grad = loss_fn.backward()                        # dL/d(output)
model.backward(grad)                             # chain rule, reverse order; fills layer.grads
optimizer.step(model.trainable_layers())         # update every parameter
```

The full derivation of every gradient (dense layer, activations, fused
softmax-cross-entropy, Adam, etc.) is in **[docs/MATH.md](docs/MATH.md)**.

### Correctness: gradient checking
Backprop bugs often fail silently — the loss still goes down, just worse. So each
layer, activation and loss is tested against central finite differences:

```python
from tinynet.gradcheck import check_model_gradients
errors = check_model_gradients(model, X, y, SoftmaxCrossEntropy())
# {'0.W': 8.2e-10, '0.b': 2.1e-09, '2.W': 1.0e-09, '2.b': 4.4e-11}   all < 1e-6
```

### Numerical stability
* softmax and cross-entropy use the log-sum-exp trick;
* binary cross-entropy uses `max(z,0) - z·y + log(1+e^{-|z|})`;
* sigmoid switches formulas by sign so `exp` never overflows;
* tests feed logits of ±1000 to confirm results stay finite.

## What I learned
* Deriving and implementing backpropagation by hand, and proving it correct with gradient checks.
* Why initialisation (He vs Xavier), optimizer choice and regularization matter — measured, not just read about.
* Writing numerically stable maths and a clean, tested, documented Python API.

## Possible extensions
Batch normalisation · convolutional layers · a computational-graph autograd engine · GPU support via CuPy.

## License
MIT
