"""Train the web-demo model on MNIST + synthetic digits in under-represented writing styles.

    python web/train_demo_model.py      # ~5-8 min on a laptop CPU; writes data/demo_mlp.npz
    python web/export_model.py data/demo_mlp.npz

The synthetic "style test set" uses a different random seed from the training
styles, so it measures whether each *style* is learned, not memorised samples.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tinynet import Adam, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy, StepDecay  # noqa: E402
from tinynet.datasets import load_mnist  # noqa: E402
from tinynet.handwriting import STYLES, make_styled_digits  # noqa: E402

DATA = os.path.join(ROOT, "data")


def styled(name, n, seed):
    path = os.path.join(DATA, f"styled_{name}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["X"], d["y"], d["s"]
    t = time.perf_counter()
    X, y, s = make_styled_digits(n, rng=np.random.default_rng(seed), return_styles=True)
    np.savez_compressed(path, X=X, y=y, s=s)
    print(f"generated {len(X)} styled digits ({name}) in {time.perf_counter() - t:.0f}s")
    return X, y, s


def build(rng):
    return Sequential([
        Dense(784, 256, rng=rng), ReLU(), Dropout(0.2, rng=rng),
        Dense(256, 128, rng=rng), ReLU(), Dropout(0.2, rng=rng),
        Dense(128, 10, rng=rng),
    ])


def per_style_accuracy(model, X, y, s):
    pred = model.predict(X)
    return {name: float(np.mean(pred[s == name] == y[s == name])) for name in STYLES}


def main():
    X_train, y_train, X_test, y_test = load_mnist(os.path.join(DATA, "mnist"), download=False)
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]

    Xs_train, ys_train, _ = styled("train", 500, seed=0)    # 13,000 digits
    Xs_val, ys_val, _ = styled("val", 40, seed=1)
    Xs_test, ys_test, ss_test = styled("test", 100, seed=2)  # 2,600 held-out digits

    # ---- baseline: the MNIST-only model
    baseline = build(np.random.default_rng(0))
    baseline.load(os.path.join(DATA, "mnist_mlp.npz"))
    before = per_style_accuracy(baseline, Xs_test, ys_test, ss_test)
    before_mnist = float(np.mean(baseline.predict(X_test) == y_test))

    # ---- retrain on MNIST + styled digits
    rng = np.random.default_rng(0)
    X = np.concatenate([X_train, Xs_train])
    y = np.concatenate([y_train, ys_train])
    Xv = np.concatenate([X_val, Xs_val])
    yv = np.concatenate([y_val, ys_val])
    model = build(rng)
    model.fit(X, y, SoftmaxCrossEntropy(label_smoothing=0.05), Adam(lr=1e-3), epochs=12, batch_size=128,
              validation_data=(Xv, yv), lr_schedule=StepDecay(1e-3, drop=0.5, every=4),
              early_stopping=4, rng=rng)
    after = per_style_accuracy(model, Xs_test, ys_test, ss_test)
    after_mnist = float(np.mean(model.predict(X_test) == y_test))
    model.save(os.path.join(DATA, "demo_mlp.npz"))

    print(f"\n{'style':<26}{'MNIST-only':>12}{'+ styles':>10}")
    for name in STYLES:
        flag = "  <-- fixed" if before[name] < 0.9 <= after[name] else ""
        print(f"{name:<26}{before[name]:>11.1%}{after[name]:>10.1%}{flag}")
    avg_b, avg_a = np.mean(list(before.values())), np.mean(list(after.values()))
    print(f"{'all styles (2,600 digits)':<26}{avg_b:>11.1%}{avg_a:>10.1%}")
    print(f"{'MNIST test (10,000)':<26}{before_mnist:>11.2%}{after_mnist:>10.2%}")


if __name__ == "__main__":
    main()
