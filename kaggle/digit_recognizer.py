"""Kaggle "Digit Recognizer" submission, trained with tinynet (no ML libraries).

    1. Download the competition data (train.csv, test.csv) into data/kaggle/
    2. python kaggle/digit_recognizer.py
    3. Upload data/kaggle/submission.csv on the competition's "Submit Predictions" page

Honest-score rule: Kaggle's test images are drawn from MNIST, so this script
trains ONLY on Kaggle's train.csv - never on the MNIST files or the demo model,
which have seen some of the test digits.

Recipe: an ensemble of 3 MLPs (784-512-256-10, ReLU, dropout), each trained with
Adam + cosine learning-rate decay on augmented data (random shifts, small
rotations and scaling), with predictions averaged at the end.
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tinynet import Adam, CosineDecay, Dense, Dropout, ReLU, Sequential, SoftmaxCrossEntropy, softmax  # noqa: E402
from tinynet.metrics import accuracy  # noqa: E402

DATA = os.path.join(ROOT, "data", "kaggle")


def load_csv(name):
    path = os.path.join(DATA, name)
    cache = path.replace(".csv", ".npy")
    if os.path.exists(cache):
        return np.load(cache)
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int32)
    np.save(cache, arr)
    return arr


def augment(X, rng, max_shift=2, max_rot=10, scale=(0.9, 1.1)):
    """Random affine distortion of 28x28 images (nearest-neighbour, vectorised)."""
    n = len(X)
    imgs = X.reshape(n, 28, 28)
    th = np.deg2rad(rng.uniform(-max_rot, max_rot, n))
    s = rng.uniform(*scale, n)
    tx, ty = rng.integers(-max_shift, max_shift + 1, (2, n))
    yy, xx = np.mgrid[0:28, 0:28] - 13.5
    c, sn = np.cos(th)[:, None, None], np.sin(th)[:, None, None]
    s = s[:, None, None]
    # inverse map: output pixel -> source pixel
    src_x = (c * xx + sn * yy) / s + 13.5 - tx[:, None, None]
    src_y = (-sn * xx + c * yy) / s + 13.5 - ty[:, None, None]
    sx, sy = np.rint(src_x).astype(int), np.rint(src_y).astype(int)
    ok = (sx >= 0) & (sx < 28) & (sy >= 0) & (sy < 28)
    out = np.zeros_like(imgs)
    idx = np.nonzero(ok)
    out[idx] = imgs[idx[0], sy[ok], sx[ok]]
    return out.reshape(n, 784)


def build(rng):
    return Sequential([
        Dense(784, 512, rng=rng), ReLU(), Dropout(0.3, rng=rng),
        Dense(512, 256, rng=rng), ReLU(), Dropout(0.3, rng=rng),
        Dense(256, 10, rng=rng),
    ])


def train_one(X, y, Xv, yv, epochs, seed):
    rng = np.random.default_rng(seed)
    model, opt = build(rng), Adam(lr=1e-3)
    loss = SoftmaxCrossEntropy(label_smoothing=0.05)
    schedule = CosineDecay(1e-3, epochs, min_lr=2e-5)
    for epoch in range(epochs):
        opt.lr = schedule(epoch)
        t = time.perf_counter()
        Xa = augment(X, rng) if epoch < epochs - 2 else X  # last 2 epochs on clean data
        h = model.fit(Xa, y, loss, opt, epochs=1, batch_size=128, rng=rng, verbose=0)
        msg = f"  model {seed} epoch {epoch + 1:>2}/{epochs}  loss {h['loss'][-1]:.4f}"
        if Xv is not None:
            msg += f"  val_acc {accuracy(model.forward(Xv), yv):.4f}"
        print(msg + f"  ({time.perf_counter() - t:.0f}s)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--models", type=int, default=3)
    ap.add_argument("--no-validation", action="store_true",
                    help="train on all 42,000 images (use for the final submission)")
    args = ap.parse_args()

    train = load_csv("train.csv")
    test = load_csv("test.csv").astype(np.float32) / 255.0
    y_all, X_all = train[:, 0], train[:, 1:].astype(np.float32) / 255.0
    print(f"train {X_all.shape}  test {test.shape}")

    if args.no_validation:
        X, y, Xv, yv = X_all, y_all, None, None
    else:  # hold out 4,200 images to measure accuracy honestly
        idx = np.random.default_rng(0).permutation(len(X_all))
        v, t = idx[:4200], idx[4200:]
        X, y, Xv, yv = X_all[t], y_all[t], X_all[v], y_all[v]

    models = [train_one(X, y, Xv, yv, args.epochs, seed) for seed in range(args.models)]

    def ensemble(A):
        return np.mean([softmax(m.forward(A)) for m in models], axis=0)

    if Xv is not None:
        for i, m in enumerate(models):
            print(f"model {i} validation accuracy: {accuracy(m.forward(Xv), yv):.4%}")
        print(f"ENSEMBLE validation accuracy: {np.mean(ensemble(Xv).argmax(1) == yv):.4%}")

    pred = ensemble(test).argmax(1)
    out = os.path.join(DATA, "submission.csv")
    with open(out, "w", newline="") as f:
        f.write("ImageId,Label\n")
        f.writelines(f"{i},{p}\n" for i, p in enumerate(pred, start=1))
    print(f"wrote {out} ({len(pred)} predictions); class counts {np.bincount(pred, minlength=10).tolist()}")


if __name__ == "__main__":
    main()
