"""Kaggle "Digit Recognizer" submission, trained with tinynet (no ML libraries).

    1. Download the competition data (train.csv, test.csv) into data/kaggle/
    2. Train models (one process per seed uses several CPU cores at once):
           python kaggle/digit_recognizer.py --seeds 0 &
           python kaggle/digit_recognizer.py --seeds 1 &
           python kaggle/digit_recognizer.py --seeds 2 &
    3. python kaggle/digit_recognizer.py --combine      # -> data/kaggle/submission.csv

    Weights are checkpointed after every epoch. To stop early, create the file
    data/kaggle/runs/STOP: each run finishes its current epoch and saves its
    predictions. (Or kill it and use --from-checkpoint --seeds N to predict
    from the last saved epoch.)
    4. Upload data/kaggle/submission.csv on the competition's "Submit Predictions" page

Honest-score rule: Kaggle's test images are drawn from MNIST, so this script
trains ONLY on Kaggle's train.csv - never on the MNIST files or the demo model,
which have seen some of the test digits.

Recipe: an ensemble of networks - a CNN (default, --arch cnn) or an MLP
(--arch mlp) - each trained with Adam + cosine learning-rate decay on augmented
data (random shifts, small rotations and scaling), predictions averaged.

    CNN: [conv3x3(32) relu conv3x3(32) relu maxpool] -> [conv3x3(64) relu conv3x3(64) relu maxpool]
         -> dense 256 relu dropout -> dense 10

"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tinynet import (Adam, Conv2D, CosineDecay, Dense, Dropout, Flatten, MaxPool2D, ReLU,  # noqa: E402
                     Reshape, Sequential, SoftmaxCrossEntropy, softmax)
from tinynet.metrics import accuracy  # noqa: E402

DATA = os.path.join(ROOT, "data", "kaggle")
RUNS = os.path.join(DATA, "runs")


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


def build_cnn(rng):
    f = np.float32
    return Sequential([
        Reshape((1, 28, 28)),
        Conv2D(1, 32, 3, padding=1, rng=rng, dtype=f, input_grad=False), ReLU(),
        Conv2D(32, 32, 3, padding=1, rng=rng, dtype=f), ReLU(), MaxPool2D(2), Dropout(0.1, rng=rng),
        Conv2D(32, 64, 3, padding=1, rng=rng, dtype=f), ReLU(),
        Conv2D(64, 64, 3, padding=1, rng=rng, dtype=f), ReLU(), MaxPool2D(2), Dropout(0.2, rng=rng),
        Flatten(), Dense(64 * 7 * 7, 256, rng=rng, dtype=f), ReLU(), Dropout(0.4, rng=rng),
        Dense(256, 10, rng=rng, dtype=f),
    ], eval_batch_size=500)


def build_mlp(rng):
    return Sequential([
        Dense(784, 512, rng=rng), ReLU(), Dropout(0.3, rng=rng),
        Dense(512, 256, rng=rng), ReLU(), Dropout(0.3, rng=rng),
        Dense(256, 10, rng=rng),
    ])


def train_one(X, y, Xv, yv, epochs, seed, arch="cnn"):
    rng = np.random.default_rng(seed)
    model, opt = (build_cnn if arch == "cnn" else build_mlp)(rng), Adam(lr=1e-3)
    loss = SoftmaxCrossEntropy(label_smoothing=0.05)
    schedule = CosineDecay(1e-3, epochs, min_lr=2e-5)
    for epoch in range(epochs):
        opt.lr = schedule(epoch)
        t = time.perf_counter()
        Xa = augment(X, rng) if epoch < epochs - 2 else X  # last 2 epochs on clean data
        h = model.fit(Xa, y, loss, opt, epochs=1, batch_size=128, rng=rng, verbose=0, train_accuracy=False)
        msg = f"  model {seed} epoch {epoch + 1:>2}/{epochs}  loss {h['loss'][-1]:.4f}"
        if Xv is not None:
            msg += f"  val_acc {accuracy(model.predict_logits(Xv), yv):.4f}"
        print(msg + f"  ({time.perf_counter() - t:.0f}s)", flush=True)
        # checkpoint every epoch, so a long run can be stopped without losing it
        model.save(os.path.join(RUNS, f"{arch}_seed{seed}_checkpoint.npz"))
        if os.path.exists(os.path.join(RUNS, "STOP")):
            print(f"  STOP file found - finishing model {seed} after epoch {epoch + 1}", flush=True)
            break
    return model


def write_submission(probs, path):
    pred = probs.argmax(1)
    with open(path, "w", newline="") as f:
        f.write("ImageId,Label\n")
        f.writelines(f"{i},{p}\n" for i, p in enumerate(pred, start=1))
    print(f"wrote {path} ({len(pred)} predictions); class counts {np.bincount(pred, minlength=10).tolist()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", choices=["cnn", "mlp"], default="cnn")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="train one model per seed (run several processes in parallel to use more CPU cores)")
    ap.add_argument("--combine", action="store_true",
                    help="average every saved model of --arch into data/kaggle/submission.csv")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N training images (quick tests)")
    ap.add_argument("--from-checkpoint", action="store_true",
                    help="skip training: load each seed's last checkpoint and save its predictions")
    args = ap.parse_args()
    runs = RUNS
    os.makedirs(runs, exist_ok=True)

    train = load_csv("train.csv")
    y_all, X_all = train[:, 0], train[:, 1:].astype(np.float32) / 255.0
    # hold out the same 4,200 images in every run to measure accuracy honestly
    idx = np.random.default_rng(0).permutation(len(X_all))
    v, t = idx[:4200], idx[4200:]
    X, y, Xv, yv = X_all[t], y_all[t], X_all[v], y_all[v]

    if args.combine:
        files = sorted(f for f in os.listdir(runs) if f.startswith(args.arch + "_seed"))
        probs = [np.load(os.path.join(runs, f)) for f in files]
        for f, p in zip(files, probs):
            print(f"{f:<24} validation accuracy {np.mean(p['val'].argmax(1) == yv):.4%}")
        val = np.mean([p["val"] for p in probs], axis=0)
        print(f"ENSEMBLE of {len(probs)} validation accuracy: {np.mean(val.argmax(1) == yv):.4%}")
        write_submission(np.mean([p["test"] for p in probs], axis=0), os.path.join(DATA, "submission.csv"))
        return

    test = load_csv("test.csv").astype(np.float32) / 255.0
    if args.limit:
        X, y = X[:args.limit], y[:args.limit]
    print(f"train {X.shape}  validation {Xv.shape}  test {test.shape}", flush=True)
    for seed in args.seeds:
        if args.from_checkpoint:
            model = (build_cnn if args.arch == "cnn" else build_mlp)(np.random.default_rng(seed))
            model.load(os.path.join(runs, f"{args.arch}_seed{seed}_checkpoint.npz"))
        else:
            model = train_one(X, y, Xv, yv, args.epochs, seed, args.arch)
        val, tst = softmax(model.predict_logits(Xv)), softmax(model.predict_logits(test))
        np.savez(os.path.join(runs, f"{args.arch}_seed{seed}.npz"), val=val, test=tst)
        model.save(os.path.join(runs, f"{args.arch}_seed{seed}_weights.npz"))
        print(f"seed {seed}: validation accuracy {np.mean(val.argmax(1) == yv):.4%}", flush=True)


if __name__ == "__main__":
    main()
