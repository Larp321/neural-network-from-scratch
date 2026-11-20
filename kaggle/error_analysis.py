"""Look at every validation digit the ensemble gets wrong, and sort the mistakes by kind.

    python kaggle/error_analysis.py            # after training + --combine

Writes docs/images/kaggle_errors.svg (every mistake with its true label, the
prediction and confidence) and prints a breakdown:

* confusion pairs   - which digits get mistaken for which
* confident errors  - wrong with >90% confidence: often a mislabelled image or a
                      digit that genuinely looks like the other class
* unsure errors     - top guess under 60%: the model "knew" it was hard
* disagreements     - the models in the ensemble did not agree with each other
* near misses       - the true label was the second guess
"""
import base64
import os
import re
import sys
from collections import Counter
from html import escape

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tinynet.viz import FONT, png_bytes  # noqa: E402

DATA = os.path.join(ROOT, "data", "kaggle")
RUNS = os.path.join(DATA, "runs")


def load(arch="cnn"):
    train = np.load(os.path.join(DATA, "train.npy"))
    y_all, X_all = train[:, 0], train[:, 1:].astype(np.float32) / 255.0
    idx = np.random.default_rng(0).permutation(len(X_all))   # same split as digit_recognizer.py
    v = idx[:4200]
    files = sorted(f for f in os.listdir(RUNS) if re.fullmatch(rf"{arch}_seed\d+\.npz", f))
    per_model = np.stack([np.load(os.path.join(RUNS, f))["val"] for f in files])
    return X_all[v], y_all[v], per_model, files


def tile_png(img, scale=3):
    g = (255 * (1 - img.reshape(28, 28))).astype(np.uint8)
    g = np.kron(g, np.ones((scale, scale), dtype=np.uint8))
    return base64.b64encode(png_bytes(np.repeat(g[:, :, None], 3, axis=2))).decode()


def gallery(X, y, probs, errs, path, cols=10):
    cell_w, cell_h, img = 96, 124, 84
    rows = -(-len(errs) // cols)
    w, h = cols * cell_w + 20, rows * cell_h + 60
    out = [f"<rect width='{w}' height='{h}' fill='white'/>",
           f"<text x='10' y='28' font-size='16' font-weight='600' fill='#222' {FONT}>"
           f"All {len(errs)} validation mistakes (true → predicted, confidence)</text>"]
    for k, i in enumerate(errs):
        r, c = divmod(k, cols)
        x0, y0 = 10 + c * cell_w, 44 + r * cell_h
        p = probs[i]
        pred, conf = int(p.argmax()), float(p.max())
        colour = "#c8312a" if conf > 0.9 else "#b8860b" if conf > 0.6 else "#555"
        out.append(f"<image x='{x0 + 6}' y='{y0}' width='{img}' height='{img}' href='data:image/png;base64,{tile_png(X[i])}'/>")
        out.append(f"<rect x='{x0 + 6}' y='{y0}' width='{img}' height='{img}' fill='none' stroke='#ccc'/>")
        out.append(f"<text x='{x0 + 48}' y='{y0 + img + 17}' text-anchor='middle' font-size='14' "
                   f"font-weight='700' fill='{colour}' {FONT}>{y[i]} → {pred}  {conf:.0%}</text>")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}'>\n"
                + "\n".join(out) + "\n</svg>\n")


def main():
    X, y, per_model, files = load()
    probs = per_model.mean(axis=0)
    pred = probs.argmax(1)
    errs = np.flatnonzero(pred != y)
    errs = errs[np.argsort(-probs[errs].max(1))]   # most confident mistakes first
    n = len(y)
    print(f"{len(files)} models, {n} validation images, {len(errs)} mistakes -> accuracy {1 - len(errs) / n:.4%}\n")

    conf = probs[errs].max(1)
    second = np.argsort(-probs[errs], axis=1)[:, 1]
    model_preds = per_model.argmax(2)[:, errs]           # (models, errors)
    disagree = (model_preds != model_preds[0]).any(axis=0)
    some_right = (model_preds == y[errs]).any(axis=0)

    print("Most common confusions (true -> predicted):")
    for (t, p), c in Counter(zip(y[errs].tolist(), pred[errs].tolist())).most_common(10):
        print(f"  {t} -> {p}: {c}")
    print(f"\nconfident errors (>90%):        {np.sum(conf > 0.9):>3}")
    print(f"medium (60-90%):                {np.sum((conf > 0.6) & (conf <= 0.9)):>3}")
    print(f"unsure errors (<60%):           {np.sum(conf <= 0.6):>3}")
    print(f"true label was the 2nd guess:   {np.sum(second == y[errs]):>3}")
    print(f"models disagreed:               {np.sum(disagree):>3}  (at least one model was right on {np.sum(some_right)})")
    per_class = np.bincount(y[errs], minlength=10) / np.bincount(y, minlength=10)
    print("\nerror rate per true digit: " + "  ".join(f"{d}:{r:.1%}" for d, r in enumerate(per_class)))
    for f, pm in zip(files, per_model):
        print(f"{f}: {np.mean(pm.argmax(1) == y):.4%}")

    out = os.path.join(ROOT, "docs", "images", "kaggle_errors.svg")
    gallery(X, y, probs, errs, out)
    print(f"\nwrote {out}")
    np.save(os.path.join(RUNS, "val_errors.npy"), np.c_[errs, y[errs], pred[errs]])


if __name__ == "__main__":
    main()
