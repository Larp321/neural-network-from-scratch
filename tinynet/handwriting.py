"""Synthetic hand-drawn digits in writing styles that MNIST under-represents.

MNIST was written mostly by American writers, so e.g. the European "1" with a
long flag, the crossed 7 or the slashed 0 are rare in it, and a model trained
only on MNIST misreads them. This module draws digits from stroke templates
(several styles per digit), randomises them like real handwriting (slant,
rotation, aspect, size, stroke width, wobble), rasterises them and applies the
same MNIST-style preprocessing as the web demo (``web/app.js``):

    crop to the ink -> scale the longer side to 20 px (area average)
    -> paste into 28x28 -> shift the centre of mass to the middle.
"""

from __future__ import annotations

import numpy as np

TAU = 2 * np.pi
Stroke = np.ndarray  # (n, 2) array of (x, y) points, y pointing down, digit height ~1


def _arc(cx, cy, rx, ry, a0, a1, n=28) -> Stroke:
    a = np.linspace(a0, a1, n)
    return np.c_[cx + rx * np.cos(a), cy + ry * np.sin(a)]


def _pts(*p) -> Stroke:
    return np.array(p, dtype=float)


def _cat(*parts) -> Stroke:
    return np.vstack(parts)


def _u(rng, lo, hi):
    return rng.uniform(lo, hi)


# --------------------------------------------------------------------------- #
# Style templates: each returns a list of strokes for one randomised instance.
# --------------------------------------------------------------------------- #
def _zero_oval(r):
    a0 = -np.pi / 2 + _u(r, -0.6, 0.6)
    return [_arc(0.5, 0.5, _u(r, .24, .32), .48, a0, a0 + TAU + _u(r, 0, .3), 40)]


def _zero_slashed(r):
    return _zero_oval(r) + [_pts((_u(r, .7, .85), _u(r, .05, .2)), (_u(r, .15, .3), _u(r, .8, .95)))]


def _one_plain(r):
    return [_pts((0.5 + _u(r, -.04, .04), 0), (0.5, 1))]


def _one_flag(r):
    return [_pts((_u(r, .3, .42), _u(r, .15, .3)), (0.55, 0), (0.5, 1))]


def _one_flag_base(r):
    return _one_flag(r) + [_pts((_u(r, .25, .35), 1), (_u(r, .65, .78), 1))]


def _one_european(r):  # long flag, sometimes nearly half the height
    return [_pts((_u(r, .15, .3), _u(r, .3, .5)), (0.6, 0), (0.55, 1))]


def _one_base(r):
    return _one_plain(r) + [_pts((_u(r, .25, .35), 1), (_u(r, .65, .75), 1))]


def _two_plain(r):
    top = _arc(0.5, 0.27, 0.27, 0.25, np.pi * _u(r, 1.0, 1.2), np.pi * 2.25)
    return [_cat(top, _pts((0.22, 1), (_u(r, .75, .85), 1)))]


def _two_loop(r):
    top = _arc(0.5, 0.27, 0.27, 0.25, np.pi * 1.1, np.pi * 2.25)
    loop = _pts((0.24, 0.97), (0.18, 0.88), (0.27, 0.83), (0.36, 0.93), (0.46, 1.0), (0.82, 0.99))
    return [_cat(top, loop)]


def _three_round(r):
    return [_cat(_arc(0.5, 0.25, 0.25, 0.24, np.pi * _u(r, 1.0, 1.2), np.pi * 2.5),
                 _arc(0.5, 0.74, _u(r, .26, .31), 0.26, np.pi * 1.5, np.pi * 2.85))]


def _three_flat(r):
    return [_cat(_pts((0.22, 0.02), (0.76, 0.02), (0.45, 0.42)),
                 _arc(0.5, 0.72, 0.28, 0.28, np.pi * 1.4, np.pi * 2.85))]


def _four_closed(r):
    t = _u(r, .6, .72)
    return [_pts((0.62, 0), (0.15, t), (0.86, t)), _pts((0.62, 0), (0.62, 1))]


def _four_open(r):
    t = _u(r, .55, .68)
    return [_pts((_u(r, .25, .35), 0), (0.18, t), (0.86, t)), _pts((0.66, _u(r, .05, .3)), (0.63, 1))]


def _four_one_stroke(r):
    return [_pts((0.86, 0.68), (0.15, 0.68), (0.62, 0), (0.62, 1))]


def _five(r):
    bowl = _arc(0.48, 0.68, 0.3, 0.3, np.pi * 1.2, np.pi * _u(r, 2.65, 2.85))
    if r.random() < 0.5:  # bar drawn first, one stroke
        return [_cat(_pts((0.76, 0), (0.3, 0), (0.25, 0.42)), bowl)]
    return [_cat(_pts((0.3, 0), (0.25, 0.42)), bowl), _pts((0.3, 0), (_u(r, .7, .82), 0))]


def _six_curved(r):
    return [_cat(_pts((0.7, 0.02), (0.5, 0.15), (0.33, 0.4)),
                 _arc(0.5, 0.72, 0.25, 0.26, np.pi, np.pi * 3))]


def _six_straight(r):
    return [_cat(_pts((0.64, 0), (0.25, 0.72)), _arc(0.5, 0.72, 0.25, 0.26, np.pi, np.pi * 3))]


def _seven_plain(r):
    return [_pts((0.2, 0.02), (0.8, 0.02), (_u(r, .35, .5), 1))]


def _seven_crossed(r):
    y = _u(r, .45, .58)
    return _seven_plain(r) + [_pts((0.32, y), (0.78, y))]


def _seven_serif(r):
    return [_pts((0.2, _u(r, .12, .22)), (0.2, 0.02), (0.8, 0.02), (0.42, 1))]


def _seven_curved(r):
    return [_pts((0.2, 0.02), (0.8, 0.02), (0.62, 0.35), (0.5, 0.65), (0.45, 1))]


def _eight_two_loops(r):
    return [_arc(0.5, 0.26, 0.22, 0.24, np.pi / 2, np.pi / 2 + TAU),
            _arc(0.5, 0.73, 0.27, 0.27, -np.pi / 2, 1.5 * np.pi)]


def _eight_continuous(r):
    t = np.linspace(0, TAU + 0.2, 60)
    w = _u(r, .22, .3)
    return [np.c_[0.5 + w * np.sin(2 * t), 0.5 - 0.5 * np.cos(t)]]


def _nine_straight(r):
    return [_arc(0.48, 0.27, 0.25, 0.25, 0, TAU), _pts((0.73, 0.27), (_u(r, .62, .74), 1))]


def _nine_one_stroke(r):
    return [_cat(_arc(0.48, 0.27, 0.25, 0.25, 0, -TAU), _pts((0.68, 1)))]


def _nine_curved(r):
    return [_arc(0.48, 0.27, 0.25, 0.25, 0, TAU), _pts((0.73, 0.27), (0.7, 0.7), (0.55, 0.98), (0.35, 0.95))]


STYLES: dict[str, tuple[int, callable]] = {
    "0 oval": (0, _zero_oval), "0 slashed": (0, _zero_slashed),
    "1 single stroke": (1, _one_plain), "1 with flag": (1, _one_flag),
    "1 flag + base": (1, _one_flag_base), "1 long flag (European)": (1, _one_european),
    "1 with base": (1, _one_base),
    "2 plain": (2, _two_plain), "2 with loop": (2, _two_loop),
    "3 round top": (3, _three_round), "3 flat top": (3, _three_flat),
    "4 closed": (4, _four_closed), "4 open top": (4, _four_open), "4 one stroke": (4, _four_one_stroke),
    "5": (5, _five),
    "6 curved stem": (6, _six_curved), "6 straight stem": (6, _six_straight),
    "7 plain": (7, _seven_plain), "7 crossed": (7, _seven_crossed),
    "7 with serif": (7, _seven_serif), "7 curved stem": (7, _seven_curved),
    "8 two loops": (8, _eight_two_loops), "8 one stroke": (8, _eight_continuous),
    "9 straight stem": (9, _nine_straight), "9 one stroke": (9, _nine_one_stroke),
    "9 curved stem": (9, _nine_curved),
}


# --------------------------------------------------------------------------- #
# Randomisation, rendering and preprocessing
# --------------------------------------------------------------------------- #
CANVAS = 112          # render resolution (the web canvas is 280; same proportions)
_SCALE = CANVAS / 280  # pixels per web-canvas pixel


def _handwrite(strokes: list[Stroke], rng) -> list[Stroke]:
    """Apply slant, rotation, aspect, size, position and smooth wobble; map to canvas pixels."""
    slant = rng.uniform(-0.35, 0.25)
    rot = np.deg2rad(rng.uniform(-10, 10))
    sx = rng.uniform(0.75, 1.25)
    height = rng.uniform(140, 235) * _SCALE
    cx = CANVAS / 2 + rng.uniform(-0.12, 0.12) * CANVAS
    cy = CANVAS / 2 + rng.uniform(-0.1, 0.1) * CANVAS
    c, s = np.cos(rot), np.sin(rot)
    out = []
    for st in strokes:
        p = st.copy()
        # smooth wobble along the stroke
        t = np.linspace(0, 1, len(p))
        for _ in range(2):
            f, ph = rng.uniform(0.5, 2.0), rng.uniform(0, TAU)
            p += rng.normal(0, 0.012, 2) * np.sin(TAU * f * t + ph)[:, None]
        p += rng.normal(0, 0.015, 2)  # whole-stroke offset (pen lifted and placed again)
        x, y = (p[:, 0] - 0.5) * sx, p[:, 1] - 0.5
        x = x + slant * -y
        x, y = c * x - s * y, s * x + c * y
        out.append(np.c_[cx + x * height, cy + y * height])
    return out


def render(strokes: list[Stroke], width: float) -> np.ndarray:
    """Anti-aliased rasterisation of thick round-capped polylines -> (CANVAS, CANVAS) in [0, 1]."""
    segs = np.vstack([np.c_[st[:-1], st[1:]] if len(st) > 1 else np.c_[st, st] for st in strokes])
    ax, ay, bx, by = segs.T
    yy, xx = np.mgrid[0:CANVAS, 0:CANVAS] + 0.5
    P = np.c_[xx.ravel(), yy.ravel()]
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((P[:, 0:1] - ax) * dx + (P[:, 1:2] - ay) * dy) / L2, 0, 1)
    d = np.hypot(P[:, 0:1] - (ax + t * dx), P[:, 1:2] - (ay + t * dy)).min(axis=1)
    return np.clip(width / 2 + 0.5 - d, 0, 1).reshape(CANVAS, CANVAS)


def preprocess(g: np.ndarray) -> np.ndarray | None:
    """MNIST-style normalisation - a line-by-line port of ``preprocess()`` in web/app.js."""
    ys, xs = np.nonzero(g > 0.1)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    s = 20 / max(w, h)
    tw, th = max(1, round(w * s)), max(1, round(h * s))
    crop = g[y0:y1 + 1, x0:x1 + 1]
    if s < 1:
        ty = np.minimum(th - 1, np.floor(np.arange(h) * s).astype(int))
        tx = np.minimum(tw - 1, np.floor(np.arange(w) * s).astype(int))
        total, count = np.zeros((th, tw)), np.zeros((th, tw))
        np.add.at(total, (ty[:, None], tx[None, :]), crop)
        np.add.at(count, (ty[:, None], tx[None, :]), 1)
        small = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    else:
        iy = np.minimum(h - 1, np.floor(np.arange(th) / s).astype(int))
        ix = np.minimum(w - 1, np.floor(np.arange(tw) / s).astype(int))
        small = crop[iy[:, None], ix[None, :]]
    ox, oy = (28 - tw) // 2, (28 - th) // 2
    mass = small.sum()
    yy, xx = np.mgrid[0:th, 0:tw]
    dx = int(np.round(14 - ((xx + ox) * small).sum() / mass))
    dy = int(np.round(14 - ((yy + oy) * small).sum() / mass))
    img = np.zeros((28, 28))
    for y in range(th):
        Y = y + oy + dy
        if not 0 <= Y < 28:
            continue
        X = np.arange(tw) + ox + dx
        ok = (X >= 0) & (X < 28)
        img[Y, X[ok]] = small[y, ok]
    return img


def make_styled_digits(n_per_style: int = 200, styles: list[str] | None = None, rng=None,
                       return_styles: bool = False):
    """Generate ``(X, y)``: X is (N, 784) float32 in [0, 1], like ``load_mnist``.

    With ``return_styles=True`` also returns the style name of every sample.
    """
    rng = rng if rng is not None else np.random.default_rng()
    names = styles or list(STYLES)
    X, y, which = [], [], []
    for name in names:
        label, template = STYLES[name]
        for _ in range(n_per_style):
            strokes = _handwrite(template(rng), rng)
            img = preprocess(render(strokes, width=rng.uniform(12, 28) * _SCALE))
            X.append(img.ravel())
            y.append(label)
            which.append(name)
    X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
    return (X, y, np.asarray(which)) if return_styles else (X, y)
