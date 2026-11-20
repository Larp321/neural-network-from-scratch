"""Dependency-free plotting: writes SVG charts (with embedded PNG rasters).

No matplotlib required - charts are built from scratch with the standard
library (``zlib`` + ``base64`` for the PNG encoder).
"""

from __future__ import annotations

import base64
import math
import struct
import zlib
from html import escape
from typing import Sequence

import numpy as np

from .losses import Loss, probabilities

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
           "#B279A2", "#EECA3B", "#9D755D", "#FF9DA6", "#BAB0AC"]
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"


# --------------------------------------------------------------------------- #
# PNG encoder
# --------------------------------------------------------------------------- #
def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 array as PNG bytes."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def save_png(path: str, rgb: np.ndarray) -> None:
    with open(path, "wb") as f:
        f.write(png_bytes(rgb))


def _hex_to_rgb(h: str) -> np.ndarray:
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


# --------------------------------------------------------------------------- #
# Axis helpers
# --------------------------------------------------------------------------- #
def _nice_ticks(lo: float, hi: float, n: int = 5, integer: bool = False) -> list[float]:
    if hi == lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    if integer:
        step = max(1, round(step if step != 2.5 * mag else 2 * mag))
    start = math.ceil(lo / step) * step
    ticks, t = [], start
    while t <= hi + 1e-9 * step:
        ticks.append(round(t, 10))
        t += step
    return ticks


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 1000 or abs(v) < 0.01:
        return f"{v:.0e}"
    return f"{v:g}"


class _Frame:
    """Maps data coordinates to an SVG plotting rectangle."""

    def __init__(self, width, height, xlim, ylim, margin=(48, 24, 52, 64), log_y=False, int_x=False):
        self.w, self.h, self.int_x = width, height, int_x
        self.top, self.right, self.bottom, self.left = margin
        self.xlim, self.log_y = xlim, log_y
        self.ylim = (math.log10(ylim[0]), math.log10(ylim[1])) if log_y else ylim

    @property
    def pw(self):
        return self.w - self.left - self.right

    @property
    def ph(self):
        return self.h - self.top - self.bottom

    def x(self, v):
        lo, hi = self.xlim
        return self.left + (v - lo) / (hi - lo or 1) * self.pw

    def y(self, v):
        if self.log_y:
            v = math.log10(max(v, 1e-12))
        lo, hi = self.ylim
        return self.top + self.ph - (v - lo) / (hi - lo or 1) * self.ph

    def axes(self, title="", xlabel="", ylabel="") -> list[str]:
        out = [f"<rect width='{self.w}' height='{self.h}' fill='white'/>"]
        xt = _nice_ticks(*self.xlim, integer=self.int_x)
        if self.log_y:
            yt = [10.0 ** e for e in range(math.floor(self.ylim[0]), math.ceil(self.ylim[1]) + 1)]
        else:
            yt = _nice_ticks(*self.ylim)
        for t in yt:
            yy = self.y(t)
            if self.top - 0.5 <= yy <= self.top + self.ph + 0.5:
                out.append(f"<line x1='{self.left}' x2='{self.left + self.pw}' y1='{yy:.1f}' y2='{yy:.1f}' stroke='#e5e5e5'/>")
                out.append(f"<text x='{self.left - 8}' y='{yy + 4:.1f}' text-anchor='end' font-size='11' fill='#555' {FONT}>{_fmt(t)}</text>")
        for t in xt:
            xx = self.x(t)
            if self.left - 0.5 <= xx <= self.left + self.pw + 0.5:
                out.append(f"<line x1='{xx:.1f}' x2='{xx:.1f}' y1='{self.top}' y2='{self.top + self.ph}' stroke='#f0f0f0'/>")
                out.append(f"<text x='{xx:.1f}' y='{self.top + self.ph + 18}' text-anchor='middle' font-size='11' fill='#555' {FONT}>{_fmt(t)}</text>")
        out.append(f"<rect x='{self.left}' y='{self.top}' width='{self.pw}' height='{self.ph}' fill='none' stroke='#999'/>")
        if title:
            out.append(f"<text x='{self.w / 2}' y='26' text-anchor='middle' font-size='15' font-weight='600' fill='#222' {FONT}>{escape(title)}</text>")
        if xlabel:
            out.append(f"<text x='{self.left + self.pw / 2}' y='{self.h - 12}' text-anchor='middle' font-size='12' fill='#333' {FONT}>{escape(xlabel)}</text>")
        if ylabel:
            cy = self.top + self.ph / 2
            out.append(f"<text x='16' y='{cy}' transform='rotate(-90 16 {cy})' text-anchor='middle' font-size='12' fill='#333' {FONT}>{escape(ylabel)}</text>")
        return out


def _legend(names: Sequence[str], colors: Sequence[str], x: float, y: float) -> list[str]:
    out, width = [], max(len(n) for n in names) * 7 + 34
    out.append(f"<rect x='{x - width}' y='{y}' width='{width}' height='{len(names) * 18 + 8}' fill='white' fill-opacity='0.9' stroke='#ccc' rx='4'/>")
    for i, (n, c) in enumerate(zip(names, colors)):
        yy = y + 14 + i * 18
        out.append(f"<line x1='{x - width + 8}' x2='{x - width + 24}' y1='{yy}' y2='{yy}' stroke='{c}' stroke-width='3'/>")
        out.append(f"<text x='{x - width + 30}' y='{yy + 4}' font-size='11' fill='#333' {FONT}>{escape(n)}</text>")
    return out


def _svg(width, height, body) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink' "
            f"width='{width}' height='{height}' viewBox='0 0 {width} {height}'>\n" + "\n".join(body) + "\n</svg>\n")


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------- #
# Public plotting functions
# --------------------------------------------------------------------------- #
def line_chart(series: dict[str, Sequence[float]], path: str, title: str = "", xlabel: str = "epoch",
               ylabel: str = "", log_y: bool = False, width: int = 640, height: int = 400,
               x_start: int = 1) -> None:
    """Plot one line per entry of ``series`` (e.g. training curves) to an SVG file."""
    all_y = np.concatenate([np.asarray(v, dtype=float) for v in series.values()])
    n = max(len(v) for v in series.values())
    if log_y:
        pos = all_y[all_y > 0]
        ylim = (10 ** math.floor(math.log10(pos.min())), 10 ** math.ceil(math.log10(pos.max())))
    else:
        lo, hi = float(all_y.min()), float(all_y.max())
        pad = (hi - lo) * 0.05 or 0.5
        ylim = (min(0.0, lo) if lo >= 0 and lo < (hi - lo) else lo - pad, hi + pad)
    f = _Frame(width, height, (x_start, x_start + max(n - 1, 1)), ylim, log_y=log_y, int_x=True)
    body = f.axes(title, xlabel, ylabel)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(series))]
    for (name, values), c in zip(series.items(), colors):
        pts = " ".join(f"{f.x(x_start + i):.1f},{f.y(v):.1f}" for i, v in enumerate(values))
        body.append(f"<polyline points='{pts}' fill='none' stroke='{c}' stroke-width='2' stroke-linejoin='round'/>")
    # put the legend in whichever right-hand corner the curves' last third avoids
    tail = [f.y(v) for vals in series.values() for v in list(vals)[-max(1, len(vals) // 3):]]
    legend_h = len(series) * 18 + 8
    near_top = any(yy < f.top + legend_h + 12 for yy in tail)
    near_bottom = any(yy > f.top + f.ph - legend_h - 12 for yy in tail)
    ly = f.top + f.ph - legend_h - 8 if near_top and not near_bottom else f.top + 8
    body += _legend(list(series), colors, f.left + f.pw - 8, ly)
    _write(path, _svg(width, height, body))


def decision_boundary(model, X: np.ndarray, y: np.ndarray, path: str, loss: Loss | None = None,
                      title: str = "", resolution: int = 300, size: int = 480, padding: float = 0.25) -> float:
    """Shade the 2-D input plane by predicted class (opacity = confidence) and overlay the data.

    Returns the accuracy of the model on ``(X, y)``.
    """
    (x0, y0), (x1, y1) = X.min(axis=0) - padding, X.max(axis=0) + padding
    gx, gy = np.meshgrid(np.linspace(x0, x1, resolution), np.linspace(y1, y0, resolution))
    grid = np.c_[gx.ravel(), gy.ravel()]
    out = model.forward(grid)
    if out.shape[1] == 1:
        p1 = probabilities(loss, out)[:, 0] if loss is not None else (out[:, 0] > 0).astype(float)
        probs = np.c_[1 - p1, p1]
    else:
        probs = probabilities(loss, out) if loss is not None else np.eye(out.shape[1])[out.argmax(1)]
    k = probs.shape[1]
    cols = np.stack([_hex_to_rgb(PALETTE[i % len(PALETTE)]) for i in range(k)])
    pred = probs.argmax(1)
    conf = (probs.max(1) - 1.0 / k) / (1 - 1.0 / k)  # 0 at chance, 1 when certain
    alpha = (0.18 + 0.32 * conf)[:, None]
    rgb = (255 * (1 - alpha) + cols[pred] * alpha).reshape(resolution, resolution, 3)
    img = base64.b64encode(png_bytes(rgb.astype(np.uint8))).decode()

    m = 40
    f = _Frame(size + 2 * m, size + m + 48, (x0, x1), (y0, y1), margin=(48, m, m, m))
    body = [f"<rect width='{f.w}' height='{f.h}' fill='white'/>",
            f"<image x='{f.left}' y='{f.top}' width='{f.pw}' height='{f.ph}' preserveAspectRatio='none' "
            f"xlink:href='data:image/png;base64,{img}' href='data:image/png;base64,{img}'/>"]
    r = 3.6 if len(X) > 50 else 9.0
    for xi, yi in zip(X, np.asarray(y).ravel().astype(int)):
        body.append(f"<circle cx='{f.x(xi[0]):.1f}' cy='{f.y(xi[1]):.1f}' r='{r}' fill='{PALETTE[yi % len(PALETTE)]}' stroke='white' stroke-width='{r / 4:.1f}'/>")
    body.append(f"<rect x='{f.left}' y='{f.top}' width='{f.pw}' height='{f.ph}' fill='none' stroke='#999'/>")
    acc = float(np.mean(model.predict(X) == np.asarray(y).ravel()))
    label = f"{title}  -  accuracy {acc:.1%}" if title else f"accuracy {acc:.1%}"
    body.append(f"<text x='{f.w / 2}' y='28' text-anchor='middle' font-size='15' font-weight='600' fill='#222' {FONT}>{escape(label)}</text>")
    _write(path, _svg(f.w, f.h, body))
    return acc


def regression_plot(X: np.ndarray, y: np.ndarray, model, path: str, title: str = "",
                    true_fn=None, width: int = 640, height: int = 400) -> None:
    """Scatter of 1-D data with the model's prediction (and optional ground truth) as lines."""
    xs = np.linspace(X.min(), X.max(), 300)[:, None]
    pred = model.forward(xs)[:, 0]
    lo = min(y.min(), pred.min()); hi = max(y.max(), pred.max()); pad = 0.08 * (hi - lo)
    f = _Frame(width, height, (float(X.min()), float(X.max())), (lo - pad, hi + pad))
    body = f.axes(title, "x", "y")
    for xi, yi in zip(X[:, 0], np.ravel(y)):
        body.append(f"<circle cx='{f.x(xi):.1f}' cy='{f.y(yi):.1f}' r='3' fill='{PALETTE[0]}' fill-opacity='0.55'/>")
    names, colors = ["network prediction"], [PALETTE[3]]
    if true_fn is not None:
        pts = " ".join(f"{f.x(a):.1f},{f.y(b):.1f}" for a, b in zip(xs[:, 0], true_fn(xs[:, 0])))
        body.append(f"<polyline points='{pts}' fill='none' stroke='#333' stroke-width='1.5' stroke-dasharray='5 4'/>")
        names.append("ground truth"); colors.append("#333")
    pts = " ".join(f"{f.x(a):.1f},{f.y(b):.1f}" for a, b in zip(xs[:, 0], pred))
    body.append(f"<polyline points='{pts}' fill='none' stroke='{PALETTE[3]}' stroke-width='2.5'/>")
    body += _legend(names, colors, f.left + f.pw - 8, f.top + 8)
    _write(path, _svg(width, height, body))


def confusion_matrix_plot(cm: np.ndarray, path: str, class_names: Sequence[str] | None = None,
                          title: str = "Confusion matrix", cell: int = 40) -> None:
    k = cm.shape[0]
    names = list(class_names) if class_names else [str(i) for i in range(k)]
    left, top = 70, 60
    w, h = left + k * cell + 20, top + k * cell + 50
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    base = _hex_to_rgb(PALETTE[0])
    body = [f"<rect width='{w}' height='{h}' fill='white'/>",
            f"<text x='{w / 2}' y='26' text-anchor='middle' font-size='15' font-weight='600' fill='#222' {FONT}>{escape(title)}</text>"]
    for i in range(k):
        for j in range(k):
            a = norm[i, j]
            c = (255 * (1 - a) + base * a).astype(int)
            body.append(f"<rect x='{left + j * cell}' y='{top + i * cell}' width='{cell}' height='{cell}' fill='rgb({c[0]},{c[1]},{c[2]})' stroke='white'/>")
            body.append(f"<text x='{left + j * cell + cell / 2}' y='{top + i * cell + cell / 2 + 4}' text-anchor='middle' font-size='{10 if cm.max() >= 1000 else 11}' fill='{'white' if a > 0.5 else '#333'}' {FONT}>{cm[i, j]}</text>")
        body.append(f"<text x='{left - 8}' y='{top + i * cell + cell / 2 + 4}' text-anchor='end' font-size='11' fill='#333' {FONT}>{escape(names[i])}</text>")
        body.append(f"<text x='{left + i * cell + cell / 2}' y='{top - 8}' text-anchor='middle' font-size='11' fill='#333' {FONT}>{escape(names[i])}</text>")
    body.append(f"<text x='{left + k * cell / 2}' y='{h - 14}' text-anchor='middle' font-size='12' fill='#333' {FONT}>predicted (columns) vs true (rows)</text>")
    _write(path, _svg(w, h, body))


def image_grid(images: np.ndarray, path: str, ncols: int = 10, scale: int = 3, diverging: bool = False) -> None:
    """Save a grid of images (N, H, W) as a PNG on a white background.

    diverging=False : grayscale, high values drawn dark (e.g. digits: dark ink on white).
    diverging=True  : signed values (e.g. learned weights) - red positive, blue negative,
                      white zero, each image scaled by its own max |value|.
    """
    n, h, w = images.shape
    nrows = math.ceil(n / ncols)
    gap = 2
    canvas = np.full((nrows * (h + gap) + gap, ncols * (w + gap) + gap, 3), 255.0)
    pos, neg = _hex_to_rgb(PALETTE[3]), _hex_to_rgb(PALETTE[0])
    for i, im in enumerate(images):
        r, c = divmod(i, ncols)
        if diverging:
            v = (im / (np.abs(im).max() + 1e-12))[:, :, None]
            tile = np.where(v > 0, 255 * (1 - v) + pos * v, 255 * (1 + v) - neg * v)
        else:
            lo, hi = im.min(), im.max()
            tile = np.repeat((255 * (1 - (im - lo) / (hi - lo + 1e-12)))[:, :, None], 3, axis=2)
        y0, x0 = gap + r * (h + gap), gap + c * (w + gap)
        canvas[y0:y0 + h, x0:x0 + w] = tile
    canvas = canvas.repeat(scale, axis=0).repeat(scale, axis=1)
    save_png(path, np.clip(canvas, 0, 255).astype(np.uint8))
