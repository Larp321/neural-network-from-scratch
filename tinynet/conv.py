"""Convolutional layers: Conv2D, MaxPool2D and Reshape/Flatten.

Images use the (N, C, H, W) layout. Convolution is computed with *im2col*:
every k x k patch of the input is unrolled into a row, so the whole layer
becomes one matrix multiplication (fast in NumPy). The backward pass is the
same idea in reverse (*col2im*): patch gradients are added back to the pixels
they came from.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .layers import Layer


class Conv2D(Layer):
    """2-D convolution: ``out[n, o] = sum_c W[o, c] * x[n, c] + b[o]`` over k x k windows.

    Parameters
    ----------
    in_channels, out_channels, kernel_size : layer shape.
    padding : zero-padding on each side ("same" output size for k=3 is padding=1).
    stride : step between windows.
    l2 : L2 weight decay.
    dtype : np.float32 trains about 2x faster; use float64 for gradient checks.
    input_grad : set False on the first layer of a network - nothing needs the
        gradient w.r.t. the input image, so it is not computed.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1,
                 stride: int = 1, l2: float = 0.0, rng: np.random.Generator | None = None,
                 dtype=np.float64, input_grad: bool = True) -> None:
        super().__init__()
        rng = rng if rng is not None else np.random.default_rng()
        k = kernel_size
        fan_in = in_channels * k * k
        self.c_in, self.c_out, self.k, self.pad, self.stride, self.l2 = in_channels, out_channels, k, padding, stride, l2
        self.input_grad = input_grad
        self.params["W"] = rng.normal(0.0, np.sqrt(2.0 / fan_in), (out_channels, in_channels, k, k)).astype(dtype)
        self.params["b"] = np.zeros(out_channels, dtype=dtype)
        self.grads = {name: np.zeros_like(p) for name, p in self.params.items()}

    def forward(self, x, training=False):
        n, c, h, w = x.shape
        k, s, p = self.k, self.stride, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p))) if p else x
        ho, wo = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
        win = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]  # (n, c, ho, wo, k, k)
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * k * k)  # im2col (copies)
        out = cols @ self.params["W"].reshape(self.c_out, -1).T + self.params["b"]
        self._cache = (cols, x.shape, xp.shape, ho, wo)
        return out.reshape(n, ho, wo, self.c_out).transpose(0, 3, 1, 2)

    def backward(self, grad):
        cols, x_shape, xp_shape, ho, wo = self._cache
        n, c, h, w = x_shape
        k, s, p = self.k, self.stride, self.pad
        W = self.params["W"]
        g = grad.transpose(0, 2, 3, 1).reshape(-1, self.c_out)          # (n*ho*wo, c_out)
        self.grads["W"] = (g.T @ cols).reshape(W.shape) + self.l2 * W
        self.grads["b"] = g.sum(axis=0)
        if not self.input_grad:
            return None
        if s == 1 and p <= k - 1:
            # dL/dx is a "full" convolution of the output gradient with the flipped kernel:
            # pad the gradient by k-1-p and reuse im2col -> one matrix multiplication.
            q = k - 1 - p
            gp = np.pad(grad, ((0, 0), (0, 0), (q, q), (q, q))) if q else grad
            gcols = sliding_window_view(gp, (k, k), axis=(2, 3)).transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, -1)
            Wf = W[:, :, ::-1, ::-1].transpose(1, 0, 2, 3).reshape(c, -1)   # (c_in, c_out*k*k)
            return (gcols @ Wf.T).reshape(n, h, w, c).transpose(0, 3, 1, 2)
        dcols = (g @ W.reshape(self.c_out, -1)).reshape(n, ho, wo, c, k, k)
        dcols = np.ascontiguousarray(dcols.transpose(4, 5, 0, 3, 1, 2))   # (k, k, n, c, ho, wo)
        dxp = np.zeros(xp_shape, dtype=grad.dtype)
        for i in range(k):                                               # col2im (general stride)
            for j in range(k):
                dxp[:, :, i:i + s * ho:s, j:j + s * wo:s] += dcols[i, j]
        return dxp[:, :, p:p + h, p:p + w] if p else dxp

    def regularization_loss(self):
        return 0.0 if self.l2 == 0.0 else 0.5 * self.l2 * float(np.sum(self.params["W"] ** 2))

    def __repr__(self):
        return f"Conv2D({self.c_in} -> {self.c_out}, {self.k}x{self.k}, pad={self.pad})"


class MaxPool2D(Layer):
    """Non-overlapping max pooling (``size`` x ``size``); H and W must be divisible by ``size``."""

    def __init__(self, size: int = 2) -> None:
        super().__init__()
        self.size = size

    def forward(self, x, training=False):
        n, c, h, w = x.shape
        s = self.size
        if h % s or w % s:
            raise ValueError(f"MaxPool2D({s}) needs H and W divisible by {s}, got {h}x{w}")
        blocks = x.reshape(n, c, h // s, s, w // s, s).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // s, w // s, s * s)
        self._idx = blocks.argmax(axis=-1)
        self._shape = x.shape
        return np.take_along_axis(blocks, self._idx[..., None], axis=-1)[..., 0]

    def backward(self, grad):
        n, c, h, w = self._shape
        s = self.size
        blocks = np.zeros((n, c, h // s, w // s, s * s), dtype=grad.dtype)
        np.put_along_axis(blocks, self._idx[..., None], grad[..., None], axis=-1)  # gradient goes to the max
        return blocks.reshape(n, c, h // s, w // s, s, s).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)

    def __repr__(self):
        return f"MaxPool2D({self.size})"


class Reshape(Layer):
    """Reshape each sample, e.g. ``Reshape((1, 28, 28))`` turns 784-vectors into images."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.shape = tuple(shape)

    def forward(self, x, training=False):
        self._in = x.shape
        return x.reshape((x.shape[0],) + self.shape)

    def backward(self, grad):
        return grad.reshape(self._in)

    def __repr__(self):
        return f"Reshape({self.shape})"


class Flatten(Reshape):
    """(N, C, H, W) -> (N, C*H*W), to connect convolutional layers to Dense layers."""

    def __init__(self) -> None:
        super().__init__((-1,))

    def __repr__(self):
        return "Flatten()"
