# The maths behind tinynet

This page derives every gradient the library implements. Notation: a mini-batch
has `N` samples; `X` is `N × d_in`; upper-case `G` is the gradient of the loss
w.r.t. a layer's output (`∂L/∂Y`), which is what each `backward()` receives.

## 1. Dense layer

Forward: `Y = X W + b` with `W` of shape `d_in × d_out`.

Because `Y_ij = Σ_k X_ik W_kj + b_j`, the chain rule gives

| gradient | formula | code |
|---|---|---|
| `∂L/∂W` | `Xᵀ G` | `self._x.T @ grad` |
| `∂L/∂b` | `Σ_rows G` | `grad.sum(axis=0)` |
| `∂L/∂X` | `G Wᵀ` | `grad @ W.T` (passed to the previous layer) |

With L2 weight decay the loss gains `½ λ ‖W‖²`, adding `λ W` to `∂L/∂W`.

### Initialisation
* **He** (`std = √(2/d_in)`) keeps activation variance roughly constant through ReLU layers, because ReLU zeroes half its inputs.
* **Xavier/Glorot** (`std = √(2/(d_in+d_out))`) balances forward and backward variance for symmetric activations like tanh.

## 2. Activations (element-wise, so `∂L/∂X = G ⊙ f'(X)`)

| activation | f(x) | f'(x) |
|---|---|---|
| ReLU | `max(0, x)` | `1[x > 0]` |
| Leaky ReLU | `x` if `x>0` else `αx` | `1` or `α` |
| Sigmoid | `σ(x) = 1/(1+e^{-x})` | `σ(x)(1-σ(x))` |
| Tanh | `tanh x` | `1 - tanh² x` |

Sigmoid and tanh derivatives are computed from the cached *output*, so no
extra exponentials are needed in the backward pass. `sigmoid` is evaluated as
`1/(1+e^{-x})` for `x ≥ 0` and `e^x/(1+e^x)` for `x < 0` so `exp` never overflows.

## 3. Dropout

During training each unit is kept with probability `1-p` and scaled by
`1/(1-p)` ("inverted dropout"), so the expected activation is unchanged and
inference needs no rescaling. The backward pass multiplies by the same mask.

## 4. Losses

### Softmax + cross-entropy (fused)
With logits `z`, `p = softmax(z)`, one-hot targets `y`:

```
L = -(1/N) Σ_i Σ_c y_ic log p_ic
∂L/∂z = (p - y) / N
```

Computing `log p` as `z - max(z) - log Σ exp(z - max(z))` (log-sum-exp) keeps
it finite for any logits. Fusing softmax into the loss avoids the full softmax
Jacobian and is the reason the gradient is so simple. Label smoothing replaces
`y` by `(1-ε) y + ε/C`.

### Sigmoid + binary cross-entropy (fused)
```
L = (1/N) Σ [ max(z,0) - z·y + log(1 + e^{-|z|}) ]      (stable form)
∂L/∂z = (σ(z) - y) / N
```

### Mean squared error
`L = (1/2N) Σ ‖ŷ - y‖²`, so `∂L/∂ŷ = (ŷ - y)/N`.

## 5. Backpropagation

`Sequential.backward` simply calls each layer's `backward` in reverse order —
the chain rule applied layer by layer:

```
G_L = ∂loss/∂output
G_{k-1} = layer_k.backward(G_k)   # also stores ∂L/∂W_k, ∂L/∂b_k
```

## 6. Optimizers

For parameter `θ` with gradient `g` at step `t`:

| optimizer | update |
|---|---|
| SGD | `θ ← θ - η g` |
| Momentum | `v ← μv - ηg;  θ ← θ + v` |
| Nesterov | `v ← μv - ηg;  θ ← θ + μv - ηg` |
| RMSProp | `s ← ρs + (1-ρ)g²;  θ ← θ - η g / (√s + ε)` |
| Adam | `m ← β₁m + (1-β₁)g;  v ← β₂v + (1-β₂)g²;  m̂ = m/(1-β₁ᵗ);  v̂ = v/(1-β₂ᵗ);  θ ← θ - η m̂/(√v̂ + ε)` |

Optional global-norm gradient clipping rescales all gradients by
`clip / ‖g‖` when `‖g‖ > clip`.

## 7. Verifying it: gradient checking

`tinynet.gradcheck` perturbs every single parameter by `±ε` and estimates the
derivative with central differences,

```
∂L/∂θ ≈ (L(θ + ε) - L(θ - ε)) / 2ε        (error O(ε²))
```

then compares it with backprop using the relative error
`|a - b| / (|a| + |b|)`. The test-suite requires this to be below `1e-6` for
every layer, activation, loss and for the input gradient — which is strong
evidence that each `backward()` above is implemented correctly.
