/* Draw-a-digit demo: runs the tinynet MNIST model (exported by export_model.py) in plain JavaScript. */
(() => {
  "use strict";
  const M = window.TINYNET_MODEL;
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ model
  function b64ToBytes(s) {
    const bin = atob(s);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  // Dequantize: W[i][j] = q[i][j] * scale[j]   (stored row-major, shape in x out)
  const layers = M.layers.map((L) => {
    const q = new Int8Array(b64ToBytes(L.w).buffer);
    const W = new Float32Array(L.in * L.out);
    for (let i = 0; i < L.in; i++) {
      for (let j = 0; j < L.out; j++) W[i * L.out + j] = q[i * L.out + j] * L.scale[j];
    }
    return { nIn: L.in, nOut: L.out, W, b: Float32Array.from(L.b) };
  });

  // y = x @ W + b, skipping zero inputs (most of a drawing is blank)
  function dense(x, L) {
    const y = Float32Array.from(L.b);
    for (let i = 0; i < L.nIn; i++) {
      const xi = x[i];
      if (xi === 0) continue;
      const row = i * L.nOut;
      for (let j = 0; j < L.nOut; j++) y[j] += xi * L.W[row + j];
    }
    return y;
  }

  /** Returns the activations of every layer: [hidden1, hidden2, logits]. */
  function forward(x) {
    const acts = [];
    let h = x;
    layers.forEach((L, k) => {
      h = dense(h, L);
      if (k < layers.length - 1) for (let j = 0; j < h.length; j++) if (h[j] < 0) h[j] = 0; // ReLU
      acts.push(h);
    });
    return acts;
  }

  function softmax(z) {
    const m = Math.max(...z);
    const e = Array.from(z, (v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  // ------------------------------------------------------------------ theme
  function cssColor(name) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    const m = v.match(/^#([0-9a-f]{6})$/i);
    if (!m) return [31, 58, 147];
    const n = parseInt(m[1], 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const rgb = (c) => `rgb(${c[0]},${c[1]},${c[2]})`;

  // ---------------------------------------------------------------- drawing
  // Ink is stored in the canvas alpha channel; the squared paper is the CSS background.
  const board = $("board");
  const ctx = board.getContext("2d", { willReadFrequently: true });
  const SIZE = board.width; // 280
  let drawing = false, last = null, hasInk = false, pending = false, showingSample = false;

  function resetBoard() {
    ctx.clearRect(0, 0, SIZE, SIZE);
    hasInk = false;
    $("boardEmpty").classList.remove("hidden");
  }

  function pos(e) {
    const r = board.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  }

  function stroke(a, b) {
    ctx.strokeStyle = rgb(cssColor("--pen"));
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  board.addEventListener("pointerdown", (e) => {
    e.preventDefault();
    if (showingSample) { resetBoard(); showingSample = false; }
    board.setPointerCapture(e.pointerId);
    drawing = true;
    last = pos(e);
    stroke(last, last);
    hasInk = true;
    $("boardEmpty").classList.add("hidden");
    schedule();
  });
  board.addEventListener("pointermove", (e) => {
    if (!drawing) return;
    const p = pos(e);
    stroke(last, p);
    last = p;
    schedule();
  });
  const end = () => { drawing = false; schedule(); };
  board.addEventListener("pointerup", end);
  board.addEventListener("pointercancel", end);

  function schedule() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; predictFromBoard(); });
  }

  // ------------------------------------------------------------ preprocessing
  /**
   * Turn the 280x280 drawing into a 28x28 MNIST-style image: crop to the ink,
   * scale the longer side to 20 px (area averaging), paste into a 28x28 field and
   * shift so the centre of mass sits in the middle - the same recipe used to build MNIST.
   * (tinynet/handwriting.py contains a line-by-line Python port used for training data.)
   */
  function preprocess() {
    const data = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const g = new Float32Array(SIZE * SIZE);
    let x0 = SIZE, y0 = SIZE, x1 = -1, y1 = -1;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = data[(y * SIZE + x) * 4 + 3] / 255; // alpha = ink
        g[y * SIZE + x] = v;
        if (v > 0.1) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
      }
    }
    if (x1 < 0) return null;
    const w = x1 - x0 + 1, h = y1 - y0 + 1;
    const s = 20 / Math.max(w, h);
    const tw = Math.max(1, Math.round(w * s)), th = Math.max(1, Math.round(h * s));
    const small = new Float32Array(tw * th);
    if (s < 1) {
      const sum = new Float32Array(tw * th), cnt = new Float32Array(tw * th);
      for (let y = y0; y <= y1; y++) {
        const ty = Math.min(th - 1, Math.floor((y - y0) * s));
        for (let x = x0; x <= x1; x++) {
          const k = ty * tw + Math.min(tw - 1, Math.floor((x - x0) * s));
          sum[k] += g[y * SIZE + x];
          cnt[k] += 1;
        }
      }
      for (let k = 0; k < small.length; k++) small[k] = cnt[k] ? sum[k] / cnt[k] : 0;
    } else {
      for (let ty = 0; ty < th; ty++)
        for (let tx = 0; tx < tw; tx++)
          small[ty * tw + tx] = g[(y0 + Math.min(h - 1, Math.floor(ty / s))) * SIZE + x0 + Math.min(w - 1, Math.floor(tx / s))];
    }
    const ox = Math.floor((28 - tw) / 2), oy = Math.floor((28 - th) / 2);
    let mass = 0, mx = 0, my = 0;
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
      const v = small[y * tw + x];
      mass += v; mx += v * (x + ox); my += v * (y + oy);
    }
    const dx = Math.round(14 - mx / mass), dy = Math.round(14 - my / mass);
    const img = new Float32Array(784);
    for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) {
      const X = x + ox + dx, Y = y + oy + dy;
      if (X >= 0 && X < 28 && Y >= 0 && Y < 28) img[Y * 28 + X] = small[y * tw + x];
    }
    return { img, boxSize: Math.max(w, h) };
  }

  // ---------------------------------------------------------------- rendering
  const cols = [];
  for (let d = 0; d < 10; d++) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="pct"></span><span class="bar"></span><span class="d">${d}</span>`;
    $("bars").appendChild(li);
    cols.push({ li, bar: li.querySelector(".bar"), pct: li.querySelector(".pct") });
  }

  /** Paint values as squares of colour whose opacity is the (max-normalised) value. */
  function paint(canvas, values, color, normalise = true) {
    const c = canvas.getContext("2d");
    const { width: w, height: h } = canvas;
    const max = normalise ? Math.max(1e-6, ...values) : 1;
    const im = c.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const a = Math.max(0, Math.min(1, (values[i] || 0) / max));
      im.data[i * 4] = color[0]; im.data[i * 4 + 1] = color[1]; im.data[i * 4 + 2] = color[2];
      im.data[i * 4 + 3] = Math.round(255 * a);
    }
    c.clearRect(0, 0, w, h);
    c.putImageData(im, 0, 0);
  }

  let lastAnswer = null;
  function markAnswer(digit) {
    $("bigDigit").textContent = digit;
    $("bigDigit").parentElement.classList.remove("idle");
    if (digit === lastAnswer) return;
    lastAnswer = digit;
    const ring = $("ring");
    ring.classList.remove("drawn");
    void ring.getBoundingClientRect(); // restart the red-pen stroke
    ring.classList.add("drawn");
  }

  function show(img) {
    const [h1, h2, logits] = forward(img);
    const probs = softmax(logits);
    const order = probs.map((p, d) => [p, d]).sort((a, b) => b[0] - a[0]);
    const [[p1, d1], [p2, d2]] = order;
    markAnswer(d1);
    $("confidence").textContent = p1 >= 0.5 ? `Looks like a ${d1}` : `Hard to read. Maybe a ${d1}?`;
    $("runner").textContent = `${pct(p1)} sure. Second guess: ${d2} (${pct(p2)})`;
    cols.forEach((c, d) => {
      c.bar.style.height = `${(probs[d] * 100).toFixed(1)}%`;
      c.pct.textContent = probs[d] >= 0.01 ? pct(probs[d]) : "";
      c.li.classList.toggle("top", d === d1);
    });
    const pen = cssColor("--pen");
    paint($("seen"), img, pen, false);
    paint($("hidden1"), h1, pen);
    paint($("hidden2"), h2, pen);
    paint($("outputs"), probs, cssColor("--red"));
  }

  function pct(p) {
    return p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`;
  }

  function clearResult() {
    lastAnswer = null;
    $("bigDigit").textContent = "?";
    $("bigDigit").parentElement.classList.add("idle");
    $("ring").classList.remove("drawn");
    $("confidence").textContent = "Start drawing on the squared paper.";
    $("runner").textContent = "";
    cols.forEach((c) => { c.bar.style.height = "0"; c.pct.textContent = ""; c.li.classList.remove("top"); });
    ["seen", "hidden1", "hidden2", "outputs"].forEach((id) => paint($(id), [], [0, 0, 0]));
  }

  const HINT = "Each square is one pixel of what the network sees.";
  function predictFromBoard() {
    if (!hasInk) return clearResult();
    const r = preprocess();
    if (!r) return clearResult();
    $("hint").textContent = r.boxSize < 70 ? "Try writing bigger, using most of the paper." : HINT;
    show(r.img);
  }

  // ------------------------------------------------------------ test samples
  const samples = M.samples.map((s) => ({
    label: s.label,
    logits: s.logits,
    img: Float32Array.from(b64ToBytes(s.pixels), (v) => v / 255),
  }));

  const sampleCanvases = [];
  samples.forEach((s) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample";
    btn.setAttribute("aria-label", `Show a handwritten ${s.label} from the test set`);
    const c = document.createElement("canvas");
    c.width = c.height = 28;
    sampleCanvases.push(c);
    btn.appendChild(c);
    btn.addEventListener("click", () => {
      resetBoard();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(c, 0, 0, SIZE, SIZE);
      $("boardEmpty").classList.add("hidden");
      showingSample = true;
      hasInk = true;
      $("hint").textContent = `A real ${s.label} from the test set. The network never saw it during training.`;
      show(s.img); // feed the exact 28x28 test pixels
    });
    $("samples").appendChild(btn);
  });
  function paintSamples() {
    const pen = cssColor("--pen");
    samples.forEach((s, i) => paint(sampleCanvases[i], s.img, pen, false));
  }

  $("clear").addEventListener("click", () => {
    resetBoard();
    showingSample = false;
    $("hint").textContent = HINT;
    clearResult();
  });

  // Re-colour ink when the system switches between light and dark.
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    paintSamples();
    if (hasInk && !showingSample) {
      const tmp = ctx.getImageData(0, 0, SIZE, SIZE), pen = cssColor("--pen");
      for (let i = 0; i < tmp.data.length; i += 4) { tmp.data[i] = pen[0]; tmp.data[i + 1] = pen[1]; tmp.data[i + 2] = pen[2]; }
      ctx.putImageData(tmp, 0, 0);
    }
    predictFromBoard();
  });

  // ------------------------------------------------- self-test vs Python output
  function selfTest() {
    let maxDiff = 0, allMatch = true;
    for (const s of samples) {
      const logits = forward(s.img)[2];
      logits.forEach((v, j) => { maxDiff = Math.max(maxDiff, Math.abs(v - s.logits[j])); });
      if (logits.indexOf(Math.max(...logits)) !== s.label) allMatch = false;
    }
    const ok = allMatch && maxDiff < 1e-2;
    const el = $("verify");
    el.classList.add(ok ? "ok" : "bad");
    el.textContent = ok
      ? `Checked on load: for 10 reference digits this page computes the same outputs as the Python model (largest difference ${maxDiff.toExponential(1)}).`
      : `These results differ from the Python model (largest difference ${maxDiff.toFixed(4)}). Re-export web/model.js.`;
    window.tinynetSelfTest = { ok, maxDiff };
  }

  $("accText").textContent = `${(M.test_accuracy * 100).toFixed(1)}%`;
  paintSamples();
  resetBoard();
  clearResult();
  selfTest();
  window.tinynet = { forward, preprocess, softmax };
})();
