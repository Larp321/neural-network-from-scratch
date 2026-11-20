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

  // ---------------------------------------------------------------- drawing
  const board = $("board");
  const ctx = board.getContext("2d", { willReadFrequently: true });
  const SIZE = board.width; // 280
  let drawing = false, last = null, hasInk = false, pending = false;

  function resetBoard() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, SIZE, SIZE);
    hasInk = false;
    $("boardEmpty").classList.remove("hidden");
  }

  function pos(e) {
    const r = board.getBoundingClientRect();
    return { x: ((e.clientX - r.left) / r.width) * SIZE, y: ((e.clientY - r.top) / r.height) * SIZE };
  }

  function stroke(a, b) {
    ctx.strokeStyle = "#fff";
    ctx.fillStyle = "#fff";
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
   */
  function preprocess() {
    const data = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const g = new Float32Array(SIZE * SIZE);
    let x0 = SIZE, y0 = SIZE, x1 = -1, y1 = -1;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = data[(y * SIZE + x) * 4] / 255;
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
    // paste centred, then shift by centre of mass
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
  const bars = [];
  for (let d = 0; d < 10; d++) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="d">${d}</span><span class="track"><span class="fill"></span></span><span class="p">–</span>`;
    $("bars").appendChild(li);
    bars.push({ li, fill: li.querySelector(".fill"), p: li.querySelector(".p") });
  }

  function paint(canvas, values, color) {
    const c = canvas.getContext("2d");
    const { width: w, height: h } = canvas;
    const max = Math.max(1e-6, ...values);
    const im = c.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const a = Math.max(0, Math.min(1, (values[i] || 0) / max));
      im.data[i * 4] = color[0]; im.data[i * 4 + 1] = color[1]; im.data[i * 4 + 2] = color[2];
      im.data[i * 4 + 3] = Math.round(255 * a);
    }
    c.clearRect(0, 0, w, h);
    c.putImageData(im, 0, 0);
  }

  const WHITE = [255, 255, 255], BLUE = [122, 162, 255], GOLD = [255, 196, 87];

  function show(img) {
    const [h1, h2, logits] = forward(img);
    const probs = softmax(logits);
    const order = probs.map((p, d) => [p, d]).sort((a, b) => b[0] - a[0]);
    const [[p1, d1], [p2, d2]] = order;
    $("bigDigit").textContent = d1;
    $("confidence").textContent = p1 >= 0.5 ? `${pct(p1)} sure it's a ${d1}` : `Not sure… maybe a ${d1}?`;
    $("runner").textContent = `Second guess: ${d2} (${pct(p2)})`;
    bars.forEach((b, d) => {
      b.fill.style.width = `${(probs[d] * 100).toFixed(1)}%`;
      b.p.textContent = pct(probs[d]);
      b.li.classList.toggle("top", d === d1);
    });
    paint($("seen"), img, WHITE);
    paint($("hidden1"), h1, BLUE);
    paint($("hidden2"), h2, BLUE);
    paint($("outputs"), probs, GOLD);
  }

  function pct(p) {
    return p >= 0.995 ? ">99%" : p < 0.005 ? "<1%" : `${Math.round(p * 100)}%`;
  }

  function clearResult() {
    $("bigDigit").textContent = "?";
    $("confidence").textContent = "Waiting for a drawing";
    $("runner").textContent = "";
    bars.forEach((b) => { b.fill.style.width = "0"; b.p.textContent = "–"; b.li.classList.remove("top"); });
    ["seen", "hidden1", "hidden2", "outputs"].forEach((id) => paint($(id), [], WHITE));
  }

  function predictFromBoard() {
    if (!hasInk) return clearResult();
    const r = preprocess();
    if (!r) return clearResult();
    $("hint").textContent = r.boxSize < 70 ? "Tip: draw bigger for better results" : "Draw one digit, 0–9, large and centered";
    show(r.img);
  }

  // ------------------------------------------------------------ test samples
  let showingSample = false;
  const samples = M.samples.map((s) => ({
    label: s.label,
    logits: s.logits,
    img: Float32Array.from(b64ToBytes(s.pixels), (v) => v / 255),
  }));

  function drawSmall(canvas, img) {
    paint(canvas, img, WHITE);
  }

  samples.forEach((s) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "sample";
    btn.setAttribute("aria-label", `Example: a handwritten ${s.label}`);
    const c = document.createElement("canvas");
    c.width = c.height = 28;
    drawSmall(c, s.img);
    btn.appendChild(c);
    btn.addEventListener("click", () => {
      resetBoard();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.drawImage(c, 0, 0, SIZE, SIZE);
      $("boardEmpty").classList.add("hidden");
      showingSample = true;
      hasInk = true;
      $("hint").textContent = `A real ${s.label} from the test set, never seen in training`;
      show(s.img); // feed the exact 28x28 test pixels
    });
    $("samples").appendChild(btn);
  });

  $("clear").addEventListener("click", () => {
    resetBoard();
    showingSample = false;
    $("hint").textContent = "Draw one digit, 0–9, large and centered";
    clearResult();
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
      ? `Verified: on 10 reference digits, this JavaScript gives the same outputs as the Python model (largest difference ${maxDiff.toExponential(1)}).`
      : `Warning: the browser results differ from the Python reference (largest difference ${maxDiff.toFixed(4)}).`;
    window.tinynetSelfTest = { ok, maxDiff };
  }

  resetBoard();
  clearResult();
  selfTest();
  window.tinynet = { forward, preprocess, softmax };
})();
