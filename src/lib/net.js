// Forward and backward pass for the tiny CNN, in plain JS, no dependencies.
// Layout is channels first, flattened: index = c*H*W + y*W + x.
// Shared by the browser and scripts/check_parity.mjs.

export const LAYERS = [
  { key: 'input', label: 'Input', shape: [1, 8, 8] },
  { key: 'conv1', label: 'Conv 1', shape: [6, 8, 8] },
  { key: 'conv2', label: 'Conv 2', shape: [12, 4, 4] },
  { key: 'dense', label: 'Dense', shape: [32] },
  { key: 'out', label: 'Output', shape: [10] },
];

export function layerSize(key) {
  const L = LAYERS.find((l) => l.key === key);
  return L.shape.reduce((a, b) => a * b, 1);
}

function flat(a) {
  return Float32Array.from(a.flat(Infinity));
}

export function loadNet(json) {
  return {
    c1w: flat(json.conv1.w), c1b: flat(json.conv1.b), // [6][1][3][3]
    c2w: flat(json.conv2.w), c2b: flat(json.conv2.b), // [12][6][3][3]
    dw: flat(json.dense.w), db: flat(json.dense.b), // [32][48]
    ow: flat(json.out.w), ob: flat(json.out.b), // [10][32]
    meta: json,
  };
}

function conv(inp, C, H, W, w, b, O) {
  const out = new Float32Array(O * H * W);
  for (let o = 0; o < O; o++)
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let s = b[o];
        for (let c = 0; c < C; c++)
          for (let ky = 0; ky < 3; ky++) {
            const iy = y + ky - 1;
            if (iy < 0 || iy >= H) continue;
            for (let kx = 0; kx < 3; kx++) {
              const ix = x + kx - 1;
              if (ix < 0 || ix >= W) continue;
              s += w[((o * C + c) * 3 + ky) * 3 + kx] * inp[(c * H + iy) * W + ix];
            }
          }
        out[(o * H + y) * W + x] = s;
      }
  return out;
}

function convBack(g, inp, C, H, W, w, O) {
  const gi = new Float32Array(C * H * W);
  for (let o = 0; o < O; o++)
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const go = g[(o * H + y) * W + x];
        if (go === 0) continue;
        for (let c = 0; c < C; c++)
          for (let ky = 0; ky < 3; ky++) {
            const iy = y + ky - 1;
            if (iy < 0 || iy >= H) continue;
            for (let kx = 0; kx < 3; kx++) {
              const ix = x + kx - 1;
              if (ix < 0 || ix >= W) continue;
              gi[(c * H + iy) * W + ix] += go * w[((o * C + c) * 3 + ky) * 3 + kx];
            }
          }
      }
  return gi;
}

const relu = (a) => a.map((v) => (v > 0 ? v : 0));

function pool(inp, C, H, W) {
  const h = H / 2, w = W / 2;
  const out = new Float32Array(C * h * w);
  const arg = new Int32Array(C * h * w);
  for (let c = 0; c < C; c++)
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        let best = -Infinity, bi = 0;
        for (let dy = 0; dy < 2; dy++)
          for (let dx = 0; dx < 2; dx++) {
            const i = (c * H + 2 * y + dy) * W + 2 * x + dx;
            if (inp[i] > best) { best = inp[i]; bi = i; }
          }
        out[(c * h + y) * w + x] = best;
        arg[(c * h + y) * w + x] = bi;
      }
  return { out, arg };
}

function dense(inp, w, b, O) {
  const I = inp.length;
  const out = new Float32Array(O);
  for (let o = 0; o < O; o++) {
    let s = b[o];
    for (let i = 0; i < I; i++) s += w[o * I + i] * inp[i];
    out[o] = s;
  }
  return out;
}

// x: Float32Array(64), values 0..1
export function forward(net, x) {
  const input = Float32Array.from(x);
  const c1pre = conv(input, 1, 8, 8, net.c1w, net.c1b, 6);
  const c1 = relu(c1pre);
  const p1 = pool(c1, 6, 8, 8);
  const c2pre = conv(p1.out, 6, 4, 4, net.c2w, net.c2b, 12);
  const c2 = relu(c2pre);
  const p2 = pool(c2, 12, 4, 4);
  const dpre = dense(p2.out, net.dw, net.db, 32);
  const d = relu(dpre);
  const logits = dense(d, net.ow, net.ob, 10);
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp(v - m));
  const z = e.reduce((a, b) => a + b, 0);
  const probs = e.map((v) => v / z);
  let pred = 0;
  for (let i = 1; i < 10; i++) if (logits[i] > logits[pred]) pred = i;
  return {
    input, c1pre, c1, p1, c2pre, c2, p2, dpre, d, logits, probs, pred,
    // what the 3D view renders: post-ReLU activations, logits shown as probabilities
    acts: { input, conv1: c1, conv2: c2, dense: d, out: probs },
    pre: { input, conv1: c1pre, conv2: c2pre, dense: dpre, out: logits },
  };
}

// Gradient of one neuron's pre-activation (or logit) with respect to the input.
// Max-pool routes the gradient to the winning cell; ReLU passes it where the unit was on.
export function inputGradient(net, f, layer, index) {
  let gOut = null, gD = null, gC2 = null, gC1 = null;
  if (layer === 'out') {
    gOut = new Float32Array(10);
    gOut[index] = 1;
  }
  if (gOut) {
    gD = new Float32Array(32);
    for (let o = 0; o < 10; o++) for (let i = 0; i < 32; i++) gD[i] += gOut[o] * net.ow[o * 32 + i];
    for (let i = 0; i < 32; i++) if (f.dpre[i] <= 0) gD[i] = 0;
  } else if (layer === 'dense') {
    gD = new Float32Array(32);
    gD[index] = 1;
  }
  if (gD) {
    const gP2 = new Float32Array(48);
    for (let o = 0; o < 32; o++) if (gD[o]) for (let i = 0; i < 48; i++) gP2[i] += gD[o] * net.dw[o * 48 + i];
    gC2 = new Float32Array(12 * 16);
    for (let i = 0; i < 48; i++) gC2[f.p2.arg[i]] += gP2[i];
    for (let i = 0; i < gC2.length; i++) if (f.c2pre[i] <= 0) gC2[i] = 0;
  } else if (layer === 'conv2') {
    gC2 = new Float32Array(12 * 16);
    gC2[index] = 1;
  }
  if (gC2) {
    const gP1 = convBack(gC2, f.p1.out, 6, 4, 4, net.c2w, 12);
    gC1 = new Float32Array(6 * 64);
    for (let i = 0; i < 96; i++) gC1[f.p1.arg[i]] += gP1[i];
    for (let i = 0; i < gC1.length; i++) if (f.c1pre[i] <= 0) gC1[i] = 0;
  } else if (layer === 'conv1') {
    gC1 = new Float32Array(6 * 64);
    gC1[index] = 1;
  }
  if (gC1) return convBack(gC1, f.input, 1, 8, 8, net.c1w, 6);
  return new Float32Array(64); // input layer: no gradient
}

// Receptive field on the 8x8 input: boolean mask of the pixels that can affect a neuron.
export function receptiveField(layer, index) {
  const m = new Uint8Array(64);
  const mark = (y0, y1, x0, x1) => {
    for (let y = Math.max(0, y0); y <= Math.min(7, y1); y++)
      for (let x = Math.max(0, x0); x <= Math.min(7, x1); x++) m[y * 8 + x] = 1;
  };
  if (layer === 'input') m[index] = 1;
  else if (layer === 'conv1') {
    const p = index % 64, y = Math.floor(p / 8), x = p % 8;
    mark(y - 1, y + 1, x - 1, x + 1);
  } else if (layer === 'conv2') {
    // conv2 cell (y,x) sees pooled cells y-1..y+1, each covering conv1 rows 2y'..2y'+1, each seeing +-1 input px
    const p = index % 16, y = Math.floor(p / 4), x = p % 4;
    mark(2 * (y - 1) - 1, 2 * (y + 1) + 2, 2 * (x - 1) - 1, 2 * (x + 1) + 2);
  } else m.fill(1);
  return m;
}

// Incoming connections of one neuron: [{layer, index, w}] where w is the weight
// (for conv layers, the kernel entry linking that source unit to this one).
export function incoming(net, layer, index) {
  const list = [];
  if (layer === 'conv1') {
    const o = Math.floor(index / 64), p = index % 64, y = Math.floor(p / 8), x = p % 8;
    for (let ky = 0; ky < 3; ky++)
      for (let kx = 0; kx < 3; kx++) {
        const iy = y + ky - 1, ix = x + kx - 1;
        if (iy < 0 || iy > 7 || ix < 0 || ix > 7) continue;
        list.push({ layer: 'input', index: iy * 8 + ix, w: net.c1w[o * 9 + ky * 3 + kx] });
      }
  } else if (layer === 'conv2') {
    // source is the pooled conv1 map; we draw the edge to the conv1 cell that won the pool
    const o = Math.floor(index / 16), p = index % 16, y = Math.floor(p / 4), x = p % 4;
    for (let c = 0; c < 6; c++)
      for (let ky = 0; ky < 3; ky++)
        for (let kx = 0; kx < 3; kx++) {
          const iy = y + ky - 1, ix = x + kx - 1;
          if (iy < 0 || iy > 3 || ix < 0 || ix > 3) continue;
          list.push({ layer: 'conv1', pooled: c * 16 + iy * 4 + ix, w: net.c2w[((o * 6 + c) * 3 + ky) * 3 + kx] });
        }
  } else if (layer === 'dense') {
    for (let i = 0; i < 48; i++) list.push({ layer: 'conv2', pooled: i, w: net.dw[index * 48 + i] });
  } else if (layer === 'out') {
    for (let i = 0; i < 32; i++) list.push({ layer: 'dense', index: i, w: net.ow[index * 32 + i] });
  }
  return list;
}

// Resolve a pooled source to the concrete pre-pool neuron that won max-pool for this input.
export function resolveSource(f, src) {
  if (src.index !== undefined) return src.index;
  if (src.layer === 'conv1') return f.p1.arg[src.pooled];
  if (src.layer === 'conv2') return f.p2.arg[src.pooled];
  return 0;
}

// The path of strongest contribution, traced backward from the predicted class.
// At each step we pick the incoming unit with the largest weight * activation.
export function strongestPath(net, f) {
  const path = [{ layer: 'out', index: f.pred }];
  let cur = path[0];
  while (cur.layer !== 'input') {
    const inc = incoming(net, cur.layer, cur.index);
    let best = null, bv = -Infinity;
    for (const s of inc) {
      const idx = resolveSource(f, s);
      const v = s.w * f.acts[s.layer][idx];
      if (v > bv) { bv = v; best = { layer: s.layer, index: idx }; }
    }
    path.unshift(best);
    cur = best;
  }
  return path;
}

// Gradient ascent on the input to maximise one neuron's pre-activation (Erhan et al. 2009; Olah et al. 2017).
// Returns a generator so the UI can animate each step.
export function* activationMax(net, layer, index, { steps = 120, lr = 0.08, l2 = 0.02, seed = 1 } = {}) {
  let s = seed;
  const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const x = new Float32Array(64);
  for (let i = 0; i < 64; i++) x[i] = 0.05 + 0.05 * rnd(); // near-blank start
  for (let t = 0; t < steps; t++) {
    const f = forward(net, x);
    const g = inputGradient(net, f, layer, index);
    for (let i = 0; i < 64; i++) x[i] = Math.min(1, Math.max(0, x[i] + lr * (g[i] - l2 * x[i])));
    yield { x: Float32Array.from(x), value: f.pre[layer][index], step: t + 1 };
  }
}

export function runActivationMax(net, layer, index, opts) {
  let last = null;
  for (const st of activationMax(net, layer, index, opts)) last = st;
  return last;
}
