// Linear classifier probes (Alain & Bengio 2016): a softmax regression trained on one layer's
// frozen activations to predict the label. Held-out accuracy says how much label information a
// straight line can read off that layer. It does not say the model uses it, which is why every
// probe here comes with controls and a steering test.
// Plain JS with no imports, shared by the Tour, the Lab's worker and the node tests.

export function rng(seed = 1) {
  let r = seed >>> 0 || 1;
  return () => { r = (r * 16807) % 2147483647; return r / 2147483647; };
}

function shuffle(a, rnd) {
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

// A stratified train/test split, so every class keeps the same share in both halves.
export function splitOf(labels, testFrac = 0.3, seed = 11) {
  const rnd = rng(seed), by = new Map();
  labels.forEach((y, i) => { if (!by.has(y)) by.set(y, []); by.get(y).push(i); });
  const train = [], test = [];
  for (const idx of by.values()) {
    shuffle(idx, rnd);
    const nt = Math.round(idx.length * testFrac);
    test.push(...idx.slice(0, nt)); train.push(...idx.slice(nt));
  }
  return { train: Int32Array.from(train), test: Int32Array.from(test) };
}

// The same labels in a random order: a control task with the same class frequencies but no
// relation to the input (in the spirit of Hewitt & Liang 2019).
export function shuffledLabels(labels, seed = 23) {
  return shuffle(Array.from(labels), rng(seed));
}

// X: Float32Array of N rows of D. Feature-wise mean and spread from the training rows only.
function standardizer(X, D, train) {
  const mu = new Float64Array(D), sd = new Float64Array(D);
  for (const i of train) for (let j = 0; j < D; j++) mu[j] += X[i * D + j];
  for (let j = 0; j < D; j++) mu[j] /= train.length;
  for (const i of train) for (let j = 0; j < D; j++) sd[j] += (X[i * D + j] - mu[j]) ** 2;
  for (let j = 0; j < D; j++) sd[j] = Math.sqrt(sd[j] / train.length) || 1;
  return { mu: Float32Array.from(mu), sd: Float32Array.from(sd) };
}

function logitsOf(p, X, i, z, out) {
  const { D, K, W, b, mu, sd } = p;
  for (let j = 0; j < D; j++) z[j] = (X[i * D + j] - mu[j]) / sd[j];
  for (let k = 0; k < K; k++) {
    let s = b[k];
    const w = k * D;
    for (let j = 0; j < D; j++) s += W[w + j] * z[j];
    out[k] = s;
  }
  return out;
}

// Softmax regression with Adam on minibatches and a small L2 penalty.
export function trainProbe(X, D, y, K, train, { epochs = 60, lr = 0.02, l2 = 1e-3, batch = 64, seed = 3 } = {}) {
  const { mu, sd } = standardizer(X, D, train);
  const p = { D, K, W: new Float32Array(K * D), b: new Float32Array(K), mu, sd };
  const params = [p.W, p.b];
  const G = params.map((a) => new Float32Array(a.length));
  const M = params.map((a) => new Float32Array(a.length));
  const V = params.map((a) => new Float32Array(a.length));
  const z = new Float32Array(D), lg = new Float32Array(K);
  const order = Array.from(train), rnd = rng(seed);
  let t = 0;
  for (let ep = 0; ep < epochs; ep++) {
    shuffle(order, rnd);
    for (let s = 0; s < order.length; s += batch) {
      const rows = order.slice(s, s + batch);
      G[0].fill(0); G[1].fill(0);
      for (const i of rows) {
        logitsOf(p, X, i, z, lg);
        let m = -Infinity;
        for (let k = 0; k < K; k++) m = Math.max(m, lg[k]);
        let Z = 0;
        for (let k = 0; k < K; k++) { lg[k] = Math.exp(lg[k] - m); Z += lg[k]; }
        for (let k = 0; k < K; k++) {
          const g = (lg[k] / Z - (k === y[i] ? 1 : 0)) / rows.length;
          if (!g) continue;
          G[1][k] += g;
          const w = k * D;
          for (let j = 0; j < D; j++) G[0][w + j] += g * z[j];
        }
      }
      for (let j = 0; j < p.W.length; j++) G[0][j] += l2 * p.W[j];
      t++;
      const c1 = 1 - 0.9 ** t, c2 = 1 - 0.999 ** t;
      params.forEach((a, q) => {
        for (let j = 0; j < a.length; j++) {
          M[q][j] = 0.9 * M[q][j] + 0.1 * G[q][j];
          V[q][j] = 0.999 * V[q][j] + 0.001 * G[q][j] * G[q][j];
          a[j] -= (lr * M[q][j]) / c1 / (Math.sqrt(V[q][j] / c2) + 1e-8);
        }
      });
    }
  }
  return p;
}

export function predictProbe(p, X, i) {
  const lg = logitsOf(p, X, i, new Float32Array(p.D), new Float32Array(p.K));
  let best = 0;
  for (let k = 1; k < p.K; k++) if (lg[k] > lg[best]) best = k;
  return best;
}

export function accuracyOf(p, X, y, idx) {
  let c = 0;
  for (const i of idx) if (predictProbe(p, X, i) === y[i]) c++;
  return idx.length ? c / idx.length : 0;
}

// Share of the test items in the most common training class: what a probe that ignores the input gets.
export function majorityOf(labels, split) {
  const n = {};
  for (const i of split.train) n[labels[i]] = (n[labels[i]] ?? 0) + 1;
  const top = Number(Object.entries(n).sort((a, b) => b[1] - a[1])[0][0]);
  let c = 0;
  for (const i of split.test) if (labels[i] === top) c++;
  return c / split.test.length;
}

// One layer: the real probe and the shuffled-label control, both on the same split.
// Bigger layers get fewer epochs so a layer of thousands of units still finishes in seconds.
export function probeLayer(X, D, labels, K, split, opts = {}) {
  const epochs = opts.epochs ?? (D > 2000 ? 25 : D > 500 ? 40 : 60);
  const real = trainProbe(X, D, labels, K, split.train, { ...opts, epochs });
  const sh = shuffledLabels(labels, opts.seed ?? 23);
  const ctrl = trainProbe(X, D, sh, K, split.train, { ...opts, epochs });
  return {
    D,
    train: accuracyOf(real, X, labels, split.train),
    test: accuracyOf(real, X, labels, split.test),
    ctrlTrain: accuracyOf(ctrl, X, sh, split.train),
    ctrlTest: accuracyOf(ctrl, X, sh, split.test),
    probe: real,
  };
}

// Mass-mean direction for class k: the class's mean activation minus the mean of every other
// item, from training items only. Marks & Tegmark (2023) and Li et al. (2023) find this steers
// more reliably than the probe's own weights, which also chase noise directions.
export function massMeanDirection(X, D, labels, k, train) {
  const a = new Float64Array(D), o = new Float64Array(D);
  let na = 0, no = 0;
  for (const i of train) {
    const t = labels[i] === k ? a : o;
    for (let j = 0; j < D; j++) t[j] += X[i * D + j];
    if (labels[i] === k) na++; else no++;
  }
  return Float32Array.from({ length: D }, (_, j) => a[j] / (na || 1) - o[j] / (no || 1));
}

// A random direction with the same length as v, for the steering control.
export function randomLike(v, seed = 41) {
  const rnd = rng(seed);
  const r = Float32Array.from(v, () => { const u = rnd() || 1e-9; return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd()); });
  const n = (a) => Math.sqrt(a.reduce((s, x) => s + x * x, 0)) || 1;
  const f = n(v) / n(r);
  return r.map((x) => x * f);
}

// A copy of a weight set with each array's values shuffled: the same weight statistics with the
// learned structure destroyed. A layer of a random network can still be easy to probe.
export function scrambleArrays(arrays, seed = 101) {
  const rnd = rng(seed);
  return arrays.map((a) => shuffle(Float32Array.from(a), rnd));
}
