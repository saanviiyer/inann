// Latent space tools: PCA of a layer's representation of the held-out digits, projection of
// new inputs, nearest neighbours, and a per-layer separability number.
import { ACTS, LABELS } from './model.js';

export const LATENT_LAYERS = [
  { key: 'input', label: 'Input pixels', dim: 64 },
  { key: 'conv1', label: 'Conv 1', dim: 384 },
  { key: 'conv2', label: 'Conv 2', dim: 192 },
  { key: 'dense', label: 'Dense', dim: 32 },
  { key: 'sae', label: 'SAE features', dim: 256 },
  { key: 'out', label: 'Output probabilities', dim: 10 },
];

// ---------- generic versions: X is an array of equal-length vectors, labels optional ----------

// Top-3 principal components by power iteration on X^T X, never forming the D x D matrix.
// Stops early once a component stops moving.
export function pcaOf(X, { k: K = 3, iters = 150 } = {}) {
  const N = X.length, D = X[0].length;
  const mean = new Float64Array(D);
  for (const x of X) for (let j = 0; j < D; j++) mean[j] += x[j] / N;
  const C = X.map((x) => Float64Array.from(x, (v, j) => v - mean[j]));
  let total = 0;
  for (const c of C) for (const v of c) total += v * v;
  const comps = [], vars = [];
  for (let k = 0; k < K; k++) {
    let v = Float64Array.from({ length: D }, (_, j) => Math.sin(1 + j * (k + 1.3)));
    let lam = 0;
    for (let it = 0; it < iters; it++) {
      const w = new Float64Array(D);
      for (const c of C) {
        let s = 0;
        for (let j = 0; j < D; j++) s += c[j] * v[j];
        for (let j = 0; j < D; j++) w[j] += s * c[j];
      }
      for (const u of comps) {
        let d = 0;
        for (let j = 0; j < D; j++) d += w[j] * u[j];
        for (let j = 0; j < D; j++) w[j] -= d * u[j];
      }
      let n = 0;
      for (let j = 0; j < D; j++) n += w[j] * w[j];
      n = Math.sqrt(n);
      if (!n) break;
      let delta = 0;
      for (let j = 0; j < D; j++) { const nv = w[j] / n; delta += Math.abs(nv - v[j]); v[j] = nv; }
      const done = Math.abs(n - lam) < 1e-7 * n || delta < 1e-6;
      lam = n;
      if (done) break;
    }
    comps.push(v);
    vars.push(lam);
  }
  const project = (x) => comps.map((u) => {
    let s = 0;
    for (let j = 0; j < D; j++) s += (x[j] - mean[j]) * u[j];
    return s;
  });
  const pts = X.map(project);
  const scale = Math.max(...pts.flat().map(Math.abs)) || 1;
  return { project, pts, scale, explained: vars.map((v) => v / (total || 1)), D };
}

const d2 = (a, b) => {
  let s = 0;
  for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2;
  return s;
};

export function neighboursOf(X, x, k = 5) {
  return X.map((a, i) => [d2(a, x), i]).sort((p, q) => p[0] - q[0]).slice(0, k).map(([d, i]) => ({ i, d: Math.sqrt(d) }));
}

// Leave-one-out 5-nearest-neighbour accuracy.
export function knnAccOf(X, labels, k = 5) {
  const K = Math.max(...labels) + 1;
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const nb = X.map((a, j) => [j === i ? Infinity : d2(a, X[i]), j]).sort((p, q) => p[0] - q[0]).slice(0, k);
    const c = new Array(K).fill(0);
    nb.forEach(([, j]) => c[labels[j]]++);
    if (c.indexOf(Math.max(...c)) === labels[i]) correct++;
  }
  return correct / X.length;
}

// Mean distance between class centroids over mean distance of an item to its own centroid.
export function separationOf(X, labels) {
  const K = Math.max(...labels) + 1, D = X[0].length;
  const cent = Array.from({ length: K }, () => new Float64Array(D));
  const n = new Array(K).fill(0);
  X.forEach((x, i) => { n[labels[i]]++; for (let j = 0; j < D; j++) cent[labels[i]][j] += x[j]; });
  cent.forEach((c, d) => { if (n[d]) for (let j = 0; j < D; j++) c[j] /= n[d]; });
  let within = 0;
  X.forEach((x, i) => { within += Math.sqrt(d2(x, cent[labels[i]])); });
  within /= X.length;
  let between = 0, c = 0;
  for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) if (n[a] && n[b]) { between += Math.sqrt(d2(cent[a], cent[b])); c++; }
  return between / c / within;
}

// ---------- the Tour's versions, over its bundled held-out digits ----------
const cache = new Map();
export function pca(layer) {
  if (!cache.has(layer)) cache.set(layer, pcaOf(ACTS[layer]));
  return cache.get(layer);
}
export function neighbours(layer, x, k = 5) {
  return neighboursOf(ACTS[layer], x, k).map((n) => ({ ...n, label: LABELS[n.i] }));
}
const knnCache = new Map();
export function knnAccuracy(layer, k = 5) {
  if (!knnCache.has(layer)) knnCache.set(layer, knnAccOf(ACTS[layer], LABELS, k));
  return knnCache.get(layer);
}
const sepCache = new Map();
export function separation(layer) {
  if (!sepCache.has(layer)) sepCache.set(layer, separationOf(ACTS[layer], LABELS));
  return sepCache.get(layer);
}
