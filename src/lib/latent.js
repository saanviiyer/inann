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

const cache = new Map();

// Top-3 principal components by power iteration on X^T X, never forming the D x D matrix.
export function pca(layer) {
  if (cache.has(layer)) return cache.get(layer);
  const X = ACTS[layer], N = X.length, D = X[0].length;
  const mean = new Float64Array(D);
  for (const x of X) for (let j = 0; j < D; j++) mean[j] += x[j] / N;
  const C = X.map((x) => Float64Array.from(x, (v, j) => v - mean[j]));
  let total = 0;
  for (const c of C) for (const v of c) total += v * v;
  const comps = [], vars = [];
  for (let k = 0; k < 3; k++) {
    let v = Float64Array.from({ length: D }, (_, j) => Math.sin(1 + j * (k + 1.3)));
    let lam = 0;
    for (let it = 0; it < 200; it++) {
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
      lam = Math.hypot(...w);
      if (!lam) break;
      v = w.map((x) => x / lam);
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
  const res = { project, pts, scale, explained: vars.map((v) => v / (total || 1)), D };
  cache.set(layer, res);
  return res;
}

const dist2 = (a, b) => {
  let s = 0;
  for (let j = 0; j < a.length; j++) s += (a[j] - b[j]) ** 2;
  return s;
};

// Nearest held-out digits in the full layer space (not the 3D projection).
export function neighbours(layer, x, k = 5) {
  return ACTS[layer].map((a, i) => [dist2(a, x), i]).sort((p, q) => p[0] - q[0]).slice(0, k).map(([d, i]) => ({ i, d: Math.sqrt(d), label: LABELS[i] }));
}

// Leave-one-out 5-nearest-neighbour accuracy: how well digits of the same class cluster
// in this layer's full space. One number per layer, computed once.
const knnCache = new Map();
export function knnAccuracy(layer, k = 5) {
  if (knnCache.has(layer)) return knnCache.get(layer);
  const X = ACTS[layer];
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const nb = X.map((a, j) => [j === i ? Infinity : dist2(a, X[i]), j]).sort((p, q) => p[0] - q[0]).slice(0, k);
    const c = new Array(10).fill(0);
    nb.forEach(([, j]) => c[LABELS[j]]++);
    if (c.indexOf(Math.max(...c)) === LABELS[i]) correct++;
  }
  const acc = correct / X.length;
  knnCache.set(layer, acc);
  return acc;
}

// Mean distance between class centroids divided by the mean distance of a digit to its own
// class centroid. Scale-free, so layers of different sizes compare. Higher means same digits
// sit tight relative to how far apart different digits are.
const sepCache = new Map();
export function separation(layer) {
  if (sepCache.has(layer)) return sepCache.get(layer);
  const X = ACTS[layer], D = X[0].length;
  const cent = Array.from({ length: 10 }, () => new Float64Array(D));
  const n = new Array(10).fill(0);
  X.forEach((x, i) => { n[LABELS[i]]++; for (let j = 0; j < D; j++) cent[LABELS[i]][j] += x[j]; });
  cent.forEach((c, d) => { for (let j = 0; j < D; j++) c[j] /= n[d]; });
  let within = 0;
  X.forEach((x, i) => { within += Math.sqrt(dist2(x, cent[LABELS[i]])); });
  within /= X.length;
  let between = 0, c = 0;
  for (let a = 0; a < 10; a++) for (let b = a + 1; b < 10; b++) { between += Math.sqrt(dist2(cent[a], cent[b])); c++; }
  const r = between / c / within;
  sepCache.set(layer, r);
  return r;
}
