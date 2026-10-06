// Analyses the UI shows, all computed from the bundled model and held-out digits.
import { net, ACTS, DIGITS, LABELS } from './model.js';
import { runActivationMax, receptiveField } from './net.js';

const fvCache = new Map();
export function featureVis(layer, index) {
  const key = `${layer}:${index}`;
  if (!fvCache.has(key)) fvCache.set(key, runActivationMax(net, layer, index).x);
  return fvCache.get(key);
}

// Bounding box of a receptive field mask.
export function rfBox(mask) {
  let x0 = 8, y0 = 8, x1 = -1, y1 = -1;
  for (let i = 0; i < 64; i++)
    if (mask[i]) {
      const x = i % 8, y = Math.floor(i / 8);
      x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
    }
  return { x0, y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

export function crop(img, b) {
  const out = new Float32Array(b.w * b.h);
  for (let y = 0; y < b.h; y++) for (let x = 0; x < b.w; x++) out[y * b.w + x] = img[(b.y0 + y) * 8 + b.x0 + x];
  return out;
}

// The held-out digits that drive one neuron hardest.
export function topImages(layer, index, k = 5) {
  const scored = ACTS[layer].map((a, i) => [a[index], i]).sort((a, b) => b[0] - a[0]);
  return scored.slice(0, k).map(([act, i]) => ({ i, act, label: LABELS[i], img: DIGITS[i] }));
}

export function topPatches(layer, index, k = 5) {
  const box = rfBox(receptiveField(layer, index));
  return topImages(layer, index, k).map((t) => ({ ...t, patch: crop(t.img, box), w: box.w, h: box.h }));
}

export function labelCounts(layer, index, k = 20) {
  const c = new Array(10).fill(0);
  for (const t of topImages(layer, index, k)) if (t.act > 0) c[t.label]++;
  return c;
}

// Units of a layer for the neighborhood map. Conv layers: one unit per channel, whose response
// to an image is the channel's mean activation. Dense: one unit per neuron.
export function unitVectors(layer) {
  const per = layer === 'conv1' ? 64 : layer === 'conv2' ? 16 : 1;
  const nUnits = layer === 'conv1' ? 6 : layer === 'conv2' ? 12 : 32;
  const vecs = Array.from({ length: nUnits }, () => new Float32Array(ACTS[layer].length));
  ACTS[layer].forEach((a, img) => {
    for (let u = 0; u < nUnits; u++) {
      let s = 0;
      for (let p = 0; p < per; p++) s += a[u * per + p];
      vecs[u][img] = s / per;
    }
  });
  return vecs;
}

export function cosineMatrix(vecs) {
  const n = vecs.length, norms = vecs.map((v) => Math.hypot(...v));
  const M = Array.from({ length: n }, () => new Float32Array(n));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      if (!norms[i] || !norms[j]) { M[i][j] = NaN; continue; }
      let s = 0;
      for (let t = 0; t < vecs[i].length; t++) s += vecs[i][t] * vecs[j][t];
      M[i][j] = s / (norms[i] * norms[j]);
    }
  return { M, dead: norms.map((v) => v === 0) };
}

// Classical MDS on distance 1 - cosine, top two eigenvectors by power iteration.
export function mds2(M, alive) {
  const idx = alive.map((a, i) => (a ? i : -1)).filter((i) => i >= 0);
  const n = idx.length;
  const D2 = idx.map((i) => idx.map((j) => (1 - M[i][j]) ** 2));
  const rm = D2.map((r) => r.reduce((a, b) => a + b, 0) / n);
  const gm = rm.reduce((a, b) => a + b, 0) / n;
  const B = D2.map((r, i) => r.map((v, j) => -0.5 * (v - rm[i] - rm[j] + gm)));
  const eig = [];
  for (let k = 0; k < 2; k++) {
    let v = Array.from({ length: n }, (_, i) => Math.sin(i + 1 + k));
    let lam = 0;
    for (let it = 0; it < 300; it++) {
      const w = B.map((r) => r.reduce((s, b, j) => s + b * v[j], 0));
      for (const e of eig) {
        const d = w.reduce((s, x, i) => s + x * e.v[i], 0);
        for (let i = 0; i < n; i++) w[i] -= d * e.v[i];
      }
      lam = Math.hypot(...w);
      if (!lam) break;
      v = w.map((x) => x / lam);
    }
    eig.push({ v, lam });
  }
  const pos = {};
  idx.forEach((u, i) => (pos[u] = [eig[0].v[i] * Math.sqrt(eig[0].lam), eig[1].v[i] * Math.sqrt(eig[1].lam)]));
  return pos;
}
